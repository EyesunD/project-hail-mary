"""Combined-book diagnostic engine + HTML report.

Functions consume a collection of :class:`Portfolio` objects and produce:

- :func:`combined_exposure` — sector / region / asset-class breakdown across
  ``HOLDING``-tagged portfolios
- :func:`correlation_matrix` — pairwise correlation across all portfolios
- :func:`redundancy_pairs` — pairs above a configurable threshold (PROTECTED
  excluded from candidate side)
- :func:`risk_contribution` — by-holding and by-portfolio risk contribution to
  total book volatility (covariance computed inline; see CLAUDE.md note —
  ``analytics/risk.py`` referenced in spec is missing on disk)
- :func:`benchmark_comparison` — Sharpe / max-DD / annualised-vol with deltas
  vs each ``MANAGED_BENCHMARK`` portfolio
- :func:`render_html_report` — self-contained HTML rolling all sections up

All functions consuming return series accept optional ``(start, end)`` per
design ``D7`` to keep regime-conditional analysis (Phase 3) cheap.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from hailmary.allocation.portfolios import Portfolio, Role
from hailmary.allocation.returns import _CASH_ANNUAL_YIELDS, portfolio_returns
from hailmary.analytics.metrics import PerformanceMetrics
from hailmary.viz.theme import PALETTE, apply_theme

if TYPE_CHECKING:
    pass


_TRADING_DAYS = 252


def book_common_history_start(
    portfolios: Sequence[Portfolio],
    *,
    price_source: Any | None = None,
    returns: pd.DataFrame | None = None,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    fx_series_usd_sgd: pd.Series | None = None,
) -> date | None:
    """Latest first-available date across ``HOLDING``-tagged portfolios.

    Use this to align the diagnostic window so every portfolio is comparable on
    the same dates. Returns ``None`` if no holding portfolios produce a series.
    """
    holdings_books = [p for p in portfolios if Role.HOLDING in p.roles]
    if not holdings_books:
        return None
    panel = _build_returns_panel(
        holdings_books,
        price_source=price_source,
        returns=returns,
        start=start,
        end=end,
        fx_series_usd_sgd=fx_series_usd_sgd,
    )
    if panel.empty:
        return None
    first_dates: list[pd.Timestamp] = []
    for col in panel.columns:
        s = panel[col].dropna()
        if s.empty:
            continue
        first_dates.append(s.index.min())
    if not first_dates:
        return None
    return max(first_dates).date()


class PortfolioDroppedError(Exception):
    """Raised when one or more portfolios would be silently dropped from the diagnostic.

    Common causes: ticker not in the universe map → no Yahoo fetch → no data;
    every holding's ticker missing from the supplied returns DataFrame; weights
    sum to zero (e.g. all holdings collapse to a duplicate ticker that was dropped).
    """

    def __init__(self, dropped: list[tuple[str, str]]) -> None:
        self.dropped = dropped
        body = "\n  - ".join(f"{name}: {reason}" for name, reason in dropped)
        super().__init__(
            f"Diagnostic would silently drop {len(dropped)} portfolio(s):\n  - {body}"
        )

_WINDOWS: dict[str, int | None] = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "All": None,
}


def _fmt_compact(value: float) -> str:
    """Format 6500 → '6.5K', 1_200_000 → '1.2M', 3.2e9 → '3.2B'."""
    if value is None or pd.isna(value):
        return "—"
    av = abs(value)
    sign = "-" if value < 0 else ""
    if av >= 1e9:
        return f"{sign}{av / 1e9:.2f}B"
    if av >= 1e6:
        return f"{sign}{av / 1e6:.2f}M"
    if av >= 1e3:
        return f"{sign}{av / 1e3:.1f}K"
    return f"{sign}{av:,.0f}"


def _fmt_compact_signed(value: float) -> str:
    """Same as `_fmt_compact` but always shows leading sign for non-zero values."""
    if value is None or pd.isna(value):
        return "—"
    if value == 0:
        return "0"
    formatted = _fmt_compact(value)
    return formatted if formatted.startswith("-") else f"+{formatted}"


def _fmt_compact_precise(value: float) -> str:
    """Compact $ formatter with one more decimal than ``_fmt_compact`` (e.g.
    40,510 → '40.51K', 1,234,567 → '1.235M'). Use where ~$50 precision on
    K-scale values matters (e.g. reconciliation rows you compare to the app)."""
    if value is None or pd.isna(value):
        return "—"
    av = abs(value)
    sign = "-" if value < 0 else ""
    if av >= 1e9:
        return f"{sign}{av / 1e9:.3f}B"
    if av >= 1e6:
        return f"{sign}{av / 1e6:.3f}M"
    if av >= 1e3:
        return f"{sign}{av / 1e3:.2f}K"
    return f"{sign}{av:,.0f}"


def _fmt_compact_precise_signed(value: float) -> str:
    """Signed variant of ``_fmt_compact_precise``."""
    if value is None or pd.isna(value):
        return "—"
    if value == 0:
        return "0"
    formatted = _fmt_compact_precise(value)
    return formatted if formatted.startswith("-") else f"+{formatted}"


# ---------------------------------------------------------------------------
# Conditional-formatting cell stylers (return CSS strings for pandas.Styler)
# ---------------------------------------------------------------------------


def _style_pos_neg(val: float) -> str:
    if pd.isna(val) or val == 0:
        return ""
    if val > 0:
        return "background-color: rgba(63, 185, 80, 0.18); color: #3fb950;"
    return "background-color: rgba(248, 81, 73, 0.18); color: #f85149;"


def _style_sharpe(val: float) -> str:
    if pd.isna(val):
        return ""
    if val >= 1.0:
        a = min(0.42, 0.18 + (val - 1.0) * 0.10)
        return f"background-color: rgba(63, 185, 80, {a:.2f}); color: #3fb950;"
    if val > 0:
        return "background-color: rgba(63, 185, 80, 0.08);"
    return "background-color: rgba(248, 81, 73, 0.20); color: #f85149;"


def _style_dd(val: float) -> str:
    if pd.isna(val) or val == 0:
        return ""
    intensity = min(abs(val) / 0.50, 1.0)
    a = 0.10 + 0.30 * intensity
    return f"background-color: rgba(248, 81, 73, {a:.2f}); color: #f85149;"


def _style_rho(val: float) -> str:
    if pd.isna(val):
        return ""
    a = max(0.18, min(0.55, 0.18 + (val - 0.70) * 0.90))
    return f"background-color: rgba(248, 81, 73, {a:.2f}); color: #f85149;"


def _style_pct_total(val: float) -> str:
    if pd.isna(val) or val <= 0:
        return ""
    a = min(0.45, val * 1.20)
    return f"background-color: rgba(255, 166, 87, {a:.2f}); color: #ffa657;"


def _style_dd_delta(val: float) -> str:
    """Drawdown delta: positive = less DD = good (green); negative = more DD = bad (red)."""
    return _style_pos_neg(val)


# ---------------------------------------------------------------------------
# Combined exposure
# ---------------------------------------------------------------------------


def combined_exposure(portfolios: Sequence[Portfolio]) -> dict[str, pd.DataFrame]:
    """Sector / region / asset-class breakdowns across all ``HOLDING`` portfolios.

    Returns a dict ``{"asset_class": df, "region": df, "sector": df}`` where each
    DataFrame has columns ``[bucket, value, weight]`` summed across portfolios
    without double-counting.
    """
    holdings_books = [p for p in portfolios if Role.HOLDING in p.roles]
    rows: list[dict[str, Any]] = []
    for p in holdings_books:
        for h in p.holdings:
            rows.append(
                {
                    "ticker": h.metadata.ticker,
                    "value": h.weight * p.total_value,
                    "asset_class": h.metadata.asset_class,
                    "region": h.metadata.region,
                    "sector": h.metadata.sector or "Unspecified",
                }
            )
    if not rows:
        empty = pd.DataFrame(columns=["bucket", "value", "weight"])
        return {"asset_class": empty.copy(), "region": empty.copy(), "sector": empty.copy()}

    df = pd.DataFrame(rows)
    total = df["value"].sum()
    out = {}
    for dim in ("asset_class", "region", "sector"):
        agg = (
            df.groupby(dim, as_index=False)["value"]
            .sum()
            .rename(columns={dim: "bucket"})
        )
        agg["weight"] = agg["value"] / total
        out[dim] = agg.sort_values("value", ascending=False).reset_index(drop=True)
    return out


def single_exposure_donut(df: pd.DataFrame, *, title: str) -> go.Figure:
    """Compact single-dimension donut (no legend; bucket labels carried by the table beside it)."""
    fig = go.Figure(
        go.Pie(
            labels=df["bucket"],
            values=df["value"],
            hole=0.55,
            textinfo="percent",
            textposition="inside",
            insidetextorientation="horizontal",
            hovertemplate="%{label}<br>$%{value:,.0f} (%{percent})<extra></extra>",
            sort=True,
            direction="clockwise",
            showlegend=False,
        )
    )
    fig = apply_theme(fig, title=title, height=260)
    fig.update_layout(margin={"l": 10, "r": 10, "t": 44, "b": 10})
    return fig


def combined_exposure_figure(exposure: dict[str, pd.DataFrame]) -> go.Figure:
    """1×3 donut chart with grouped legends — one ring per exposure dimension."""
    dims = [("asset_class", "Asset Class"), ("region", "Region"), ("sector", "Sector")]
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "domain"}] * 3],
        subplot_titles=[label for _, label in dims],
        horizontal_spacing=0.02,
    )
    for i, (key, label) in enumerate(dims, start=1):
        df = exposure[key]
        fig.add_trace(
            go.Pie(
                labels=df["bucket"],
                values=df["value"],
                hole=0.55,
                textinfo="percent",
                textposition="inside",
                insidetextorientation="horizontal",
                hovertemplate="%{label}<br>%{value:$,.0f} (%{percent})<extra></extra>",
                showlegend=True,
                legendgroup=key,
                legendgrouptitle_text=label,
                sort=True,
                direction="clockwise",
            ),
            row=1,
            col=i,
        )
    fig = apply_theme(fig, title="Combined-book exposure", height=520)
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.05,
            "xanchor": "center",
            "x": 0.5,
            "groupclick": "toggleitem",
            "bgcolor": PALETTE["surface"],
            "bordercolor": PALETTE["border"],
            "borderwidth": 1,
            "font": {"size": 11},
        },
    )
    return fig


# ---------------------------------------------------------------------------
# Whole-book performance
# ---------------------------------------------------------------------------


def book_performance(
    portfolios: Sequence[Portfolio],
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    price_source: Any | None = None,
    returns: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
    fx_rate_usd_sgd: float | None = None,
    fx_series_usd_sgd: pd.Series | None = None,
    align_window: bool = True,
) -> dict[str, Any]:
    """Whole-book metrics + NAV series (current-snapshot reconstruction).

    Combines per-portfolio return series with weights ∝ ``total_value`` across
    ``HOLDING``-tagged portfolios. Weights are re-normalised per timestep over
    the portfolios that have data on that date, so a recently-launched holding
    in one portfolio doesn't truncate the whole book's history.

    Returns a dict::

        {
            "aum": float,             # total HOLDING-tagged $ (in source currency)
            "ann_return": float,      # annualised return
            "ann_vol": float,         # annualised vol
            "sharpe": float,
            "max_dd": float,
            "nav": pd.Series,         # daily NAV indexed to 100 at first date
            "n_days": int,
        }
    """
    holdings_books = [p for p in portfolios if Role.HOLDING in p.roles]
    if fx_rate_usd_sgd is None:
        # Prefer the statement-date FX rate parsed from the Stashaway PDF
        # (Singapore-EOD, matches the app exactly). Fall back to Yahoo's spot,
        # then to 1.0 with a warning.
        statement_fx_rates = [
            p.metadata.get("statement_fx_usd_sgd")
            for p in holdings_books
            if p.metadata.get("statement_fx_usd_sgd") is not None
        ]
        if statement_fx_rates:
            fx_rate_usd_sgd = float(statement_fx_rates[0])
        elif fx_series_usd_sgd is not None and not fx_series_usd_sgd.empty:
            fx_rate_usd_sgd = float(fx_series_usd_sgd.dropna().iloc[-1])
        else:
            fx_rate_usd_sgd = 1.0
            if any(p.currency.upper() == "USD" for p in holdings_books):
                warnings.warn(
                    "fx_rate_usd_sgd not supplied; USD-reported portfolios are summed "
                    "into AUM as if 1 USD = 1 SGD. Pass an FX rate for an SGD-denominated total.",
                    stacklevel=2,
                )

    def _to_sgd(value: float, ccy: str) -> float:
        return value * fx_rate_usd_sgd if ccy.upper() == "USD" else value

    if not holdings_books:
        return {
            "aum": 0.0,
            "currency": "SGD",
            "fx_rate_usd_sgd": fx_rate_usd_sgd,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "nav": pd.Series(dtype=float),
            "n_days": 0,
        }

    panel = _build_returns_panel(
        holdings_books,
        price_source=price_source,
        returns=returns,
        start=start,
        end=end,
        fx_series_usd_sgd=fx_series_usd_sgd,
    )
    aum = sum(_to_sgd(p.total_value, p.currency) for p in holdings_books)
    if panel.empty:
        return {
            "aum": aum,
            "currency": "SGD",
            "fx_rate_usd_sgd": fx_rate_usd_sgd,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "nav": pd.Series(dtype=float),
            "n_days": 0,
        }

    weights = pd.Series({
        p.name: _to_sgd(p.total_value, p.currency)
        for p in holdings_books
        if p.name in panel.columns
    })
    weights = weights / weights.sum()
    panel = panel[weights.index]

    if align_window:
        # Apples-to-apples: align every column to the common window so weights
        # are static (no per-day renormalisation, no "early book ≠ late book"
        # mixing). Shorter history, cleaner numbers.
        common_start = max(panel[c].dropna().index.min() for c in panel.columns)
        aligned_panel = panel.loc[common_start:].dropna(how="any")
        if aligned_panel.empty:
            book_return = pd.Series(dtype=float)
        else:
            book_return = (aligned_panel * weights).sum(axis=1)
    else:
        # Full history with dynamic per-timestep weight renormalisation —
        # early dates use only the older portfolios at boosted weights.
        # Longer history but the early "book" isn't the same as today's book.
        available = panel.notna().astype(float)
        eff_weights = available.mul(weights, axis=1)
        row_sums = eff_weights.sum(axis=1).replace(0.0, np.nan)
        eff_weights = eff_weights.div(row_sums, axis=0)
        book_return = (panel.fillna(0.0) * eff_weights).sum(axis=1).dropna()

    if book_return.empty:
        return {
            "aum": aum,
            "currency": "SGD",
            "fx_rate_usd_sgd": fx_rate_usd_sgd,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "nav": pd.Series(dtype=float),
            "n_days": 0,
        }

    metrics = PerformanceMetrics(book_return, risk_free_rate=risk_free_rate)
    nav = (1.0 + book_return).cumprod() * 100.0
    return {
        "aum": float(aum),
        "currency": "SGD",
        "fx_rate_usd_sgd": float(fx_rate_usd_sgd),
        "ann_return": float(metrics.annualised_return),
        "ann_vol": float(metrics.annualised_vol),
        "sharpe": float(metrics.sharpe),
        "max_dd": float(metrics.max_drawdown),
        "nav": nav,
        "n_days": len(book_return),
        "windowed": _windowed_metrics(book_return, risk_free_rate=risk_free_rate),
    }


def _windowed_metrics(returns: pd.Series, *, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """Per-window cumulative return + annualised metrics over standard lookbacks
    plus one row per calendar year that overlaps the series.

    Columns: ``period, cum_return, ann_return, ann_vol, sharpe, max_dd, days``.
    ``cum_return`` is the simple cumulative return over the window; every other
    return/vol number is annualised so windows are directly comparable.
    """
    def _row(label: str, sub: pd.Series) -> dict[str, Any]:
        if len(sub) < 5:
            return {
                "period": label,
                "cum_return": float("nan"),
                "ann_return": float("nan"),
                "ann_vol": float("nan"),
                "sharpe": float("nan"),
                "max_dd": float("nan"),
                "days": int(len(sub)),
            }
        m = PerformanceMetrics(sub, risk_free_rate=risk_free_rate)
        return {
            "period": label,
            "cum_return": float(m.total_return),
            "ann_return": float(m.annualised_return),
            "ann_vol": float(m.annualised_vol),
            "sharpe": float(m.sharpe),
            "max_dd": float(m.max_drawdown),
            "days": int(len(sub)),
        }

    rows: list[dict[str, Any]] = []
    for label, n_days in _WINDOWS.items():
        sub = returns if n_days is None else returns.tail(n_days)
        rows.append(_row(label, sub))

    if not returns.empty:
        first_year = returns.index.min().year
        last_year = returns.index.max().year
        for year in range(first_year, last_year + 1):
            year_sub = returns[returns.index.year == year]
            if year_sub.empty:
                continue
            jan1 = pd.Timestamp(year=year, month=1, day=1)
            dec31 = pd.Timestamp(year=year, month=12, day=31)
            is_partial = year_sub.index.min() > jan1 + pd.Timedelta(days=7) or (
                year == last_year and year_sub.index.max() < dec31 - pd.Timedelta(days=7)
            )
            label = f"{year} (partial)" if is_partial else str(year)
            rows.append(_row(label, year_sub))

    return pd.DataFrame(rows)


def equity_curve_figure(nav: pd.Series) -> go.Figure:
    """Line chart of the whole-book NAV series (indexed to 100)."""
    fig = go.Figure()
    if not nav.empty:
        fig.add_trace(
            go.Scatter(
                x=nav.index,
                y=nav.values,
                mode="lines",
                line={"color": PALETTE["accent_blue"], "width": 2},
                hovertemplate="%{x|%Y-%m-%d}<br>NAV %{y:.1f}<extra></extra>",
                name="Book NAV",
            )
        )
    fig = apply_theme(fig, title="Combined-book NAV (statement-date weights, indexed to 100)", height=380)
    fig.update_layout(yaxis_title="NAV", showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Per-portfolio reconciliation (statement-date → today)
# ---------------------------------------------------------------------------


def holdings_reconciliation(
    portfolio: Portfolio,
    *,
    end: date | datetime,
    price_source: Any | None = None,
    fx_series_usd_sgd: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-holding statement→today reconciliation for one portfolio.

    Returns a DataFrame with columns:
    ``ticker, label, weight, stmt_value, stmt_price, today_price, return,
    today_value, delta``. All values in the portfolio's native currency
    (USD or SGD); for USD portfolios with ``fx_series_usd_sgd`` supplied,
    also returns ``stmt_value_sgd / today_value_sgd / delta_sgd``.

    Used by the per-portfolio drill-down section in the Phase 1 HTML report —
    lets the user trace a sleeve's PnL back to each individual holding.
    """
    end_date = end.date() if isinstance(end, datetime) else end
    fetch_start = portfolio.statement_date - timedelta(days=7)
    symbols = sorted({
        h.metadata.ticker
        for h in portfolio.holdings
        if not h.metadata.ticker.startswith("CASH_")
    })
    bars = None
    if symbols and price_source is not None:
        try:
            bars = price_source.get_bars(symbols, fetch_start, end_date)
        except Exception as exc:  # pragma: no cover — provider failure
            warnings.warn(f"Could not fetch price bars: {exc}", stacklevel=2)

    def _close_asof(
        ticker: str, target: date
    ) -> tuple[float | None, date | None]:
        """Return (close, actual_date_used) at or before *target*. Forward-fill
        semantics: if a ticker hasn't traded today, returns the most recent
        prior close + that date so the caller knows the data is N days stale.
        """
        if bars is None or ticker not in bars.index.get_level_values(0):
            return None, None
        ser = bars.xs(ticker, level=0)["close"]
        if ser.empty:
            return None, None
        target_ts = (
            pd.Timestamp(target).tz_localize(ser.index.tz)
            if ser.index.tz
            else pd.Timestamp(target)
        )
        ser = ser[ser.index <= target_ts]
        if ser.empty:
            return None, None
        last_idx = ser.index[-1]
        actual_date = last_idx.date() if hasattr(last_idx, "date") else last_idx
        return float(ser.iloc[-1]), actual_date

    today_spot = (
        float(fx_series_usd_sgd.dropna().iloc[-1])
        if fx_series_usd_sgd is not None and not fx_series_usd_sgd.empty
        else None
    )
    stmt_fx = portfolio.metadata.get("statement_fx_usd_sgd")
    stmt_fx_value = float(stmt_fx) if stmt_fx is not None else None
    is_usd = portfolio.currency.upper() == "USD"

    def _is_cash_like(asset_class: str | None, sector: str | None) -> bool:
        """Holdings that can be safely forward-accrued when Yahoo data lags
        the end_date — only assets where directional risk is negligible
        (money market funds, cash equivalents, short-duration T-bills)."""
        ac = (asset_class or "").lower()
        sec = (sector or "").lower()
        if ac == "cash":
            return True
        if "money market" in sec or "treasury 0-3m" in sec:
            return True
        return False

    rows: list[dict[str, Any]] = []
    for h in portfolio.holdings:
        t = h.metadata.ticker
        stmt_value = h.weight * portfolio.total_value
        label = f"{h.metadata.asset_class} · {h.metadata.region}"
        accrued_days = 0
        if t.startswith("CASH_"):
            days = max((end_date - portfolio.statement_date).days, 0)
            yield_pa = _CASH_ANNUAL_YIELDS.get(t, 0.0)
            ret = (1 + yield_pa) ** (days / 365.0) - 1
            stmt_price = today_price = float("nan")
            stmt_date_used = portfolio.statement_date
            today_date_used = end_date
        else:
            stmt_price, stmt_date_used = _close_asof(t, portfolio.statement_date)
            today_price, today_date_used = _close_asof(t, end_date)
            if stmt_price is None or today_price is None or stmt_price <= 0:
                ret = 0.0
                stmt_price = float("nan") if stmt_price is None else stmt_price
                today_price = float("nan") if today_price is None else today_price
            else:
                ret = today_price / stmt_price - 1.0
                # Forward-accrue stale low-vol prices: if Yahoo data is
                # behind end_date for a cash/MMF holding, project the
                # missing days using the holding's own realised yield.
                if (
                    today_date_used is not None
                    and stmt_date_used is not None
                    and today_date_used < end_date
                    and _is_cash_like(h.metadata.asset_class, h.metadata.sector)
                ):
                    in_data_days = max((today_date_used - stmt_date_used).days, 1)
                    missing_days = (end_date - today_date_used).days
                    if missing_days > 0 and in_data_days > 0:
                        ann_yield = (today_price / stmt_price) ** (
                            365.0 / in_data_days
                        ) - 1.0
                        accrual = (1.0 + ann_yield) ** (
                            missing_days / 365.0
                        ) - 1.0
                        today_price = today_price * (1.0 + accrual)
                        today_date_used = end_date
                        ret = today_price / stmt_price - 1.0
                        accrued_days = missing_days
        today_value = stmt_value * (1.0 + ret)
        row = {
            "ticker": t,
            "label": label,
            "weight": h.weight,
            "stmt_value": stmt_value,
            "stmt_price": stmt_price,
            "stmt_date_used": stmt_date_used,
            "today_price": today_price,
            "today_date_used": today_date_used,
            "accrued_days": accrued_days,
            "return": ret,
            "today_value": today_value,
            "delta": today_value - stmt_value,
        }
        if is_usd and today_spot and stmt_fx_value:
            row["stmt_value_sgd"] = stmt_value * stmt_fx_value
            row["today_value_sgd"] = today_value * today_spot
            row["delta_sgd"] = row["today_value_sgd"] - row["stmt_value_sgd"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("stmt_value", ascending=False).reset_index(drop=True)


def portfolio_reconciliation(
    portfolios: Sequence[Portfolio],
    *,
    end: date | datetime,
    price_source: Any | None = None,
    returns: pd.DataFrame | None = None,  # kept for API compat — not used
    fx_series_usd_sgd: pd.Series | None = None,
) -> pd.DataFrame:
    """Project each portfolio's value from its statement date to ``end``.

    For every ``HOLDING``-tagged portfolio, returns a row with:
    ``currency`` · ``stmt_native`` · ``stmt_sgd`` · ``today_native`` · ``today_sgd``
    · ``delta_sgd`` · ``return_native`` · ``return_sgd``.

    SGD values are mark-to-market: ``stmt_sgd = stmt_native × statement-date FX
    (from PDF)``, ``today_sgd = today_native × today's spot FX``. This matches
    what Stashaway's app displays.

    Computes ``today_native`` per-holding as
    ``Σᵢ wᵢ × stmt_value × (price_end_i / price_stmt_i)``. We deliberately do
    NOT compound a daily portfolio_returns series here — that approach drops
    every date where ANY holding has NaN (intersection of US/crypto/LSE/SGX
    trading calendars), losing ~30% of PnL for sleeves that mix asset classes.
    Per-holding point-to-point only needs 2 prices per ticker and is robust
    against trading-calendar mismatches.
    """
    holdings_books = [p for p in portfolios if Role.HOLDING in p.roles]
    today_spot = (
        float(fx_series_usd_sgd.dropna().iloc[-1])
        if fx_series_usd_sgd is not None and not fx_series_usd_sgd.empty
        else 1.0
    )

    # Batch-fetch all non-cash tickers' price bars once across all portfolios
    # so we don't re-hit the provider per holding.
    end_date = end.date() if isinstance(end, datetime) else end
    stmt_dates = {p.statement_date for p in holdings_books}
    earliest_start = min(stmt_dates) if stmt_dates else end_date
    # Pad the start by a few business days so the asof lookup always has data
    fetch_start = earliest_start - timedelta(days=7)
    all_symbols = sorted({
        h.metadata.ticker
        for p in holdings_books
        for h in p.holdings
        if not h.metadata.ticker.startswith("CASH_")
    })
    bars: pd.DataFrame | None = None
    if all_symbols and price_source is not None:
        try:
            bars = price_source.get_bars(all_symbols, fetch_start, end_date)
        except Exception as exc:  # pragma: no cover — provider failure
            warnings.warn(f"Could not fetch price bars: {exc}", stacklevel=2)
            bars = None

    def _close_asof_with_date(
        ticker: str, target: date
    ) -> tuple[float | None, date | None]:
        """Most-recent close at or before *target* (forward-fill semantics)."""
        if bars is None or ticker not in bars.index.get_level_values(0):
            return None, None
        ser = bars.xs(ticker, level=0)["close"]
        if ser.empty:
            return None, None
        target_ts = pd.Timestamp(target).tz_localize(ser.index.tz) if ser.index.tz else pd.Timestamp(target)
        ser = ser[ser.index <= target_ts]
        if ser.empty:
            return None, None
        last_idx = ser.index[-1]
        actual = last_idx.date() if hasattr(last_idx, "date") else last_idx
        return float(ser.iloc[-1]), actual

    def _is_cash_like(asset_class: str | None, sector: str | None) -> bool:
        ac = (asset_class or "").lower()
        sec = (sector or "").lower()
        return ac == "cash" or "money market" in sec or "treasury 0-3m" in sec

    rows: list[dict[str, Any]] = []
    for p in holdings_books:
        today_native = 0.0
        for h in p.holdings:
            stmt_value = h.weight * p.total_value
            t = h.metadata.ticker
            if t.startswith("CASH_"):
                # Synthetic cash: apply the annual yield prorated by elapsed
                # calendar days — vol is irrelevant for point-to-point.
                days = max((end_date - p.statement_date).days, 0)
                yield_pa = _CASH_ANNUAL_YIELDS.get(t, 0.0)
                holding_ret = (1 + yield_pa) ** (days / 365.0) - 1
                today_native += stmt_value * (1 + holding_ret)
                continue
            p_stmt, d_stmt = _close_asof_with_date(t, p.statement_date)
            p_end, d_end = _close_asof_with_date(t, end_date)
            if p_stmt is None or p_end is None or p_stmt <= 0:
                today_native += stmt_value
                continue
            # Forward-accrue stale low-vol prices (cash / MMF / 0-3M Treasury)
            # using the holding's own realised yield over the available window.
            if (
                d_end is not None
                and d_stmt is not None
                and d_end < end_date
                and _is_cash_like(h.metadata.asset_class, h.metadata.sector)
            ):
                in_data_days = max((d_end - d_stmt).days, 1)
                missing_days = (end_date - d_end).days
                if missing_days > 0:
                    ann_yield = (p_end / p_stmt) ** (365.0 / in_data_days) - 1.0
                    accrual = (1.0 + ann_yield) ** (missing_days / 365.0) - 1.0
                    p_end = p_end * (1.0 + accrual)
            today_native += stmt_value * (p_end / p_stmt)

        # Apply management fee (prorated) — same convention as portfolio_returns
        fee_annual = float(p.metadata.get("management_fee_annual", 0.0))
        if fee_annual > 0:
            days = max((end_date - p.statement_date).days, 0)
            today_native *= (1.0 - fee_annual * days / 365.0)

        cum_native = (today_native / p.total_value) - 1.0 if p.total_value > 0 else 0.0

        stmt_fx = p.metadata.get("statement_fx_usd_sgd")
        if p.currency.upper() == "USD":
            stmt_fx_value = float(stmt_fx) if stmt_fx is not None else 1.0
            stmt_sgd = p.total_value * stmt_fx_value
            today_sgd = today_native * today_spot  # mark-to-market — matches app
        else:
            stmt_sgd = p.total_value
            today_sgd = today_native  # SGD-native: no FX conversion

        return_sgd = (today_sgd / stmt_sgd) - 1.0 if stmt_sgd > 0 else 0.0

        rows.append(
            {
                "portfolio": p.name,
                "currency": p.currency.upper(),
                "stmt_native": p.total_value,
                "stmt_sgd": stmt_sgd,
                "today_native": today_native,
                "today_sgd": today_sgd,
                "delta_sgd": today_sgd - stmt_sgd,
                "return_native": cum_native,
                "return_sgd": return_sgd,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Return-series helpers
# ---------------------------------------------------------------------------


def _build_returns_panel(
    portfolios: Sequence[Portfolio],
    *,
    price_source: Any | None = None,
    returns: pd.DataFrame | None = None,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    fx_series_usd_sgd: pd.Series | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """Compute one return series per portfolio and stack into a wide DataFrame.

    Falls back to the in-memory wide returns DataFrame if supplied; otherwise
    delegates to ``portfolio_returns(price_source=...)`` per portfolio. When
    ``fx_series_usd_sgd`` is supplied, USD-reported portfolios are converted
    to SGD per-day inside ``portfolio_returns``.

    When ``strict=True`` and any portfolio fails to produce a return series,
    raises :class:`PortfolioDroppedError` listing every drop. Otherwise emits a
    ``UserWarning`` per drop and continues with the surviving portfolios.
    """
    series = []
    dropped: list[tuple[str, str]] = []
    for p in portfolios:
        try:
            s = portfolio_returns(
                p,
                start=start,
                end=end,
                price_source=price_source,
                returns=returns,
                fx_series_usd_sgd=fx_series_usd_sgd,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            dropped.append((p.name, reason))
            warnings.warn(f"Skipping {p.name!r} in returns panel: {exc}", stacklevel=2)
            continue
        if s.isna().all():
            reason = "all-NaN return series (weights resolved to 0?)"
            dropped.append((p.name, reason))
            warnings.warn(f"Skipping {p.name!r}: {reason}", stacklevel=2)
            continue
        series.append(s)
    if dropped and strict:
        raise PortfolioDroppedError(dropped)
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def correlation_matrix(
    portfolios: Sequence[Portfolio],
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    price_source: Any | None = None,
    returns: pd.DataFrame | None = None,
    fx_series_usd_sgd: pd.Series | None = None,
) -> pd.DataFrame:
    """Pairwise return correlation across ``HOLDING``-tagged portfolios."""
    holdings_books = [p for p in portfolios if Role.HOLDING in p.roles]
    panel = _build_returns_panel(
        holdings_books,
        price_source=price_source,
        returns=returns,
        start=start,
        end=end,
        fx_series_usd_sgd=fx_series_usd_sgd,
    )
    if panel.empty:
        return pd.DataFrame()
    # Use pandas' default pairwise correlation (NaN-aware per pair) rather than
    # collapsing the panel to its common-history window — otherwise a single
    # young series (e.g. a recently-launched ETF) would cap every other pair's
    # window to its own short history.
    return panel.corr(min_periods=20)


def correlation_figure(corr: pd.DataFrame) -> go.Figure:
    n = len(corr)
    height = max(320, min(460, 60 + 30 * n))
    masked = corr.mask(np.eye(n, dtype=bool))

    off_diag = masked.values[~np.isnan(masked.values)]
    if off_diag.size:
        zmin = max(-1.0, float(np.nanmin(off_diag)) - 0.05)
        zmax = min(1.0, float(np.nanmax(off_diag)) + 0.05)
        if zmin < 0:
            zmin, zmax = -max(abs(zmin), abs(zmax)), max(abs(zmin), abs(zmax))
    else:
        zmin, zmax = -1.0, 1.0

    text_matrix = [
        ["" if i == j or pd.isna(corr.iloc[i, j]) else f"{corr.iloc[i, j]:.2f}" for j in range(n)]
        for i in range(n)
    ]

    fig = px.imshow(
        masked,
        zmin=zmin,
        zmax=zmax,
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    fig.update_traces(
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="%{y} × %{x}<br>ρ=%{z:.3f}<extra></extra>",
    )
    fig = apply_theme(fig, title="Portfolio correlation", height=height)
    fig.update_layout(
        margin={"l": 140, "r": 30, "t": 60, "b": 100},
        xaxis={"tickangle": -35, "tickfont": {"size": 11}},
        yaxis={"tickfont": {"size": 11}},
        coloraxis_colorbar={"thickness": 12, "len": 0.75},
    )
    return fig


# ---------------------------------------------------------------------------
# Redundancy
# ---------------------------------------------------------------------------


def redundancy_pairs(
    corr_matrix: pd.DataFrame,
    *,
    threshold: float = 0.85,
    portfolios: Sequence[Portfolio] | None = None,
) -> list[tuple[str, str, float, str]]:
    """Pairs whose correlation exceeds *threshold*.

    Returns tuples ``(name_a, name_b, rho, candidate)`` where *candidate* is
    the portfolio recommended for review (never a ``PROTECTED`` portfolio).
    Both names are still listed even if both are protected — only the
    *candidate* slot is constrained.
    """
    if corr_matrix.empty:
        return []
    protected: set[str] = set()
    if portfolios is not None:
        protected = {p.name for p in portfolios if Role.PROTECTED in p.roles}

    pairs: list[tuple[str, str, float, str]] = []
    cols = list(corr_matrix.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            rho_raw = corr_matrix.loc[a, b]
            if pd.isna(rho_raw):
                continue
            rho = float(rho_raw)
            if rho < threshold:
                continue
            if a in protected and b in protected:
                candidate = ""  # neither side eligible
            elif a in protected:
                candidate = b
            elif b in protected:
                candidate = a
            else:
                candidate = b  # arbitrary; surface higher-index name
            pairs.append((a, b, rho, candidate))
    return sorted(pairs, key=lambda t: t[2], reverse=True)


# ---------------------------------------------------------------------------
# Risk contribution
# ---------------------------------------------------------------------------


def risk_contribution(
    portfolios: Sequence[Portfolio],
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    price_source: Any | None = None,
    returns: pd.DataFrame | None = None,
    fx_rate_usd_sgd: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Risk contribution per holding and per portfolio for the combined book.

    Decomposes total book volatility ``σ_p = √(wᵀ Σ w)`` into marginal
    contributions ``MCᵢ = (Σ w)ᵢ / σ_p`` and component contributions
    ``CCᵢ = wᵢ · MCᵢ`` such that ``Σᵢ CCᵢ = σ_p``.

    Returns ``{"by_holding": df, "by_portfolio": df}`` where each frame has
    columns ``[name, weight, contribution, pct_total]``. Holdings appear once
    per (portfolio, ticker) tuple to keep ownership unambiguous when the same
    ticker is held across portfolios.
    """
    holdings_books = [p for p in portfolios if Role.HOLDING in p.roles]
    if not holdings_books:
        empty_holding = pd.DataFrame(
            columns=["holding", "weight", "value", "contribution", "pct_total", "delta_vol"]
        )
        empty_portfolio = pd.DataFrame(
            columns=["name", "weight", "value", "contribution", "pct_total", "delta_vol"]
        )
        empty_ticker = pd.DataFrame(
            columns=["ticker", "weight", "value", "contribution", "pct_total", "delta_vol", "portfolios"]
        )
        return {
            "by_holding": empty_holding,
            "by_portfolio": empty_portfolio,
            "by_ticker": empty_ticker,
        }

    if returns is None:
        if price_source is None:
            raise ValueError("Either price_source or returns must be supplied.")
        symbols = sorted({
            h.metadata.ticker for p in holdings_books for h in p.holdings
            if not h.metadata.ticker.startswith("CASH_")
        })
        s_start = start if start is not None else date(2015, 1, 1)
        s_end = end if end is not None else date.today()
        returns = price_source.get_returns(symbols, s_start, s_end)
    elif start is not None or end is not None:
        returns = returns.loc[start:end]  # type: ignore[misc]

    from hailmary.allocation.returns import synthesise_cash_returns

    cash_tickers = {
        h.metadata.ticker for p in holdings_books for h in p.holdings
        if h.metadata.ticker.startswith("CASH_")
    }
    if cash_tickers:
        returns = synthesise_cash_returns(returns, cash_tickers)

    fx = fx_rate_usd_sgd if fx_rate_usd_sgd is not None else 1.0

    def _value_sgd(portfolio: Portfolio) -> float:
        return (
            portfolio.total_value * fx
            if portfolio.currency.upper() == "USD"
            else portfolio.total_value
        )

    total_value = sum(_value_sgd(p) for p in holdings_books)
    holding_records: list[dict[str, Any]] = []
    for p in holdings_books:
        p_value = _value_sgd(p)
        for h in p.holdings:
            holding_value = h.weight * p_value
            holding_records.append(
                {
                    "portfolio": p.name,
                    "ticker": h.metadata.ticker,
                    "weight_in_book": holding_value / total_value if total_value > 0 else 0.0,
                    "value_sgd": holding_value,
                }
            )
    holdings_df = pd.DataFrame(holding_records)

    book_weights = holdings_df.groupby("ticker")["weight_in_book"].sum()
    available = [t for t in book_weights.index if t in returns.columns]
    if not available:
        empty_holding = pd.DataFrame(
            columns=["holding", "weight", "value", "contribution", "pct_total", "delta_vol"]
        )
        empty_portfolio = pd.DataFrame(
            columns=["name", "weight", "value", "contribution", "pct_total", "delta_vol"]
        )
        empty_ticker = pd.DataFrame(
            columns=["ticker", "weight", "value", "contribution", "pct_total", "delta_vol", "portfolios"]
        )
        return {
            "by_holding": empty_holding,
            "by_portfolio": empty_portfolio,
            "by_ticker": empty_ticker,
        }

    rets = returns[available].dropna(how="any")
    book_weights = book_weights.loc[available] / book_weights.loc[available].sum()
    cov = rets.cov() * _TRADING_DAYS
    w = book_weights.values
    cov_w = cov.values @ w
    portfolio_var = float(w @ cov_w)
    portfolio_vol = float(np.sqrt(portfolio_var)) if portfolio_var > 0 else 0.0

    component = np.zeros_like(w) if portfolio_vol == 0.0 else w * cov_w / portfolio_vol

    ticker_portfolios = (
        holdings_df.groupby("ticker")["portfolio"]
        .agg(lambda s: ", ".join(sorted(set(s))))
    )
    ticker_values = holdings_df.groupby("ticker")["value_sgd"].sum()
    by_ticker = pd.DataFrame(
        {
            "ticker": book_weights.index,
            "weight": book_weights.values,
            "value": book_weights.index.map(ticker_values).fillna(0.0),
            "contribution": component,
        }
    )
    by_ticker["pct_total"] = (
        by_ticker["contribution"] / portfolio_vol if portfolio_vol > 0 else 0.0
    )
    by_ticker["delta_vol"] = _delta_vol_per_ticker(
        holdings_df, cov.values, available, portfolio_vol
    )
    by_ticker["portfolios"] = by_ticker["ticker"].map(ticker_portfolios).fillna("")

    holdings_df["ticker_contribution"] = holdings_df["ticker"].map(
        dict(zip(by_ticker["ticker"], by_ticker["contribution"], strict=False))
    )
    book_weights_per_ticker = dict(
        zip(by_ticker["ticker"], by_ticker["weight"], strict=False)
    )
    holdings_df["share_of_ticker"] = holdings_df.apply(
        lambda r: (
            r["weight_in_book"] / book_weights_per_ticker[r["ticker"]]
            if book_weights_per_ticker.get(r["ticker"], 0) > 0
            else 0.0
        ),
        axis=1,
    )
    holdings_df["contribution"] = (
        holdings_df["ticker_contribution"].fillna(0.0) * holdings_df["share_of_ticker"]
    )

    by_holding = (
        holdings_df.assign(holding=lambda d: d["portfolio"] + " · " + d["ticker"])
        .loc[:, ["holding", "weight_in_book", "value_sgd", "contribution"]]
        .rename(columns={"weight_in_book": "weight", "value_sgd": "value"})
    )
    by_holding["pct_total"] = (
        by_holding["contribution"] / portfolio_vol if portfolio_vol > 0 else 0.0
    )
    by_holding["delta_vol"] = _delta_vol_per_holding(
        holdings_df, cov.values, available, portfolio_vol
    )

    by_portfolio = (
        holdings_df.groupby("portfolio", as_index=False)
        .agg(
            weight=("weight_in_book", "sum"),
            value=("value_sgd", "sum"),
            contribution=("contribution", "sum"),
        )
        .rename(columns={"portfolio": "name"})
    )
    by_portfolio["pct_total"] = (
        by_portfolio["contribution"] / portfolio_vol if portfolio_vol > 0 else 0.0
    )
    by_portfolio["delta_vol"] = _delta_vol_per_portfolio(
        holdings_df, cov.values, available, portfolio_vol
    )

    return {
        "by_holding": by_holding.sort_values("contribution", ascending=False).reset_index(
            drop=True
        ),
        "by_portfolio": by_portfolio.sort_values("contribution", ascending=False).reset_index(
            drop=True
        ),
        "by_ticker": by_ticker.sort_values("contribution", ascending=False).reset_index(
            drop=True
        ),
    }


def _book_vol_from_holdings(
    holdings_df: pd.DataFrame,
    cov_matrix: np.ndarray,
    tickers: list[str],
) -> float:
    """Compute σ_p from a holdings DataFrame and a ticker-indexed covariance matrix.

    Weights are renormalised over the *available* tickers (those present in
    ``cov_matrix``) to match the convention used by the parent ``risk_contribution``
    function. Holdings whose ticker isn't in ``cov_matrix`` (e.g. ``CASH_*``) are
    effectively ignored for this risk calculation.
    """
    grouped = holdings_df.groupby("ticker")["weight_in_book"].sum()
    raw = grouped.reindex(tickers).fillna(0.0)
    total = float(raw.sum())
    if total <= 0:
        return 0.0
    ticker_weights = (raw / total).values
    var = float(ticker_weights @ cov_matrix @ ticker_weights)
    return float(np.sqrt(max(var, 0.0)))


def _delta_vol_per_holding(
    holdings_df: pd.DataFrame,
    cov_matrix: np.ndarray,
    tickers: list[str],
    sigma_p: float,
) -> np.ndarray:
    """For each holding row: σ_p − σ_p when that single (portfolio, ticker) row is removed.

    Remaining holdings are renormalised so weights sum to 1 again. Positive → removing
    this row reduces book vol (a risk source). Negative → removing it increases book
    vol (a diversifier).
    """
    deltas = np.zeros(len(holdings_df))
    for i in range(len(holdings_df)):
        without = holdings_df.drop(holdings_df.index[i])
        if without.empty or without["weight_in_book"].sum() <= 0:
            deltas[i] = sigma_p
            continue
        sigma_ex = _book_vol_from_holdings(without, cov_matrix, tickers)
        deltas[i] = sigma_p - sigma_ex
    return deltas


def _delta_vol_per_ticker(
    holdings_df: pd.DataFrame,
    cov_matrix: np.ndarray,
    tickers: list[str],
    sigma_p: float,
) -> np.ndarray:
    """For each ticker: σ_p − σ_p when every row holding that ticker is removed.

    Aggregates across portfolios — answers "what if I had zero exposure to this
    underlying anywhere in the book?"
    """
    deltas = np.zeros(len(tickers))
    for i, t in enumerate(tickers):
        without = holdings_df[holdings_df["ticker"] != t]
        if without.empty or without["weight_in_book"].sum() <= 0:
            deltas[i] = sigma_p
            continue
        sigma_ex = _book_vol_from_holdings(without, cov_matrix, tickers)
        deltas[i] = sigma_p - sigma_ex
    return deltas


def _delta_vol_per_portfolio(
    holdings_df: pd.DataFrame,
    cov_matrix: np.ndarray,
    tickers: list[str],
    sigma_p: float,
) -> np.ndarray:
    """For each portfolio: σ_p − σ_p when that entire portfolio's holdings are removed.

    Returned in the order produced by ``groupby('portfolio').agg(...)`` — sorted
    alphabetically by portfolio name.
    """
    portfolio_names = sorted(holdings_df["portfolio"].unique())
    deltas = np.zeros(len(portfolio_names))
    for i, p_name in enumerate(portfolio_names):
        without = holdings_df[holdings_df["portfolio"] != p_name]
        if without.empty or without["weight_in_book"].sum() <= 0:
            deltas[i] = sigma_p
            continue
        sigma_ex = _book_vol_from_holdings(without, cov_matrix, tickers)
        deltas[i] = sigma_p - sigma_ex
    return deltas


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------


def benchmark_comparison(
    portfolios: Sequence[Portfolio],
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    price_source: Any | None = None,
    returns: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
    fx_series_usd_sgd: pd.Series | None = None,
    fx_rate_usd_sgd: float | None = None,
    align_window: bool = True,
    target_ann_return: float = 0.05,
) -> pd.DataFrame:
    """Sharpe / max-DD / annualised-vol per portfolio, with deltas vs each benchmark.

    Returns a DataFrame indexed by portfolio name, columns include the three
    base metrics plus ``sharpe_delta_vs_<bench>``, ``max_dd_delta_vs_<bench>``,
    and ``vol_delta_vs_<bench>`` for each ``MANAGED_BENCHMARK`` portfolio.
    Only ``HOLDING``-tagged portfolios contribute rows; rows for non-overlapping
    history are dropped.
    """
    holdings_books = [p for p in portfolios if Role.HOLDING in p.roles]
    panel = _build_returns_panel(
        holdings_books,
        price_source=price_source,
        returns=returns,
        start=start,
        end=end,
        fx_series_usd_sgd=fx_series_usd_sgd,
    )
    if panel.empty:
        return pd.DataFrame()

    benchmarks = [p.name for p in holdings_books if Role.MANAGED_BENCHMARK in p.roles]
    if not benchmarks:
        warnings.warn(
            "No MANAGED_BENCHMARK portfolios — returning base metrics only.",
            stacklevel=2,
        )

    fx = fx_rate_usd_sgd if fx_rate_usd_sgd is not None else 1.0
    if fx == 1.0 and fx_series_usd_sgd is not None and not fx_series_usd_sgd.empty:
        fx = float(fx_series_usd_sgd.dropna().iloc[-1])

    def _value_sgd(portfolio: Portfolio) -> float:
        return (
            portfolio.total_value * fx
            if portfolio.currency.upper() == "USD"
            else portfolio.total_value
        )

    value_by_name = {p.name: _value_sgd(p) for p in holdings_books}

    common_start = (
        max(panel[c].dropna().index.min() for c in panel.columns) if align_window else None
    )

    def _metric_row(series: pd.Series, name: str | None) -> dict[str, Any]:
        m = PerformanceMetrics(series, risk_free_rate=risk_free_rate)
        cum = float((1.0 + series).prod() - 1.0)
        this_year = series.index.max().year
        ytd = series[series.index.year == this_year]
        ytd_cum = float((1.0 + ytd).prod() - 1.0) if not ytd.empty else float("nan")
        return {
            "value": value_by_name.get(name, float("nan")) if name else float("nan"),
            "total_return": cum,
            "ann_return": m.annualised_return,
            "ytd_return": ytd_cum,
            "sharpe": m.sharpe,
            "max_dd": m.max_drawdown,
            "annualised_vol": m.annualised_vol,
            "n_days": len(series),
        }

    # Per-portfolio rows: aligned window when align_window=True (so deltas are
    # apples-to-apples), each portfolio's native history otherwise.
    base_rows = {}
    if align_window:
        aligned_panel = panel.loc[common_start:].dropna(how="any") if common_start else panel
        for col in aligned_panel.columns:
            series = aligned_panel[col]
            if series.empty:
                continue
            base_rows[col] = _metric_row(series, col)
    else:
        for col in panel.columns:
            series = panel[col].dropna()
            if series.empty:
                continue
            base_rows[col] = _metric_row(series, col)

    # Combined book row
    weight_map = {n: v for n, v in value_by_name.items() if n in panel.columns}
    total_weight = sum(weight_map.values())
    if total_weight > 0:
        weights = pd.Series(weight_map) / total_weight
        sub_panel = panel[weights.index]
        if align_window:
            aligned = sub_panel.loc[common_start:].dropna(how="any")
            combined_return = (
                (aligned * weights).sum(axis=1) if not aligned.empty else pd.Series(dtype=float)
            )
        else:
            available = sub_panel.notna().astype(float)
            eff = available.mul(weights, axis=1)
            row_sums = eff.sum(axis=1).replace(0.0, np.nan)
            eff = eff.div(row_sums, axis=0)
            combined_return = (sub_panel.fillna(0.0) * eff).sum(axis=1).dropna()
        if not combined_return.empty:
            combined_row = _metric_row(combined_return, None)
            combined_row["value"] = float(total_weight)
            base_rows = {"Combined book": combined_row, **base_rows}

    out = pd.DataFrame(base_rows).T

    for bench in benchmarks:
        if bench not in out.index:
            continue
        out[f"sharpe_delta_vs_{bench}"] = out["sharpe"] - out.loc[bench, "sharpe"]
        out[f"max_dd_delta_vs_{bench}"] = out["max_dd"] - out.loc[bench, "max_dd"]
        out[f"vol_delta_vs_{bench}"] = out["annualised_vol"] - out.loc[bench, "annualised_vol"]
    out.attrs["target_ann_return"] = float(target_ann_return)
    return out


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


_TEMPLATE_PATH = Path(__file__).parent / "_report_template.html.j2"


def render_html_report(
    portfolios: Sequence[Portfolio],
    output_path: Path | str,
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    price_source: Any | None = None,
    returns: pd.DataFrame | None = None,
    redundancy_threshold: float = 0.85,
    fx_rate_usd_sgd: float | None = None,
    fx_series_usd_sgd: pd.Series | None = None,
    title: str = "Allocation Diagnostic",
    strict: bool = True,
    align_window: bool = True,
    target_ann_return: float = 0.05,
    reconciliation_as_of: date | datetime | None = None,
) -> Path:
    """Run all diagnostics and write a self-contained HTML report.

    plotly figures are embedded with ``include_plotlyjs="inline"`` so the
    file is openable in any modern browser without external assets.
    """
    from importlib.util import find_spec

    if find_spec("jinja2") is None:
        raise ImportError(
            "jinja2 is required for HTML reports. "
            'Install with: pip install -e ".[allocation]"'
        )

    holdings_books = [p for p in portfolios if Role.HOLDING in p.roles]
    if strict and holdings_books:
        _build_returns_panel(
            holdings_books,
            price_source=price_source,
            returns=returns,
            start=start,
            end=end,
            fx_series_usd_sgd=fx_series_usd_sgd,
            strict=True,
        )

    book_perf = book_performance(
        portfolios,
        start=start,
        end=end,
        price_source=price_source,
        returns=returns,
        fx_rate_usd_sgd=fx_rate_usd_sgd,
        fx_series_usd_sgd=fx_series_usd_sgd,
        align_window=align_window,
    )
    reconciliation_end = (
        reconciliation_as_of
        if reconciliation_as_of is not None
        else (end if end is not None else date.today())
    )
    reconciliation = portfolio_reconciliation(
        portfolios,
        end=reconciliation_end,
        price_source=price_source,
        returns=returns,
        fx_series_usd_sgd=fx_series_usd_sgd,
    )
    reconciliation_as_of_label = (
        reconciliation_end.isoformat()
        if isinstance(reconciliation_end, date)
        else str(reconciliation_end)
    )
    exposure = combined_exposure(portfolios)
    corr = correlation_matrix(
        portfolios,
        start=start,
        end=end,
        price_source=price_source,
        returns=returns,
        fx_series_usd_sgd=fx_series_usd_sgd,
    )
    pairs = redundancy_pairs(corr, threshold=redundancy_threshold, portfolios=portfolios)
    effective_fx = fx_rate_usd_sgd
    if effective_fx is None and fx_series_usd_sgd is not None and not fx_series_usd_sgd.empty:
        effective_fx = float(fx_series_usd_sgd.dropna().iloc[-1])
    risk = risk_contribution(
        portfolios,
        start=start,
        end=end,
        price_source=price_source,
        returns=returns,
        fx_rate_usd_sgd=effective_fx,
    )
    benchmarks = benchmark_comparison(
        portfolios,
        start=start,
        end=end,
        price_source=price_source,
        returns=returns,
        fx_series_usd_sgd=fx_series_usd_sgd,
        fx_rate_usd_sgd=effective_fx,
        align_window=align_window,
        target_ann_return=target_ann_return,
    )

    figs: list[tuple[str, go.Figure]] = []
    if not book_perf["nav"].empty:
        figs.append(("nav", equity_curve_figure(book_perf["nav"])))
    if not corr.empty:
        figs.append(("correlation", correlation_figure(corr)))

    fig_html: dict[str, str] = {}
    for i, (key, fig) in enumerate(figs):
        fig_html[key] = pio.to_html(
            fig,
            include_plotlyjs="inline" if i == 0 else False,
            full_html=False,
            config={"displayModeBar": False},
        )

    exposure_cards = _exposure_cards_to_html(
        exposure, include_plotlyjs=("nav" not in fig_html)
    )

    statement_dates = sorted({p.statement_date for p in portfolios})
    statement_date_label = (
        statement_dates[-1].isoformat() if statement_dates else "unknown"
    )

    template = _load_template()
    rendered = template.render(
        title=title,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        statement_date=statement_date_label,
        portfolio_count=len(portfolios),
        nav_chart=fig_html.get("nav", ""),
        correlation_chart=fig_html.get("correlation", ""),
        exposure_cards=exposure_cards,
        book_performance=_book_performance_to_html(book_perf),
        reconciliation=_reconciliation_to_html(reconciliation),
        reconciliation_as_of=reconciliation_as_of_label,
        holdings_drilldown=_holdings_drilldown_to_html(
            portfolios,
            end=reconciliation_end,
            price_source=price_source,
            fx_series_usd_sgd=fx_series_usd_sgd,
        ),
        redundancy=_redundancy_to_html(pairs, threshold=redundancy_threshold),
        risk_by_portfolio=_risk_to_html(risk["by_portfolio"]),
        risk_by_holding=_risk_to_html(risk["by_holding"]),
        risk_by_ticker=_risk_to_html(risk["by_ticker"]),
        risk_callout=_risk_callout_to_html(risk["by_portfolio"]),
        benchmarks=_benchmarks_to_html(benchmarks),
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    return out


def _load_template() -> Any:
    import jinja2

    if _TEMPLATE_PATH.exists():
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATE_PATH.parent)),
            autoescape=jinja2.select_autoescape(["html"]),
        )
        return env.get_template(_TEMPLATE_PATH.name)
    return jinja2.Environment(autoescape=True).from_string(_FALLBACK_TEMPLATE)


def _book_performance_to_html(perf: dict[str, Any]) -> str:
    if perf["n_days"] == 0:
        return "<p>No combined-book history available.</p>"
    aum = perf["aum"]
    currency = perf.get("currency", "SGD")
    fx = perf.get("fx_rate_usd_sgd")
    facts = [
        (f"Total AUM ({currency}, HOLDING-tagged)", _fmt_compact(aum)),
        ("Days of history", f"{perf['n_days']:,}"),
    ]
    if fx is not None:
        facts.append(("USDSGD applied", f"{fx:.4f}"))
    summary_cells = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in facts)
    summary = f'<table class="ds-table ds-summary">{summary_cells}</table>'
    footnote = (
        '<p class="footnote">'
        "<strong>AUM</strong> is the sum of <code>total_value</code> across <code>HOLDING</code>-tagged "
        f"portfolios (cash trio included; see book_config). USD-reported portfolios are converted to {currency} "
        "at the rate shown above. "
        "<strong>Returns are SGD-base</strong>: USD-portfolio return series are compounded with daily USDSGD "
        "changes via <code>r_SGD = (1 + r_USD)·(1 + Δfx) − 1</code>, so Sharpe / vol / drawdown / NAV all reflect "
        "the experience of an SGD-base investor (10.9b done). "
        "<em>Cum. return</em> is compounded, <code>Π(1+rᵢ) − 1</code>; per-window <em>Ann. return</em> annualises that to "
        "<code>252/N</code> trading days for cross-window comparability. "
        "<em>Cash trio</em> (Simple USD ≈ 5% p.a. via US short-rate proxy, Simple SGD + Guitsa ≈ 1.5% p.a. per "
        "stashaway.sg/simple-var3) is visible as a replacement candidate; their inclusion pulls combined-book "
        "Sharpe and vol downward vs prior runs that hid them. "
        '<em>Caveat</em>: <strong>Risk contribution</strong> below is computed per-ticker on native-currency '
        "returns (FX adjustment not applied at the ticker level); meaningful for relative ranking, not for "
        "SGD-precise risk decomposition.</p>"
    )

    windowed = perf.get("windowed")
    if windowed is None or windowed.empty:
        return summary + footnote

    display = windowed.copy()
    display["cum_dollar"] = display["cum_return"] * aum
    display = display.rename(
        columns={
            "period": "Period",
            "cum_return": "Cum. return",
            "ann_return": "Ann. return",
            "cum_dollar": "Cum. $",
            "ann_vol": "Ann. vol",
            "sharpe": "Sharpe (ann.)",
            "max_dd": "Max DD",
            "days": "Days",
        }
    )[["Period", "Cum. return", "Cum. $", "Ann. return", "Ann. vol", "Sharpe (ann.)", "Max DD", "Days"]]

    styler = (
        display.style.format(
            {
                "Cum. return": "{:+.2%}".format,
                "Ann. return": "{:+.2%}".format,
                "Cum. $": _fmt_compact_signed,
                "Ann. vol": "{:.2%}".format,
                "Sharpe (ann.)": "{:.2f}".format,
                "Max DD": "{:.2%}".format,
                "Days": "{:,}".format,
            },
            na_rep="—",
        )
        .map(_style_pos_neg, subset=["Cum. return", "Ann. return", "Cum. $"])
        .map(_style_sharpe, subset=["Sharpe (ann.)"])
        .map(_style_dd, subset=["Max DD"])
        .hide(axis="index")
        .set_table_attributes('class="ds-table ds-windowed"')
    )
    return summary + footnote + styler.to_html()


def _reconciliation_to_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No portfolios available to reconcile.</p>"

    total_stmt_sgd = float(df["stmt_sgd"].sum())
    total_today_sgd = float(df["today_sgd"].sum())
    total_row = pd.DataFrame(
        [
            {
                "portfolio": "Total",
                "currency": "",
                "stmt_native": float("nan"),  # mixed currencies — can't sum
                "stmt_sgd": total_stmt_sgd,
                "today_native": float("nan"),
                "today_sgd": total_today_sgd,
                "delta_sgd": total_today_sgd - total_stmt_sgd,
                "return_native": float("nan"),
                "return_sgd": (
                    (total_today_sgd / total_stmt_sgd - 1.0) if total_stmt_sgd > 0 else 0.0
                ),
            }
        ]
    )
    display = pd.concat([df, total_row], ignore_index=True).rename(
        columns={
            "portfolio": "Portfolio",
            "currency": "Curr",
            "stmt_native": "Stmt (native)",
            "stmt_sgd": "Stmt (SGD)",
            "today_native": "Today (native)",
            "today_sgd": "Today (SGD)",
            "delta_sgd": "Δ SGD",
            "return_native": "Δ% native",
            "return_sgd": "Δ% SGD",
        }
    )
    formatters = {
        "Stmt (native)": _fmt_compact_precise,
        "Stmt (SGD)": _fmt_compact_precise,
        "Today (native)": _fmt_compact_precise,
        "Today (SGD)": _fmt_compact_precise,
        "Δ SGD": _fmt_compact_precise_signed,
        "Δ% native": "{:+.2%}".format,
        "Δ% SGD": "{:+.2%}".format,
    }
    total_idx = len(display) - 1
    styler = (
        display.style.format(formatters, na_rep="—")
        .map(_style_pos_neg, subset=["Δ SGD", "Δ% native", "Δ% SGD"])
        .set_table_styles(
            [{"selector": f"tbody tr:nth-child({total_idx + 1}) td",
              "props": "font-weight: 600; border-top: 2px solid #30363d;"}],
            overwrite=False,
        )
        .hide(axis="index")
        .set_table_attributes('class="ds-table"')
    )
    return styler.to_html()


_PORTFOLIO_GOAL_RE = re.compile(r"/goal/([0-9a-f]{24})", re.IGNORECASE)


def _portfolio_goal_links(
    portfolios: Sequence[Portfolio],
    links_path: Path = Path("data/holding links.xlsx"),
) -> dict[str, str]:
    """Best-effort map of portfolio name → Stashaway app goal URL.

    Reads the user's holding-links file, extracts the goal_id from any URL
    associated with the portfolio (URLs look like
    ``app.stashaway.sg/asset-details/<ticker>/goal/<goal_id>``), and returns
    a goal-page URL per portfolio. Returns empty dict if the file is missing.
    """
    if not links_path.exists():
        return {}
    try:
        import openpyxl  # type: ignore[import-untyped]
    except ImportError:
        return {}
    portfolio_aliases = {"SRS": "General SRS"}
    out: dict[str, str] = {}
    try:
        wb = openpyxl.load_workbook(links_path, data_only=False)
        if "Portfolios" not in wb.sheetnames:
            return {}
        ws = wb["Portfolios"]
        HEADER_ROW = 7
        hdr_to_col: dict[str, int] = {}
        for c in range(1, ws.max_column + 1):
            v = ws.cell(row=HEADER_ROW, column=c).value
            if v is not None:
                hdr_to_col[str(v).strip()] = c
        port_col = hdr_to_col.get("Port", 6)
        link_col = hdr_to_col.get("Link", 10)
        for r in range(HEADER_ROW + 1, ws.max_row + 1):
            port = ws.cell(row=r, column=port_col).value
            link_cell = ws.cell(row=r, column=link_col)
            url = link_cell.hyperlink.target if link_cell.hyperlink else None
            if not port or not url:
                continue
            name = portfolio_aliases.get(str(port), str(port))
            if name in out:
                continue
            m = _PORTFOLIO_GOAL_RE.search(str(url))
            if m:
                out[name] = f"https://app.stashaway.sg/goal/{m.group(1)}"
    except Exception:
        return {}
    portfolio_names = {p.name for p in portfolios}
    return {n: u for n, u in out.items() if n in portfolio_names}


def _holdings_drilldown_to_html(
    portfolios: Sequence[Portfolio],
    *,
    end: date | datetime,
    price_source: Any | None = None,
    fx_series_usd_sgd: pd.Series | None = None,
) -> str:
    """Render a collapsible per-portfolio drill-down listing each holding's
    starting value, return since statement, current value, and weight.

    Uses native HTML <details>/<summary> for free collapse behaviour. One
    section per HOLDING-tagged portfolio, sorted by AUM desc. Portfolio names
    link to the Stashaway app goal page when discoverable from
    ``data/holding links.xlsx``.
    """
    holdings_books = sorted(
        [p for p in portfolios if Role.HOLDING in p.roles],
        key=lambda p: -p.total_value,
    )
    if not holdings_books:
        return "<p>No portfolios to drill into.</p>"

    goal_urls = _portfolio_goal_links(portfolios)

    intro = (
        '<p class="footnote">Click any portfolio to expand its holdings. '
        "Each row shows the holding's value at statement date, today's spot price, "
        "return since statement (price-only, no distributions reinvested past today), "
        "and current value. Each table has a <strong>📋 Copy</strong> button that "
        "copies headers + rows + the Total row as tab-separated text — paste straight "
        "into Excel/Sheets. Portfolio names are linked to the Stashaway app where "
        "discoverable from your <code>holding links.xlsx</code>.</p>"
    )
    sections: list[str] = []
    for p in holdings_books:
        df = holdings_reconciliation(
            p, end=end, price_source=price_source, fx_series_usd_sgd=fx_series_usd_sgd
        )
        if df.empty:
            continue
        # Build a Total row so it copies cleanly to Excel.
        total_stmt = float(df["stmt_value"].sum())
        total_today = float(df["today_value"].sum())
        delta = total_today - total_stmt
        ret = (total_today / total_stmt - 1) if total_stmt > 0 else 0.0
        total_row = pd.DataFrame(
            [{
                "ticker": "Total",
                "label": "",
                "weight": 1.0,
                "stmt_value": total_stmt,
                "stmt_price": float("nan"),
                "stmt_date_used": pd.NaT,
                "today_price": float("nan"),
                "today_date_used": pd.NaT,
                "return": ret,
                "today_value": total_today,
                "delta": delta,
            }]
        )
        # Preserve any sgd-mirror columns from df with NaN so concat works
        for col in df.columns:
            if col not in total_row.columns:
                if col.endswith("_sgd"):
                    total_row[col] = df[col].sum()
                else:
                    total_row[col] = float("nan")
        df_with_total = pd.concat([df, total_row], ignore_index=True)

        ccy = p.currency.upper()
        delta_cls = "pos" if delta >= 0 else "neg"
        portfolio_name_html = (
            f'<a href="{goal_urls[p.name]}" target="_blank" rel="noopener" class="port-link">{p.name}</a>'
            if p.name in goal_urls
            else f"<strong>{p.name}</strong>"
        )
        # Summary delta: native always; SGD also shown for USD portfolios so
        # the user has both currencies at a glance without expanding the table.
        delta_label = f"{_fmt_compact_precise_signed(delta)} {ccy}"
        if ccy == "USD" and "delta_sgd" in df.columns:
            delta_sgd_sum = float(df["delta_sgd"].sum())
            delta_label += f" / {_fmt_compact_precise_signed(delta_sgd_sum)} SGD"
        summary_line = (
            f"{portfolio_name_html} · "
            f"{p.currency} {_fmt_compact_precise(total_stmt)} → "
            f"<span class=\"{delta_cls}\">{_fmt_compact_precise(total_today)}</span> "
            f"(<span class=\"{delta_cls}\">{delta_label}, {ret:+.2%}</span>) · "
            f"{len(df)} holdings"
        )

        col_stmt_val = f"Stmt value ({ccy})"
        col_today_val = f"Today value ({ccy})"
        col_delta = f"Δ value ({ccy})"
        col_stmt_px = f"Stmt price ({ccy})"
        col_today_px = f"Today price ({ccy})"
        col_stmt_sgd = "Stmt value (SGD)"
        col_today_sgd = "Today value (SGD)"
        col_delta_sgd = "Δ value (SGD)"
        display = df_with_total.rename(
            columns={
                "ticker": "Ticker",
                "label": "Asset · Region",
                "weight": "Weight",
                "stmt_value": col_stmt_val,
                "stmt_price": col_stmt_px,
                "stmt_date_used": "Stmt date",
                "today_price": col_today_px,
                "today_date_used": "Today date",
                "return": "Δ%",
                "today_value": col_today_val,
                "delta": col_delta,
                "stmt_value_sgd": col_stmt_sgd,
                "today_value_sgd": col_today_sgd,
                "delta_sgd": col_delta_sgd,
            }
        )
        # Column order: native values clustered, then SGD mirror at the right
        # (only present for USD portfolios when FX is available). Date columns
        # sit next to their price columns so staleness is immediately visible.
        ordered = [
            "Ticker", "Asset · Region", "Weight",
            col_stmt_val,
            col_stmt_px, "Stmt date",
            col_today_px, "Today date",
            "Δ%", col_today_val, col_delta,
        ]
        if ccy == "USD" and col_stmt_sgd in display.columns:
            ordered += [col_stmt_sgd, col_today_sgd, col_delta_sgd]
        # Dedupe while preserving order — SGD-native portfolios have
        # col_delta == col_delta_sgd (both "Δ value (SGD)"), which would
        # otherwise create a duplicate column reference in the styler subset.
        seen: set[str] = set()
        dedup_ordered = []
        for c in ordered:
            if c in display.columns and c not in seen:
                dedup_ordered.append(c)
                seen.add(c)
        display = display[dedup_ordered]

        # Flag rows where forward-accrual was applied so the date cell
        # reads e.g. "2026-05-29* (+1d)". Pre-convert per-row so the
        # Styler formatter (which is row-blind) gets the right marker.
        def _date_with_marker(v: Any, accrued: int) -> str:
            if v is None or (isinstance(v, float) and v != v) or pd.isna(v):
                return "—"
            if isinstance(v, date):
                base = v.isoformat()
                return f"{base}* (+{int(accrued)}d)" if accrued else base
            return str(v)

        if "accrued_days" in df_with_total.columns:
            df_with_total["today_date_used"] = [
                _date_with_marker(d, a or 0)
                for d, a in zip(
                    df_with_total["today_date_used"],
                    df_with_total["accrued_days"],
                )
            ]

        def _fmt_date(v: Any) -> str:
            if v is None or (isinstance(v, float) and v != v) or pd.isna(v):
                return "—"
            if isinstance(v, date):
                return v.isoformat()
            return str(v)

        formatters: dict[str, Any] = {
            "Weight": "{:.2%}".format,
            col_stmt_val: _fmt_compact_precise,
            col_stmt_px: "{:,.4f}".format,
            "Stmt date": _fmt_date,
            col_today_px: "{:,.4f}".format,
            "Today date": _fmt_date,
            "Δ%": "{:+.2%}".format,
            col_today_val: _fmt_compact_precise,
            col_delta: _fmt_compact_precise_signed,
            col_stmt_sgd: _fmt_compact_precise,
            col_today_sgd: _fmt_compact_precise,
            col_delta_sgd: _fmt_compact_precise_signed,
        }
        # Subset for pos/neg tint — dedupe in case col_delta == col_delta_sgd
        # (true for SGD-native portfolios where both labels resolve to
        # "Δ value (SGD)" — passing the same column twice trips pandas Styler).
        delta_cols: list[str] = []
        for c in (col_delta, col_delta_sgd):
            if c in display.columns and c not in delta_cols:
                delta_cols.append(c)
        total_idx = len(display) - 1
        styler = (
            display.style.format(formatters, na_rep="—")
            .map(_style_pos_neg, subset=["Δ%"] + delta_cols)
            .set_table_styles(
                [{"selector": f"tbody tr:nth-child({total_idx + 1}) td",
                  "props": "font-weight: 600; border-top: 2px solid #30363d;"}],
                overwrite=False,
            )
            .hide(axis="index")
            .set_table_attributes('class="ds-table ds-holdings"')
        )
        sections.append(
            f'<details class="drill"><summary>{summary_line}</summary>'
            f'{styler.to_html()}</details>'
        )
    if not sections:
        return "<p>No holding-level data available.</p>"
    return intro + "\n".join(sections)


def _exposure_to_html(exposure: dict[str, pd.DataFrame]) -> dict[str, str]:
    return {
        dim.replace("_", " ").title(): df.to_html(
            index=False,
            formatters={
                "value": _fmt_compact,
                "weight": "{:.2%}".format,
            },
            classes="ds-table",
            border=0,
        )
        for dim, df in exposure.items()
        if not df.empty
    }


def _exposure_cards_to_html(
    exposure: dict[str, pd.DataFrame],
    *,
    include_plotlyjs: bool = False,
) -> str:
    """Build a 3-column grid of donut+table cards, one per exposure dimension."""
    dims = [("asset_class", "Asset Class"), ("region", "Region"), ("sector", "Sector")]
    nonempty = [(key, label) for key, label in dims if not exposure.get(key, pd.DataFrame()).empty]
    if not nonempty:
        return ""

    cards: list[str] = []
    for i, (key, label) in enumerate(nonempty):
        df = exposure[key]
        donut = single_exposure_donut(df, title=label)
        donut_html = pio.to_html(
            donut,
            include_plotlyjs="inline" if (i == 0 and include_plotlyjs) else False,
            full_html=False,
            config={"displayModeBar": False},
        )
        table_html = df.to_html(
            index=False,
            formatters={
                "value": _fmt_compact,
                "weight": "{:.2%}".format,
            },
            classes="ds-table ds-exposure",
            border=0,
        )
        cards.append(
            f'<div class="exposure-card">{donut_html}{table_html}</div>'
        )
    return f'<div class="exposure-grid">{"".join(cards)}</div>'


def _redundancy_to_html(
    pairs: Iterable[tuple[str, str, float, str]],
    *,
    threshold: float,
) -> str:
    intro = (
        '<p class="footnote">Pairs of portfolios whose daily return series move in lockstep — '
        f"correlation ≥ {threshold:.2f} over the common-history window. These are duplicate "
        "exposures: holding both adds management overhead without diversification benefit. "
        '<em>"Review"</em> flags the portfolio worth consolidating <em>into</em> the other; '
        "<code>PROTECTED</code> portfolios (e.g. General SRS) never appear in this column because their "
        "tax-locking makes them un-removable.</p>"
    )
    rows = list(pairs)
    if not rows:
        return intro + f"<p>No portfolio pairs above threshold {threshold:.2f}.</p>"
    df = pd.DataFrame(rows, columns=["a", "b", "rho", "candidate"]).rename(
        columns={
            "a": "Portfolio A",
            "b": "Portfolio B",
            "rho": "Correlation",
            "candidate": "Review",
        }
    )
    styler = (
        df.style.format({"Correlation": "{:.3f}".format})
        .map(_style_rho, subset=["Correlation"])
        .hide(axis="index")
        .set_table_attributes('class="ds-table"')
    )
    return intro + styler.to_html()


def _risk_to_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No holdings to attribute.</p>"
    display = df.rename(
        columns={
            "portfolio": "Portfolio",
            "ticker": "Ticker",
            "name": "Portfolio",
            "holding": "Portfolio · Ticker",
            "weight": "Weight",
            "value": "Value (SGD)",
            "contribution": "Vol contribution (ann.)",
            "pct_total": "% of total risk",
            "delta_vol": "ΔVol if removed",
            "portfolios": "Portfolios holding it",
        }
    )
    formatters = {
        "Weight": "{:.2%}".format,
        "Value (SGD)": _fmt_compact,
        "Vol contribution (ann.)": "{:.2%}".format,
        "% of total risk": "{:.2%}".format,
    }
    if "ΔVol if removed" in display.columns:
        formatters["ΔVol if removed"] = "{:+.2%}".format

    styler = display.style.format(formatters, na_rep="—")

    def _bar_max(col: str) -> float:
        return float(pd.to_numeric(display[col], errors="coerce").max() or 0.0)

    if "Weight" in display.columns:
        m = _bar_max("Weight")
        if m > 0:
            styler = styler.bar(
                subset=["Weight"], color="#58a6ff", vmin=0.0, vmax=m, height=70, width=92
            )
    if "Value (SGD)" in display.columns:
        m = _bar_max("Value (SGD)")
        if m > 0:
            styler = styler.bar(
                subset=["Value (SGD)"], color="#a371f7", vmin=0.0, vmax=m, height=70, width=92
            )
    if "Vol contribution (ann.)" in display.columns:
        vc = pd.to_numeric(display["Vol contribution (ann.)"], errors="coerce")
        bound = float(max(abs(vc.min() or 0.0), abs(vc.max() or 0.0)))
        if bound > 0:
            styler = styler.bar(
                subset=["Vol contribution (ann.)"],
                color=["#3fb950", "#f85149"],
                align="zero",
                vmin=-bound,
                vmax=bound,
                height=70,
                width=92,
            )
    if "% of total risk" in display.columns:
        styler = styler.bar(
            subset=["% of total risk"], color="#ffa657", vmin=0.0, vmax=1.0, height=70, width=92
        )
    if "ΔVol if removed" in display.columns:
        dv = pd.to_numeric(display["ΔVol if removed"], errors="coerce")
        bound = float(max(abs(dv.min() or 0.0), abs(dv.max() or 0.0)))
        if bound > 0:
            styler = styler.bar(
                subset=["ΔVol if removed"],
                color=["#3fb950", "#f85149"],
                align="zero",
                vmin=-bound,
                vmax=bound,
                height=70,
                width=92,
            )

    styler = styler.hide(axis="index").set_table_attributes('class="ds-table"')
    return styler.to_html()


def _risk_callout_to_html(by_portfolio: pd.DataFrame) -> str:
    """Render a callout with combined book σ_p and a concrete worked example
    using the largest single risk contributor in the book."""
    if by_portfolio.empty:
        return ""
    sigma_p = float(by_portfolio["contribution"].sum())
    if sigma_p <= 0:
        return ""
    top = by_portfolio.sort_values("contribution", ascending=False).iloc[0]
    top_name = str(top["name"])
    top_contrib = float(top["contribution"])
    top_pct = float(top["pct_total"])
    top_delta = float(top.get("delta_vol", 0.0))
    sigma_after = sigma_p - top_delta
    return f"""
<div class="ds-callout">
  <p style="margin: 0 0 6px 0;"><strong>Combined book σ<sub>p</sub> (annualised) =
     <span style="color:#58a6ff;font-size:15px;">{sigma_p:.2%}</span></strong>
     &nbsp;— every row's <em>Vol contribution</em> below sums to exactly this number.</p>
  <p style="margin: 0; color: #8b949e; font-size: 12px;">
    <strong>Worked example:</strong> {top_name} has the largest <em>Vol contribution</em> of
    <span style="color:#f85149">{top_contrib:.2%}</span>, which is
    <span style="color:#f85149">{top_pct:.0%}</span> of the book's
    {sigma_p:.2%} vol. Selling all of {top_name} and reinvesting proportionally would change
    σ<sub>p</sub> by <span style="color:#f85149">{-top_delta:+.2%}</span>
    (new σ<sub>p</sub> ≈ {sigma_after:.2%}). The two numbers differ because the contribution is the
    math decomposition <em>at current weights</em>, while ΔVol is the discrete effect after the
    remaining holdings get rebalanced upward to fill the gap.
  </p>
</div>
"""


def _style_delta_vol(val: float) -> str:
    """Red tint when removing the row reduces book vol (risk source);
    green tint when removing it raises book vol (diversifier)."""
    if pd.isna(val) or val == 0:
        return ""
    if val > 0:
        a = min(0.45, 0.15 + val * 4.0)
        return f"background-color: rgba(248, 81, 73, {a:.2f}); color: #f85149;"
    a = min(0.45, 0.15 + abs(val) * 4.0)
    return f"background-color: rgba(63, 185, 80, {a:.2f}); color: #3fb950;"


def _benchmarks_to_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No benchmark comparison available (no overlapping return history).</p>"
    cols = list(df.columns)
    sharpe_delta_cols = [c for c in cols if c.startswith("sharpe_delta_vs_")]
    dd_delta_cols = [c for c in cols if c.startswith("max_dd_delta_vs_")]
    vol_delta_cols = [c for c in cols if c.startswith("vol_delta_vs_")]

    rename_map: dict[str, str] = {
        "value": "Value (SGD)",
        "total_return": "Total return",
        "ann_return": "Ann. return",
        "ytd_return": "YTD",
        "sharpe": "Sharpe",
        "max_dd": "Max DD",
        "annualised_vol": "Ann. vol",
        "n_days": "Days",
    }
    for c in sharpe_delta_cols:
        rename_map[c] = f"ΔSharpe vs {c.removeprefix('sharpe_delta_vs_')}"
    for c in dd_delta_cols:
        rename_map[c] = f"ΔMax DD vs {c.removeprefix('max_dd_delta_vs_')}"
    for c in vol_delta_cols:
        rename_map[c] = f"ΔVol vs {c.removeprefix('vol_delta_vs_')}"

    display = df.rename(columns=rename_map)

    formatters: dict[str, Any] = {
        "Value (SGD)": _fmt_compact,
        "Total return": "{:+.2%}".format,
        "Ann. return": "{:+.2%}".format,
        "YTD": "{:+.2%}".format,
        "Sharpe": "{:.2f}".format,
        "Max DD": "{:.2%}".format,
        "Ann. vol": "{:.2%}".format,
        "Days": "{:,.0f}".format,
    }
    sharpe_delta_display = [rename_map[c] for c in sharpe_delta_cols]
    dd_delta_display = [rename_map[c] for c in dd_delta_cols]
    vol_delta_display = [rename_map[c] for c in vol_delta_cols]
    for c in sharpe_delta_display:
        formatters[c] = "{:+.2f}".format
    for c in dd_delta_display + vol_delta_display:
        formatters[c] = "{:+.2%}".format

    target_ann_return = float(df.attrs.get("target_ann_return", 0.05))

    def _style_ann_return(val: float) -> str:
        if pd.isna(val):
            return ""
        if val >= target_ann_return:
            return "background-color: rgba(63, 185, 80, 0.20); color: #3fb950;"
        if val >= 0:
            return "background-color: rgba(210, 153, 34, 0.18); color: #d29922;"
        return "background-color: rgba(248, 81, 73, 0.20); color: #f85149;"

    styler = display.style.format(formatters, na_rep="—")
    if "Total return" in display.columns:
        styler = styler.map(_style_pos_neg, subset=["Total return"])
    if "Ann. return" in display.columns:
        styler = styler.map(_style_ann_return, subset=["Ann. return"])
    if "YTD" in display.columns:
        styler = styler.map(_style_pos_neg, subset=["YTD"])
    if "Sharpe" in display.columns:
        styler = styler.map(_style_sharpe, subset=["Sharpe"])
    if "Max DD" in display.columns:
        styler = styler.map(_style_dd, subset=["Max DD"])
    if sharpe_delta_display:
        styler = styler.map(_style_pos_neg, subset=sharpe_delta_display)
    if dd_delta_display:
        styler = styler.map(_style_dd_delta, subset=dd_delta_display)
    # ΔVol vs benchmark intentionally left un-tinted — more vol than benchmark isn't
    # unambiguously good or bad (depends on whether you want more or less risk).
    styler = styler.set_table_attributes('class="ds-table"')
    return styler.to_html()


_FALLBACK_TEMPLATE = """\
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
  body { font-family: Inter, system-ui, sans-serif; background: #0d1117; color: #e6edf3;
         margin: 0; padding: 24px; }
  h1, h2 { color: #e6edf3; }
  h2 { border-bottom: 1px solid #30363d; padding-bottom: 6px; margin-top: 32px; }
  .ds-table { border-collapse: collapse; margin-top: 8px; }
  .ds-table th, .ds-table td { border: 1px solid #30363d; padding: 6px 10px; }
  .ds-table th { background: #161b22; }
  .meta { color: #8b949e; font-size: 13px; }
  .columns { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
             gap: 16px; }
</style>
</head><body>
<h1>{{ title }}</h1>
<p class="meta">Generated {{ generated_at }} · {{ portfolio_count }} portfolios ·
   statement-date-snapshot reconstruction (forward-looking estimate of the book
   as of the statement date, not realised history).</p>

<h2>Combined book — performance summary</h2>
{{ book_performance | safe }}
{% if nav_chart %}{{ nav_chart | safe }}{% endif %}

<h2>Combined-book exposure</h2>
{{ exposure_cards | safe }}

<h2>Portfolio correlation</h2>
{% if correlation_chart %}{{ correlation_chart | safe }}{% endif %}

<h2>Redundancy</h2>
{{ redundancy | safe }}

<h2>Risk contribution — by portfolio</h2>
{{ risk_by_portfolio | safe }}

<h2>Risk contribution — by holding</h2>
{{ risk_by_holding | safe }}

<h2>Benchmark comparison</h2>
{{ benchmarks | safe }}

</body></html>
"""
