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

import warnings
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from hailmary.allocation.portfolios import Portfolio, Role
from hailmary.allocation.returns import portfolio_returns
from hailmary.analytics.metrics import PerformanceMetrics
from hailmary.viz.theme import PALETTE, apply_theme

if TYPE_CHECKING:
    pass


_TRADING_DAYS = 252


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


def combined_exposure_figure(exposure: dict[str, pd.DataFrame]) -> go.Figure:
    """Stacked-bar figure summarising the three exposure dimensions."""
    fig = go.Figure()
    colours = [PALETTE["accent_blue"], PALETTE["accent_green"], PALETTE["accent_purple"]]
    for (dim, df), colour in zip(exposure.items(), colours, strict=False):
        fig.add_trace(
            go.Bar(
                name=dim.replace("_", " ").title(),
                x=df["bucket"],
                y=df["weight"],
                marker_color=colour,
                hovertemplate="%{x}: %{y:.1%}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", yaxis_tickformat=".0%")
    return apply_theme(fig, title="Combined-book exposure", height=420)


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
) -> pd.DataFrame:
    """Compute one return series per portfolio and stack into a wide DataFrame.

    Falls back to the in-memory wide returns DataFrame if supplied; otherwise
    delegates to ``portfolio_returns(price_source=...)`` per portfolio.
    """
    series = []
    for p in portfolios:
        try:
            s = portfolio_returns(
                p,
                start=start,
                end=end,
                price_source=price_source,
                returns=returns,
            )
        except Exception as exc:
            warnings.warn(f"Skipping {p.name!r} in returns panel: {exc}", stacklevel=2)
            continue
        series.append(s)
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
) -> pd.DataFrame:
    """Pairwise return correlation across all portfolios over the common window."""
    panel = _build_returns_panel(
        portfolios,
        price_source=price_source,
        returns=returns,
        start=start,
        end=end,
    )
    if panel.empty:
        return pd.DataFrame()
    # Use pandas' default pairwise correlation (NaN-aware per pair) rather than
    # collapsing the panel to its common-history window — otherwise a single
    # young series (e.g. a recently-launched ETF) would cap every other pair's
    # window to its own short history.
    return panel.corr(min_periods=20)


def correlation_figure(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        text_auto=".2f",
    )
    return apply_theme(fig, title="Portfolio correlation", height=520)


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
        empty = pd.DataFrame(columns=["name", "weight", "contribution", "pct_total"])
        return {"by_holding": empty.copy(), "by_portfolio": empty.copy()}

    if returns is None:
        if price_source is None:
            raise ValueError("Either price_source or returns must be supplied.")
        symbols = sorted({h.metadata.ticker for p in holdings_books for h in p.holdings})
        s_start = start if start is not None else date(2015, 1, 1)
        s_end = end if end is not None else date.today()
        returns = price_source.get_returns(symbols, s_start, s_end)
    elif start is not None or end is not None:
        returns = returns.loc[start:end]  # type: ignore[misc]

    holding_records: list[dict[str, Any]] = []
    total_value = sum(p.total_value for p in holdings_books)
    for p in holdings_books:
        for h in p.holdings:
            holding_records.append(
                {
                    "portfolio": p.name,
                    "ticker": h.metadata.ticker,
                    "weight_in_book": (h.weight * p.total_value) / total_value,
                }
            )
    holdings_df = pd.DataFrame(holding_records)

    book_weights = holdings_df.groupby("ticker")["weight_in_book"].sum()
    available = [t for t in book_weights.index if t in returns.columns]
    if not available:
        empty = pd.DataFrame(columns=["name", "weight", "contribution", "pct_total"])
        return {"by_holding": empty.copy(), "by_portfolio": empty.copy()}

    rets = returns[available].dropna(how="any")
    book_weights = book_weights.loc[available] / book_weights.loc[available].sum()
    cov = rets.cov() * _TRADING_DAYS
    w = book_weights.values
    cov_w = cov.values @ w
    portfolio_var = float(w @ cov_w)
    portfolio_vol = float(np.sqrt(portfolio_var)) if portfolio_var > 0 else 0.0

    component = np.zeros_like(w) if portfolio_vol == 0.0 else w * cov_w / portfolio_vol

    by_ticker = pd.DataFrame(
        {
            "name": book_weights.index,
            "weight": book_weights.values,
            "contribution": component,
        }
    )
    by_ticker["pct_total"] = by_ticker["contribution"] / portfolio_vol if portfolio_vol > 0 else 0.0

    holdings_df["ticker_contribution"] = holdings_df["ticker"].map(
        dict(zip(by_ticker["name"], by_ticker["contribution"], strict=False))
    )
    book_weights_per_ticker = dict(zip(by_ticker["name"], by_ticker["weight"], strict=False))
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
        holdings_df.assign(name=lambda d: d["portfolio"] + " · " + d["ticker"])
        .loc[:, ["name", "weight_in_book", "contribution"]]
        .rename(columns={"weight_in_book": "weight"})
    )
    by_holding["pct_total"] = (
        by_holding["contribution"] / portfolio_vol if portfolio_vol > 0 else 0.0
    )

    by_portfolio = (
        holdings_df.groupby("portfolio", as_index=False)
        .agg(weight=("weight_in_book", "sum"), contribution=("contribution", "sum"))
        .rename(columns={"portfolio": "name"})
    )
    by_portfolio["pct_total"] = (
        by_portfolio["contribution"] / portfolio_vol if portfolio_vol > 0 else 0.0
    )

    return {
        "by_holding": by_holding.sort_values("contribution", ascending=False).reset_index(
            drop=True
        ),
        "by_portfolio": by_portfolio.sort_values("contribution", ascending=False).reset_index(
            drop=True
        ),
    }


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
) -> pd.DataFrame:
    """Sharpe / max-DD / annualised-vol per portfolio, with deltas vs each benchmark.

    Returns a DataFrame indexed by portfolio name, columns include the three
    base metrics plus ``sharpe_delta_vs_<bench>``, ``max_dd_delta_vs_<bench>``,
    and ``vol_delta_vs_<bench>`` for each ``MANAGED_BENCHMARK`` portfolio.
    Custom + benchmark portfolios both get rows; rows for non-overlapping
    history are dropped.
    """
    panel = _build_returns_panel(
        portfolios,
        price_source=price_source,
        returns=returns,
        start=start,
        end=end,
    )
    if panel.empty:
        return pd.DataFrame()

    benchmarks = [p.name for p in portfolios if Role.MANAGED_BENCHMARK in p.roles]
    if not benchmarks:
        warnings.warn(
            "No MANAGED_BENCHMARK portfolios — returning base metrics only.",
            stacklevel=2,
        )

    # Each portfolio's metrics are computed on its native return-series history
    # (dropping leading NaNs only). Otherwise a recently-launched holding in
    # one portfolio would truncate every other portfolio's metrics window.
    base_rows = {}
    for col in panel.columns:
        series = panel[col].dropna()
        if series.empty:
            continue
        m = PerformanceMetrics(series, risk_free_rate=risk_free_rate)
        base_rows[col] = {
            "sharpe": m.sharpe,
            "max_dd": m.max_drawdown,
            "annualised_vol": m.annualised_vol,
            "n_days": len(series),
        }
    out = pd.DataFrame(base_rows).T

    for bench in benchmarks:
        if bench not in out.index:
            continue
        out[f"sharpe_delta_vs_{bench}"] = out["sharpe"] - out.loc[bench, "sharpe"]
        out[f"max_dd_delta_vs_{bench}"] = out["max_dd"] - out.loc[bench, "max_dd"]
        out[f"vol_delta_vs_{bench}"] = out["annualised_vol"] - out.loc[bench, "annualised_vol"]
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
    title: str = "Allocation Diagnostic",
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

    exposure = combined_exposure(portfolios)
    corr = correlation_matrix(
        portfolios,
        start=start,
        end=end,
        price_source=price_source,
        returns=returns,
    )
    pairs = redundancy_pairs(corr, threshold=redundancy_threshold, portfolios=portfolios)
    risk = risk_contribution(
        portfolios,
        start=start,
        end=end,
        price_source=price_source,
        returns=returns,
    )
    benchmarks = benchmark_comparison(
        portfolios,
        start=start,
        end=end,
        price_source=price_source,
        returns=returns,
    )

    figs: list[tuple[str, go.Figure]] = []
    if any(not df.empty for df in exposure.values()):
        figs.append(("Combined-book exposure", combined_exposure_figure(exposure)))
    if not corr.empty:
        figs.append(("Portfolio correlation", correlation_figure(corr)))

    fig_html: list[tuple[str, str]] = [
        (
            label,
            pio.to_html(
                fig,
                include_plotlyjs="inline" if i == 0 else False,
                full_html=False,
                config={"displayModeBar": False},
            ),
        )
        for i, (label, fig) in enumerate(figs)
    ]

    template = _load_template()
    rendered = template.render(
        title=title,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        portfolio_count=len(portfolios),
        figures=fig_html,
        exposure=_exposure_to_html(exposure),
        redundancy=_redundancy_to_html(pairs, threshold=redundancy_threshold),
        risk_by_portfolio=_risk_to_html(risk["by_portfolio"]),
        risk_by_holding=_risk_to_html(risk["by_holding"]),
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


def _exposure_to_html(exposure: dict[str, pd.DataFrame]) -> dict[str, str]:
    return {
        dim.replace("_", " ").title(): df.to_html(
            index=False, float_format="{:.2%}".format, classes="ds-table"
        )
        for dim, df in exposure.items()
        if not df.empty
    }


def _redundancy_to_html(
    pairs: Iterable[tuple[str, str, float, str]],
    *,
    threshold: float,
) -> str:
    rows = list(pairs)
    if not rows:
        return f"<p>No portfolio pairs above threshold {threshold:.2f}.</p>"
    df = pd.DataFrame(rows, columns=["a", "b", "rho", "candidate"])
    return df.to_html(index=False, float_format="{:.3f}".format, classes="ds-table")


def _risk_to_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No holdings to attribute.</p>"
    return df.to_html(index=False, float_format="{:.4f}".format, classes="ds-table")


def _benchmarks_to_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No benchmark comparison available (no overlapping return history).</p>"
    return df.to_html(float_format="{:.3f}".format, classes="ds-table")


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
   current-snapshot reconstruction (forward-looking estimate of today's book,
   not realised history).</p>

{% for label, fig in figures %}
<h2>{{ label }}</h2>
{{ fig | safe }}
{% endfor %}

<h2>Combined-book exposure</h2>
<div class="columns">
  {% for label, table in exposure.items() %}
    <div><h3>{{ label }}</h3>{{ table | safe }}</div>
  {% endfor %}
</div>

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
