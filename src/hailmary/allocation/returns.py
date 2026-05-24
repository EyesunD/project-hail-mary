"""Current-snapshot return reconstruction.

Each portfolio's daily return time series is computed as
``r_p,t = Σᵢ wᵢ · rᵢ,t`` with the **current** weights ``wᵢ`` held constant
across the entire historical window. See design ``D2`` — this is a *projection*
of the current book onto historical prices, not a realised track record.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from hailmary.allocation.portfolios import Portfolio


class _PriceSource(Protocol):
    def get_returns(
        self,
        symbols: list[str],
        start: date | datetime,
        end: date | datetime,
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame: ...


class InsufficientHistoryError(Exception):
    """Raised in strict mode when a holding's price history doesn't cover the window."""

    def __init__(self, portfolio: str, missing: list[str], window: tuple[date, date]) -> None:
        self.portfolio = portfolio
        self.missing = missing
        self.window = window
        super().__init__(
            f"Portfolio {portfolio!r}: tickers {missing} have insufficient history "
            f"for window {window[0]}..{window[1]}"
        )


def portfolio_returns(
    portfolio: Portfolio,
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    price_source: _PriceSource | None = None,
    returns: pd.DataFrame | None = None,
    strict: bool = False,
    fx_series_usd_sgd: pd.Series | None = None,
) -> pd.Series:
    """Reconstruct a portfolio's daily return series.

    Parameters
    ----------
    portfolio:
        :class:`Portfolio` carrying current holdings and weights.
    start, end:
        Optional analysis window. Defaults to ``[2015-01-01, today]`` — pulled
        wide so we have something to truncate down. Common-history truncation
        is applied per-holding regardless.
    price_source:
        Anything with a ``get_returns(symbols, start, end)`` method (e.g. a
        :class:`hailmary.data.providers.YahooFinanceProvider` instance or the
        :class:`ProviderRegistry`). Required if *returns* is not supplied.
    returns:
        Pre-fetched wide returns DataFrame ``(date × ticker)``. Useful for
        tests and notebooks where data has already been pulled.
    strict:
        If True, raise :class:`InsufficientHistoryError` when any holding's
        history doesn't cover the requested window. Default is to truncate
        to the common history and emit a warning.
    fx_series_usd_sgd:
        Optional daily close series of ``USDSGD=X``. When provided and the
        portfolio's reporting currency is USD, the return series is converted
        to SGD via ``r_SGD = (1 + r_USD) · (1 + Δfx) − 1`` per day so the
        result reflects an SGD-base investor's experience. Without it,
        USD-reported portfolios stay in USD terms.
    """
    if returns is None:
        if price_source is None:
            raise ValueError("Either price_source or returns must be supplied.")
        s_start: date | datetime = start if start is not None else date(2015, 1, 1)
        s_end: date | datetime = end if end is not None else date.today()
        symbols = sorted({
            h.metadata.ticker for h in portfolio.holdings
            if not h.metadata.ticker.startswith("CASH_")
        })
        returns = price_source.get_returns(symbols, s_start, s_end) if symbols else pd.DataFrame()
    elif start is not None or end is not None:
        returns = returns.loc[start:end]  # type: ignore[misc]

    returns = _add_cash_columns(returns, portfolio)

    weights_by_ticker: dict[str, float] = {}
    for h in portfolio.holdings:
        weights_by_ticker[h.metadata.ticker] = (
            weights_by_ticker.get(h.metadata.ticker, 0.0) + h.weight
        )
    weights = pd.Series(weights_by_ticker, dtype=float)
    if weights.sum() == 0:
        raise InsufficientHistoryError(
            portfolio.name,
            list(weights.index),
            window=(date(1970, 1, 1), date(1970, 1, 1)),
        )
    weights = weights / weights.sum()

    missing = [t for t in weights.index if t not in returns.columns]
    if missing:
        raise InsufficientHistoryError(
            portfolio.name,
            missing,
            window=(_as_date(returns.index.min()), _as_date(returns.index.max())),
        )

    sub = returns[weights.index]
    full_idx = sub.index
    available = sub.dropna(how="any")
    truncated_tickers = [t for t in sub.columns if sub[t].isna().any()]
    if truncated_tickers:
        msg = (
            f"Portfolio {portfolio.name!r}: tickers {truncated_tickers} have gaps; "
            f"common-history window {available.index.min().date()}.."
            f"{available.index.max().date()} (was {full_idx.min().date()}.."
            f"{full_idx.max().date()})"
        )
        if strict:
            raise InsufficientHistoryError(
                portfolio.name,
                truncated_tickers,
                window=(_as_date(full_idx.min()), _as_date(full_idx.max())),
            )
        logger.trace(msg)
        sub = available

    weighted = sub.values @ weights.values
    series = pd.Series(weighted, index=sub.index, name=portfolio.name, dtype=np.float64)
    fee_annual = float(portfolio.metadata.get("management_fee_annual", 0.0))
    if fee_annual > 0:
        series = series - (fee_annual / _TRADING_DAYS_PER_YEAR)
    if fx_series_usd_sgd is not None and portfolio.currency.upper() == "USD":
        series = _apply_fx_adjustment(series, fx_series_usd_sgd)
    return series


def _apply_fx_adjustment(returns: pd.Series, fx_series: pd.Series) -> pd.Series:
    """Compound a USD-denominated return series with daily USDSGD FX changes."""
    fx_aligned = fx_series.reindex(returns.index).ffill().bfill()
    fx_chg = fx_aligned.pct_change().fillna(0.0)
    adjusted = (1.0 + returns) * (1.0 + fx_chg) - 1.0
    return pd.Series(adjusted.values, index=returns.index, name=returns.name, dtype=np.float64)


def last_business_day_on_or_before(d: date | datetime | None = None) -> date:
    """Return *d* if it is a business day; otherwise the most recent prior business day.

    Use as the default end date for diagnostic windows so we don't request data for
    weekends/today-before-close and dump truncation noise into the logs. Skips
    weekends only — public-holiday awareness is left to the data provider.
    """
    from datetime import timedelta

    target = d if d is not None else date.today()
    if isinstance(target, datetime):
        target = target.date()
    while target.weekday() >= 5:
        target -= timedelta(days=1)
    return target


_CASH_ANNUAL_YIELDS: dict[str, float] = {
    # Stashaway Simple SGD / Guitsa: 1.5% p.a. net of fees per stashaway.sg/simple-var3
    "CASH_SGD": 0.015,
    # Stashaway Simple USD: approximated as short-rate USD (BIL ≈ SOFR ≈ 5% in 2026).
    "CASH_USD": 0.05,
}
_CASH_ANNUAL_VOLS: dict[str, float] = {
    # 30% LionGlobal SGD MMF + 70% LionGlobal SGD Enhanced Liquidity has some duration risk
    "CASH_SGD": 0.0035,
    # US 1-3M T-Bills track the front of the SOFR curve — very stable
    "CASH_USD": 0.0015,
}
_TRADING_DAYS_PER_YEAR = 252


def synthesise_cash_returns(
    returns: pd.DataFrame, cash_tickers: Iterable[str]
) -> pd.DataFrame:
    """Add synthetic daily return columns for the given ``CASH_*`` tickers.

    Drawn from ``Normal(daily_yield, daily_vol)`` where the annual yield matches the
    product's published net rate (Stashaway Simple SGD/Guitsa ≈ 1.5%, Simple USD
    proxied at the US short rate ≈ 5%) and the annual vol matches the realistic
    NAV-wobble of money-market / enhanced-liquidity sleeves (≈ 35 bps for SGD,
    ≈ 15 bps for USD). Non-zero vol so correlation against other return series is
    defined — it will still be near zero versus risk assets, which is correct for
    cash. Seeded per ticker for reproducibility.

    Tickers not starting with ``CASH_`` are silently ignored. Tickers already
    present as columns in *returns* are left as-is.
    """
    needed = sorted(
        t for t in set(cash_tickers)
        if t.startswith("CASH_") and t not in returns.columns
    )
    if not needed:
        return returns
    if returns.empty:
        idx = pd.bdate_range(date(2015, 1, 1), date.today(), name="date")
        returns = pd.DataFrame(index=idx)
    out = returns.copy()
    n = len(out.index)
    for t in needed:
        annual_yield = _CASH_ANNUAL_YIELDS.get(t, 0.0)
        annual_vol = _CASH_ANNUAL_VOLS.get(t, 0.0)
        daily_mean = (1.0 + annual_yield) ** (1.0 / _TRADING_DAYS_PER_YEAR) - 1.0
        daily_vol = annual_vol / np.sqrt(_TRADING_DAYS_PER_YEAR)
        if daily_vol == 0.0:
            out[t] = daily_mean
        else:
            rng = np.random.default_rng(seed=abs(hash(t)) % (2**32))
            out[t] = rng.normal(daily_mean, daily_vol, size=n)
    return out


def _add_cash_columns(returns: pd.DataFrame, portfolio: Portfolio) -> pd.DataFrame:
    """Per-portfolio wrapper around :func:`synthesise_cash_returns`."""
    cash_tickers = {
        h.metadata.ticker for h in portfolio.holdings
        if h.metadata.ticker.startswith("CASH_")
    }
    return synthesise_cash_returns(returns, cash_tickers)


def _as_date(ts: pd.Timestamp | Any) -> date:
    if isinstance(ts, pd.Timestamp):
        return ts.date()
    return cast(date, ts)
