"""Current-snapshot return reconstruction.

Each portfolio's daily return time series is computed as
``r_p,t = Σᵢ wᵢ · rᵢ,t`` with the **current** weights ``wᵢ`` held constant
across the entire historical window. See design ``D2`` — this is a *projection*
of the current book onto historical prices, not a realised track record.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import pandas as pd

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

    weights = pd.Series(
        {h.metadata.ticker: h.weight for h in portfolio.holdings},
        dtype=float,
    )
    weights = weights.groupby(level=0).sum()  # collapse duplicate tickers
    weights = weights / weights.sum()  # renormalise after dedup

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
        warnings.warn(msg, stacklevel=2)
        sub = available

    weighted = sub.values @ weights.values
    return pd.Series(weighted, index=sub.index, name=portfolio.name, dtype=np.float64)


def _add_cash_columns(returns: pd.DataFrame, portfolio: Portfolio) -> pd.DataFrame:
    """Synthesise zero-return series for any ``CASH_*`` holdings in *portfolio*.

    Cash positions don't have Yahoo tickers; we treat them as 0% daily return.
    The series is aligned to the existing returns DataFrame's index.
    """
    cash_tickers = sorted({
        h.metadata.ticker for h in portfolio.holdings
        if h.metadata.ticker.startswith("CASH_")
    })
    if not cash_tickers:
        return returns
    if returns.empty:
        # No price data fetched yet — synthesise a minimal index so weights can resolve.
        # In practice this only happens for cash-only portfolios that get filtered
        # upstream, but be defensive anyway.
        idx = pd.bdate_range(date(2015, 1, 1), date.today(), name="date")
        returns = pd.DataFrame(index=idx)
    out = returns.copy()
    for t in cash_tickers:
        out[t] = 0.0
    return out


def _as_date(ts: pd.Timestamp | Any) -> date:
    if isinstance(ts, pd.Timestamp):
        return ts.date()
    return cast(date, ts)
