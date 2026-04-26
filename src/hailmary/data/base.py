"""Abstract base classes for the market data platform."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Enums & value types
# ---------------------------------------------------------------------------


class Timeframe(str, Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1mo"


@dataclass(frozen=True, slots=True)
class Bar:
    """A single OHLCV bar."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trades: int | None = None


@dataclass
class FundamentalData:
    """Snapshot of fundamental / balance-sheet data for a single symbol."""

    symbol: str
    as_of: date
    market_cap: float | None = None
    enterprise_value: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    ev_ebitda: float | None = None
    roe: float | None = None
    roa: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    revenue_growth_yoy: float | None = None
    earnings_growth_yoy: float | None = None
    extra: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MarketDataError(Exception):
    """Base exception for data layer errors."""


class ProviderUnavailableError(MarketDataError):
    """Raised when a provider cannot be reached or is not configured."""


class SymbolNotFoundError(MarketDataError):
    """Raised when the provider does not recognise the requested symbol."""


class InsufficientDataError(MarketDataError):
    """Raised when the provider returns fewer bars than requested."""


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------


class DataProvider(abc.ABC):
    """Abstract market-data provider.

    Subclass this to integrate any data source.  Every method returns
    *normalised* structures so callers are fully insulated from vendor quirks.
    """

    # Human-readable name shown in logs / registries
    name: str = "unnamed"

    # ------------------------------------------------------------------ OHLCV

    @abc.abstractmethod
    def get_bars(
        self,
        symbols: list[str],
        start: date | datetime,
        end: date | datetime,
        timeframe: Timeframe = Timeframe.DAY_1,
        *,
        adjust: bool = True,
    ) -> pd.DataFrame:
        """Return OHLCV bars as a MultiIndex DataFrame (symbol, timestamp).

        Index levels: (symbol: str, timestamp: pd.Timestamp)
        Columns: open, high, low, close, volume, [vwap, trades]
        """

    @abc.abstractmethod
    def get_latest_bars(
        self,
        symbols: list[str],
        n: int = 1,
        timeframe: Timeframe = Timeframe.DAY_1,
    ) -> pd.DataFrame:
        """Return the *n* most recent bars for each symbol."""

    # -------------------------------------------------------------- Fundamentals

    def get_fundamentals(self, symbols: list[str]) -> dict[str, FundamentalData]:
        """Return the latest fundamental snapshot for each symbol.

        Providers that do not support fundamentals should raise
        ``NotImplementedError``.
        """
        raise NotImplementedError(f"{self.name} does not support fundamentals.")

    # -------------------------------------------------------------- Universe

    def search_symbols(self, query: str) -> list[str]:
        """Search for tickers matching *query*. Returns a list of symbol strings."""
        raise NotImplementedError(f"{self.name} does not support symbol search.")

    def get_universe(self, index: str) -> list[str]:
        """Return constituent tickers for a named index (e.g. 'SP500')."""
        raise NotImplementedError(f"{self.name} does not support universe lookup.")

    # -------------------------------------------------------------- Helpers

    def get_returns(
        self,
        symbols: list[str],
        start: date | datetime,
        end: date | datetime,
        timeframe: Timeframe = Timeframe.DAY_1,
        *,
            adjust: bool = False,
            log: bool = False,
    ) -> pd.DataFrame:
        """Convenience: OHLCV → close-to-close returns wide DataFrame."""
        bars = self.get_bars(symbols, start, end, timeframe,adjust=adjust)
        close = bars["close"].unstack(level=0)  # shape: (time, symbols)
        if log:
            import numpy as np

            return np.log(close / close.shift(1)).dropna(how="all")
        return close.pct_change().dropna(how="all")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
