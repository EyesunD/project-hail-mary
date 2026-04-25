"""Alpaca Markets provider (requires `pip install hailmary[alpaca]`)."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from loguru import logger

from hailmary.data.base import (
    DataProvider,
    FundamentalData,
    ProviderUnavailableError,
    Timeframe,
)
from hailmary.data.cache import DataCache

_TF_MAP: dict[Timeframe, str] = {
    Timeframe.MINUTE_1: "1Min",
    Timeframe.MINUTE_5: "5Min",
    Timeframe.MINUTE_15: "15Min",
    Timeframe.MINUTE_30: "30Min",
    Timeframe.HOUR_1: "1Hour",
    Timeframe.HOUR_4: "4Hour",
    Timeframe.DAY_1: "1Day",
    Timeframe.WEEK_1: "1Week",
    Timeframe.MONTH_1: "1Month",
}


class AlpacaProvider(DataProvider):
    """Alpaca Markets data provider — supports both historical and live feeds."""

    name = "alpaca"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        feed: str = "iex",
        cache: DataCache | None = None,
    ) -> None:
        try:
            from alpaca.data.historical import StockHistoricalDataClient  # type: ignore[import]
            from alpaca.data.requests import StockBarsRequest  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install alpaca: pip install hailmary[alpaca]") from exc

        self._client = StockHistoricalDataClient(api_key, secret_key)
        self._StockBarsRequest = StockBarsRequest
        self._feed = feed
        self._cache = cache or DataCache()

    def get_bars(
        self,
        symbols: list[str],
        start: date | datetime,
        end: date | datetime,
        timeframe: Timeframe = Timeframe.DAY_1,
        *,
        adjust: bool = True,
    ) -> pd.DataFrame:
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore[import]

        cache_key = DataCache.make_key(
            provider=self.name, symbols=sorted(symbols), start=str(start), end=str(end),
            timeframe=timeframe.value,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Build Alpaca TimeFrame from our enum
        tf_str = _TF_MAP[timeframe]
        amount, unit_str = int(tf_str[:-len(tf_str.lstrip("0123456789"))]), tf_str.lstrip("0123456789")
        unit_map = {"Min": TimeFrameUnit.Minute, "Hour": TimeFrameUnit.Hour,
                    "Day": TimeFrameUnit.Day, "Week": TimeFrameUnit.Week, "Month": TimeFrameUnit.Month}
        alpaca_tf = TimeFrame(amount, unit_map[unit_str])

        req = self._StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=alpaca_tf,
            start=start,
            end=end,
            feed=self._feed,
            adjustment="all" if adjust else "raw",
        )
        try:
            bars = self._client.get_stock_bars(req).df
        except Exception as exc:
            raise ProviderUnavailableError(f"Alpaca request failed: {exc}") from exc

        bars = bars.rename(columns={"t": "timestamp", "o": "open", "h": "high",
                                    "l": "low", "c": "close", "v": "volume", "vw": "vwap"})
        bars.index.names = ["symbol", "timestamp"]
        logger.info("Alpaca returned {} bars for {} symbols", len(bars), len(symbols))
        self._cache.set(cache_key, bars)
        return bars

    def get_latest_bars(
        self,
        symbols: list[str],
        n: int = 1,
        timeframe: Timeframe = Timeframe.DAY_1,
    ) -> pd.DataFrame:
        end = datetime.utcnow()
        # fetch enough history to guarantee n bars
        from datetime import timedelta
        lookback = {Timeframe.MINUTE_1: timedelta(hours=2), Timeframe.DAY_1: timedelta(days=n + 5)}
        start = end - lookback.get(timeframe, timedelta(days=n + 5))
        bars = self.get_bars(symbols, start, end, timeframe)
        return bars.groupby(level=0).tail(n)

    def get_fundamentals(self, symbols: list[str]) -> dict[str, FundamentalData]:
        raise NotImplementedError("Alpaca does not provide fundamental data.")
