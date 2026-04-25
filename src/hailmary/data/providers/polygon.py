"""Polygon.io provider (requires `pip install hailmary[polygon]`)."""

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

_TF_MAP: dict[Timeframe, tuple[int, str]] = {
    Timeframe.MINUTE_1: (1, "minute"),
    Timeframe.MINUTE_5: (5, "minute"),
    Timeframe.MINUTE_15: (15, "minute"),
    Timeframe.MINUTE_30: (30, "minute"),
    Timeframe.HOUR_1: (1, "hour"),
    Timeframe.HOUR_4: (4, "hour"),
    Timeframe.DAY_1: (1, "day"),
    Timeframe.WEEK_1: (1, "week"),
    Timeframe.MONTH_1: (1, "month"),
}


class PolygonProvider(DataProvider):
    """Polygon.io market data provider."""

    name = "polygon"

    def __init__(self, api_key: str, cache: DataCache | None = None) -> None:
        try:
            from polygon import RESTClient  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install polygon: pip install hailmary[polygon]") from exc
        self._client = RESTClient(api_key)
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
        multiplier, span = _TF_MAP[timeframe]
        cache_key = DataCache.make_key(
            provider=self.name, symbols=sorted(symbols), start=str(start),
            end=str(end), multiplier=multiplier, span=span,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        frames = []
        for sym in symbols:
            try:
                aggs = self._client.get_aggs(
                    sym, multiplier=multiplier, timespan=span,
                    from_=start, to=end, adjusted=adjust, limit=50_000,
                )
            except Exception as exc:
                raise ProviderUnavailableError(f"Polygon failed for {sym}: {exc}") from exc

            rows = [
                {"timestamp": pd.Timestamp(a.timestamp, unit="ms", tz="UTC"),
                 "open": a.open, "high": a.high, "low": a.low,
                 "close": a.close, "volume": a.volume, "vwap": a.vwap}
                for a in aggs
            ]
            if not rows:
                logger.warning("Polygon returned no bars for {}", sym)
                continue
            df = pd.DataFrame(rows).set_index("timestamp")
            df["symbol"] = sym
            df = df.reset_index().set_index(["symbol", "timestamp"])
            frames.append(df)

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames).sort_index()
        self._cache.set(cache_key, result)
        return result

    def get_latest_bars(
        self,
        symbols: list[str],
        n: int = 1,
        timeframe: Timeframe = Timeframe.DAY_1,
    ) -> pd.DataFrame:
        from datetime import timedelta
        end = datetime.utcnow()
        start = end - timedelta(days=max(n * 2, 10))
        bars = self.get_bars(symbols, start, end, timeframe)
        return bars.groupby(level=0).tail(n)

    def get_fundamentals(self, symbols: list[str]) -> dict[str, FundamentalData]:
        result: dict[str, FundamentalData] = {}
        for sym in symbols:
            details = self._client.get_ticker_details(sym)
            result[sym] = FundamentalData(
                symbol=sym,
                as_of=date.today(),
                market_cap=getattr(details, "market_cap", None),
            )
        return result
