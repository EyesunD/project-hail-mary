"""Yahoo Finance provider (via yfinance)."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import yfinance as yf
from loguru import logger

from hailmary.data.base import (
    DataProvider,
    FundamentalData,
    ProviderUnavailableError,
    SymbolNotFoundError,
    Timeframe,
)
from hailmary.data.cache import DataCache

_TF_MAP: dict[Timeframe, str] = {
    Timeframe.MINUTE_1: "1m",
    Timeframe.MINUTE_5: "5m",
    Timeframe.MINUTE_15: "15m",
    Timeframe.MINUTE_30: "30m",
    Timeframe.HOUR_1: "1h",
    Timeframe.HOUR_4: "1h",   # yfinance has no 4h; caller should resample
    Timeframe.DAY_1: "1d",
    Timeframe.WEEK_1: "1wk",
    Timeframe.MONTH_1: "1mo",
}

_SP500_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class YahooFinanceProvider(DataProvider):
    """Free market data from Yahoo Finance — no API key required."""

    name = "yahoo"

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache or DataCache()

    # ------------------------------------------------------------------ OHLCV

    def get_bars(
        self,
        symbols: list[str],
        start: date | datetime,
        end: date | datetime,
        timeframe: Timeframe = Timeframe.DAY_1,
        *,
        adjust: bool = True,
    ) -> pd.DataFrame:
        yf_interval = _TF_MAP.get(timeframe, "1d")
        cache_key = DataCache.make_key(
            provider=self.name, symbols=sorted(symbols), start=str(start), end=str(end),
            interval=yf_interval, adjust=adjust,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        logger.info("Fetching {} symbols from Yahoo ({} → {})", len(symbols), start, end)
        try:
            raw = yf.download(
                symbols,
                start=start,
                end=end,
                interval=yf_interval,
                auto_adjust=adjust,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            raise ProviderUnavailableError(f"yfinance download failed: {exc}") from exc

        df = self._normalise(raw, symbols)
        if df.empty:
            raise SymbolNotFoundError(f"No data returned for {symbols}")

        self._cache.set(cache_key, df)
        return df

    def get_latest_bars(
        self,
        symbols: list[str],
        n: int = 1,
        timeframe: Timeframe = Timeframe.DAY_1,
    ) -> pd.DataFrame:
        tickers = yf.Tickers(" ".join(symbols))
        frames = []
        for sym in symbols:
            hist = tickers.tickers[sym].history(period="5d", interval=_TF_MAP[timeframe])
            hist = hist.tail(n)
            hist.index = pd.MultiIndex.from_product([[sym], hist.index], names=["symbol", "timestamp"])
            frames.append(hist)
        df = pd.concat(frames)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]

    # ------------------------------------------------------------ Fundamentals

    def get_fundamentals(self, symbols: list[str]) -> dict[str, FundamentalData]:
        result: dict[str, FundamentalData] = {}
        for sym in symbols:
            info = yf.Ticker(sym).info
            result[sym] = FundamentalData(
                symbol=sym,
                as_of=date.today(),
                market_cap=info.get("marketCap"),
                enterprise_value=info.get("enterpriseValue"),
                pe_ratio=info.get("trailingPE"),
                pb_ratio=info.get("priceToBook"),
                ps_ratio=info.get("priceToSalesTrailing12Months"),
                ev_ebitda=info.get("enterpriseToEbitda"),
                roe=info.get("returnOnEquity"),
                roa=info.get("returnOnAssets"),
                gross_margin=info.get("grossMargins"),
                operating_margin=info.get("operatingMargins"),
                net_margin=info.get("profitMargins"),
                debt_to_equity=info.get("debtToEquity"),
                current_ratio=info.get("currentRatio"),
                revenue_growth_yoy=info.get("revenueGrowth"),
                earnings_growth_yoy=info.get("earningsGrowth"),
            )
        return result

    # -------------------------------------------------------------- Universe

    def get_universe(self, index: str) -> list[str]:
        if index.upper() in ("SP500", "S&P500", "SPX"):
            tables = pd.read_html(_SP500_WIKI)
            return tables[0]["Symbol"].tolist()
        raise NotImplementedError(f"Universe '{index}' not supported by {self.name}.")

    # ----------------------------------------------------------------- private

    @staticmethod
    def _normalise(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
        """Convert yfinance's wide MultiIndex output to (symbol, timestamp) MultiIndex."""
        frames = []
        for sym in symbols:
            if len(symbols) == 1:
                sym_df = raw.copy()
            else:
                try:
                    sym_df = raw[sym].copy()
                except KeyError:
                    logger.warning("Symbol {} not in response — skipping.", sym)
                    continue
            sym_df.columns = [c.lower() for c in sym_df.columns]
            sym_df = sym_df[["open", "high", "low", "close", "volume"]].dropna(how="all")
            sym_df.index = pd.to_datetime(sym_df.index)
            sym_df.index.name = "timestamp"
            sym_df["symbol"] = sym
            sym_df = sym_df.reset_index().set_index(["symbol", "timestamp"])
            frames.append(sym_df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()
