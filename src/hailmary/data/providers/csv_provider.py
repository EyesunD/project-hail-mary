"""CSV / Parquet local file provider — great for research and offline work."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from hailmary.data.base import DataProvider, SymbolNotFoundError, Timeframe


class CSVProvider(DataProvider):
    """Load OHLCV data from local CSV or Parquet files.

    Expected filename format: ``{data_dir}/{symbol}.csv`` or ``{symbol}.parquet``
    Expected columns: date/timestamp, open, high, low, close, volume
    """

    name = "csv"

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    def get_bars(
        self,
        symbols: list[str],
        start: date | datetime,
        end: date | datetime,
        timeframe: Timeframe = Timeframe.DAY_1,
        *,
        adjust: bool = True,
    ) -> pd.DataFrame:
        frames = []
        for sym in symbols:
            df = self._load_symbol(sym)
            df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
            df["symbol"] = sym
            df = df.reset_index().set_index(["symbol", "timestamp"])
            frames.append(df)

        if not frames:
            raise SymbolNotFoundError(f"No data found for {symbols} in {self.data_dir}")
        return pd.concat(frames).sort_index()

    def get_latest_bars(
        self,
        symbols: list[str],
        n: int = 1,
        timeframe: Timeframe = Timeframe.DAY_1,
    ) -> pd.DataFrame:
        frames = []
        for sym in symbols:
            df = self._load_symbol(sym).tail(n)
            df["symbol"] = sym
            df = df.reset_index().set_index(["symbol", "timestamp"])
            frames.append(df)
        return pd.concat(frames).sort_index()

    # ----------------------------------------------------------------- private

    def _load_symbol(self, symbol: str) -> pd.DataFrame:
        for ext in (".parquet", ".csv"):
            path = self.data_dir / f"{symbol}{ext}"
            if path.exists():
                logger.debug("Loading {} from {}", symbol, path)
                df = pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
                return self._normalise(df)
        raise SymbolNotFoundError(f"No file for '{symbol}' in {self.data_dir}")

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.lower().strip() for c in df.columns]
        ts_col = next((c for c in df.columns if c in ("date", "timestamp", "datetime", "time")), None)
        if ts_col is None:
            raise ValueError("Cannot find a date/timestamp column.")
        df = df.rename(columns={ts_col: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]]
