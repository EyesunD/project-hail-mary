"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def price_df() -> pd.DataFrame:
    """Synthetic wide close-price DataFrame (400 bars, 10 symbols)."""
    rng = np.random.default_rng(42)
    n_days, n_symbols = 400, 10
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    log_returns = rng.normal(0.0003, 0.015, size=(n_days, n_symbols))
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=symbols)


@pytest.fixture
def bars_df() -> pd.DataFrame:
    """Synthetic (symbol, timestamp) MultiIndex OHLCV DataFrame (400 bars, 3 symbols)."""
    rng = np.random.default_rng(42)
    n_days = 400
    symbols = ["SYM00", "SYM01", "SYM02"]
    idx = pd.bdate_range("2020-01-01", periods=n_days, name="timestamp")
    frames = []
    for sym in symbols:
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_days)))
        df = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.005, n_days)),
                "high": close * (1 + rng.uniform(0, 0.02, n_days)),
                "low": close * (1 - rng.uniform(0, 0.02, n_days)),
                "close": close,
                "volume": rng.integers(1_000_000, 10_000_000, n_days).astype(float),
            },
            index=idx,
        )
        df.index.name = "timestamp"
        df["symbol"] = sym
        frames.append(df.reset_index().set_index(["symbol", "timestamp"]))
    return pd.concat(frames).sort_index()


@pytest.fixture
def return_series() -> pd.Series:
    rng = np.random.default_rng(0)
    n = 252 * 3
    rets = rng.normal(0.0005, 0.012, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rets, index=idx, name="returns")
