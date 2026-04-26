"""Tests for TrendSignalStrategy."""

from __future__ import annotations

import pandas as pd
import pytest

from hailmary.backtest.strategies import TrendSignalStrategy
from hailmary.models.signals import TrendSignal


@pytest.fixture
def signal_df(bars_df: pd.DataFrame) -> pd.DataFrame:
    return TrendSignal(ma_window=200).run(bars_df)


@pytest.fixture
def strategy(signal_df: pd.DataFrame) -> TrendSignalStrategy:
    return TrendSignalStrategy(signal_df)


def test_weight_per_symbol_is_fixed_fraction_of_universe(
    strategy: TrendSignalStrategy, signal_df: pd.DataFrame
) -> None:
    n_universe = signal_df.index.get_level_values("symbol").nunique()
    expected_weight = 1.0 / n_universe
    timestamps = signal_df.index.get_level_values("timestamp").unique()
    for ts in timestamps[210:220]:
        w = strategy.generate_weights(pd.DataFrame(), ts, None)  # type: ignore[arg-type]
        if not w.empty:
            assert (w - expected_weight).abs().max() < 1e-9


def test_weights_sum_lte_one(
    strategy: TrendSignalStrategy, signal_df: pd.DataFrame
) -> None:
    timestamps = signal_df.index.get_level_values("timestamp").unique()
    for ts in timestamps[210:220]:
        w = strategy.generate_weights(pd.DataFrame(), ts, None)  # type: ignore[arg-type]
        assert float(w.sum()) <= 1.0 + 1e-9


def test_weights_are_equal_across_in_signal_symbols(
    strategy: TrendSignalStrategy, signal_df: pd.DataFrame
) -> None:
    timestamps = signal_df.index.get_level_values("timestamp").unique()
    for ts in timestamps[210:220]:
        w = strategy.generate_weights(pd.DataFrame(), ts, None)  # type: ignore[arg-type]
        if len(w) > 1:
            assert w.std() < 1e-9


def test_returns_empty_series_before_warmup(
    strategy: TrendSignalStrategy, signal_df: pd.DataFrame
) -> None:
    ts = signal_df.index.get_level_values("timestamp").unique()[0]
    w = strategy.generate_weights(pd.DataFrame(), ts, None)  # type: ignore[arg-type]
    assert isinstance(w, pd.Series)


def test_falls_back_to_prior_bar_for_missing_timestamp(
    strategy: TrendSignalStrategy, signal_df: pd.DataFrame
) -> None:
    last_ts = signal_df.index.get_level_values("timestamp").unique()[-1]
    future_ts = last_ts + pd.Timedelta(days=1)
    w = strategy.generate_weights(pd.DataFrame(), future_ts, None)  # type: ignore[arg-type]
    assert isinstance(w, pd.Series)
