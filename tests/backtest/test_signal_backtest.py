"""Tests for BarBacktest and BarBacktestResult."""

from __future__ import annotations

import pandas as pd
import pytest

from hailmary.backtest.signal_backtest import BarBacktest, BarBacktestResult
from hailmary.models.signals import TrendSignal

EXECUTION_COLS = {
    "return_mark_to_close", "return_conservative", "return_net",
    "position_start", "position_end",
    "strategy_equity_mark_to_close", "strategy_equity_conservative", "strategy_equity_net",
    "trade_cycle_id",
}


@pytest.fixture
def signal_df(bars_df: pd.DataFrame) -> pd.DataFrame:
    return TrendSignal(ma_window=200).run(bars_df)


@pytest.fixture
def bt_result(signal_df: pd.DataFrame) -> BarBacktestResult:
    return BarBacktest().run(signal_df)


def test_bar_backtest_returns_result(bt_result: BarBacktestResult) -> None:
    assert isinstance(bt_result, BarBacktestResult)


def test_bar_backtest_adds_execution_columns(bt_result: BarBacktestResult) -> None:
    assert EXECUTION_COLS.issubset(set(bt_result.data.columns))


def test_bar_backtest_preserves_signal_columns(
    signal_df: pd.DataFrame, bt_result: BarBacktestResult
) -> None:
    assert set(signal_df.columns).issubset(set(bt_result.data.columns))


def test_equity_starts_at_one(bt_result: BarBacktestResult) -> None:
    eq = bt_result.data["strategy_equity_mark_to_close"].unstack("symbol")
    assert (eq.iloc[0] == 1.0).all()


def test_conservative_lte_mtc(bt_result: BarBacktestResult) -> None:
    eq_mtc = bt_result.data["strategy_equity_mark_to_close"].unstack("symbol")
    eq_con = bt_result.data["strategy_equity_conservative"].unstack("symbol")
    assert (eq_con.iloc[-1] <= eq_mtc.iloc[-1]).all()


def test_flat_bars_have_zero_return(bt_result: BarBacktestResult) -> None:
    flat = bt_result.data[bt_result.data["cycle"] == "None"]
    assert (flat["return_mark_to_close"] == 0.0).all()
    assert (flat["return_conservative"] == 0.0).all()
    assert (flat["return_net"] == 0.0).all()


def test_return_net_lte_mtc_with_cost(signal_df: pd.DataFrame) -> None:
    result = BarBacktest(cost_bps=10).run(signal_df)
    eq_mtc = result.data["strategy_equity_mark_to_close"].unstack("symbol")
    eq_net = result.data["strategy_equity_net"].unstack("symbol")
    assert (eq_net.iloc[-1] <= eq_mtc.iloc[-1]).all()


def test_return_net_equals_mtc_with_zero_cost(bt_result: BarBacktestResult) -> None:
    diff = (bt_result.data["return_net"] - bt_result.data["return_mark_to_close"]).abs()
    assert (diff < 1e-12).all()
