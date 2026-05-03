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


# ----------------------------------------------------------------- entry_offset


def _trade_durations(result: BarBacktestResult) -> pd.Series:
    in_trade = result.data[result.data["cycle"] != "None"]
    return in_trade.groupby(["symbol", "trade_cycle_id"]).size()


def test_entry_offset_drops_short_trades(signal_df: pd.DataFrame) -> None:
    base = BarBacktest().run(signal_df)
    delayed = BarBacktest().run(signal_df, entry_offset=5)
    base_durations = _trade_durations(base)
    delayed_durations = _trade_durations(delayed)
    # Every surviving delayed-trade run is at least 1 bar (we enter at original
    # bar k and original exit ≥ k + 1, so post-shift run ≥ 2 bars; minimum
    # is 1 bar after we strip the now-flat bars 0..k-1 within the cycle id).
    assert (delayed_durations >= 1).all()
    # Total trade count under delay-5 cannot exceed native.
    assert delayed_durations.shape[0] <= base_durations.shape[0]


def test_entry_offset_zero_is_noop(signal_df: pd.DataFrame) -> None:
    base = BarBacktest().run(signal_df)
    same = BarBacktest().run(signal_df, entry_offset=0)
    pd.testing.assert_frame_equal(base.data, same.data)


def test_entry_offset_shifts_entry_timestamp(signal_df: pd.DataFrame) -> None:
    base = BarBacktest().run(signal_df)
    delayed = BarBacktest().run(signal_df, entry_offset=5)

    # For every (symbol, surviving original trade) whose duration > 6, the
    # delayed entry timestamp is exactly 5 bars later than the native one.
    base_in = base.data[base.data["enter"] == 1].reset_index()
    del_in  = delayed.data[delayed.data["enter"] == 1].reset_index()

    for sym, base_grp in base_in.groupby("symbol"):
        del_grp = del_in[del_in["symbol"] == sym].reset_index(drop=True)
        base_grp = base_grp.reset_index(drop=True)
        # Walk surviving delayed entries; each should match a native entry
        # 5 bars earlier (within the symbol's bar grid).
        sym_bars = base.data.xs(sym, level="symbol").index
        for _, drow in del_grp.iterrows():
            del_pos = sym_bars.get_loc(drow["timestamp"])
            assert del_pos >= 5
            base_ts = sym_bars[del_pos - 5]
            assert (base_grp["timestamp"] == base_ts).any()


def test_entry_offset_preserves_no_trade_overlap(signal_df: pd.DataFrame) -> None:
    delayed = BarBacktest().run(signal_df, entry_offset=5).data
    # Every entry bar must have signal_open == 0 (we just entered).
    entries = delayed[delayed["enter"] == 1]
    assert (entries["signal_open"] == 0).all()
    # Every "None" cycle must have enter == 0 and exit == 0.
    flats = delayed[delayed["cycle"] == "None"]
    assert (flats["enter"] == 0).all()
    assert (flats["exit"] == 0).all()


def test_entry_offset_keeps_long_trades(signal_df: pd.DataFrame) -> None:
    # Regression: bool dtype was widened to object after groupby.shift().fillna(),
    # which made `~prev_in` evaluate truthily everywhere and wiped out every
    # surviving run.  At least one trade longer than k+1 bars must remain.
    base = BarBacktest().run(signal_df)
    delayed = BarBacktest().run(signal_df, entry_offset=5)
    base_max_dur = _trade_durations(base).max()
    if base_max_dur > 6:
        assert _trade_durations(delayed).shape[0] >= 1
