"""Tests for SignalTradePerformance and SignalAllocationPerformance."""

from __future__ import annotations

import pandas as pd
import pytest

from hailmary.analytics.metrics import PerformanceMetrics
from hailmary.analytics.signal_analytics import (
    SignalAllocationPerformance,
    SignalTradePerformance,
)
from hailmary.backtest.signal_backtest import BarBacktest, BarBacktestResult
from hailmary.models.signals import TrendSignal


@pytest.fixture
def bt_result(bars_df: pd.DataFrame) -> BarBacktestResult:
    signal_df = TrendSignal(ma_window=200).run(bars_df)
    return BarBacktest().run(signal_df)


@pytest.fixture
def alloc(bt_result: BarBacktestResult) -> SignalAllocationPerformance:
    return SignalAllocationPerformance(bt_result)


@pytest.fixture
def trades(bt_result: BarBacktestResult) -> SignalTradePerformance:
    return SignalTradePerformance(bt_result)


# ----------------------------------------------------------------- per-symbol


def test_summary_index_is_symbols(
    alloc: SignalAllocationPerformance, bars_df: pd.DataFrame
) -> None:
    symbols = bars_df.index.get_level_values("symbol").unique()
    assert set(alloc.summary().index) == set(symbols)


def test_summary_columns(alloc: SignalAllocationPerformance) -> None:
    cols = set(alloc.summary().columns)
    assert {
        "return_mtc", "return_conservative", "return_net",
        "max_drawdown_mtc", "avg_drawdown_mtc",
        "max_drawdown_conservative", "avg_drawdown_conservative",
        "entries", "exits", "invested_days", "pct_invested",
    }.issubset(cols)


def test_summary_drawdown_bounded(alloc: SignalAllocationPerformance) -> None:
    s = alloc.summary()
    assert (s["max_drawdown_mtc"] <= 0).all()
    assert (s["max_drawdown_conservative"] <= 0).all()
    assert (s["avg_drawdown_mtc"] <= 0).all()
    assert (s["avg_drawdown_conservative"] <= 0).all()


def test_summary_conservative_dd_lte_mtc(alloc: SignalAllocationPerformance) -> None:
    s = alloc.summary()
    assert (s["max_drawdown_conservative"] <= s["max_drawdown_mtc"] + 1e-9).all()


def test_equity_mtc_shape(
    alloc: SignalAllocationPerformance, bars_df: pd.DataFrame
) -> None:
    eq = alloc.equity(method="mtc")
    n_symbols = bars_df.index.get_level_values("symbol").nunique()
    assert eq.shape[1] == n_symbols
    assert (eq.iloc[0] == 1.0).all()


def test_equity_conservative_lte_mtc(alloc: SignalAllocationPerformance) -> None:
    eq_mtc = alloc.equity(method="mtc")
    eq_con = alloc.equity(method="conservative")
    assert (eq_con.iloc[-1] <= eq_mtc.iloc[-1]).all()


def test_pct_invested_bounded(alloc: SignalAllocationPerformance) -> None:
    pct = alloc.summary()["pct_invested"]
    assert (pct >= 0).all() and (pct <= 100).all()


# ----------------------------------------------------------------- portfolio


def test_portfolio_equity_starts_at_one(alloc: SignalAllocationPerformance) -> None:
    eq = alloc.portfolio_equity(method="mtc")
    assert abs(float(eq.iloc[0]) - 1.0) < 1e-9


def test_portfolio_equity_conservative_lte_mtc(alloc: SignalAllocationPerformance) -> None:
    eq_mtc = alloc.portfolio_equity(method="mtc")
    eq_con = alloc.portfolio_equity(method="conservative")
    assert float(eq_con.iloc[-1]) <= float(eq_mtc.iloc[-1]) + 1e-9


def test_portfolio_equity_net_lte_mtc(alloc: SignalAllocationPerformance) -> None:
    eq_mtc = alloc.portfolio_equity(method="mtc")
    eq_net = alloc.portfolio_equity(method="net")
    assert float(eq_net.iloc[-1]) <= float(eq_mtc.iloc[-1]) + 1e-9


def test_portfolio_equity_is_series(alloc: SignalAllocationPerformance) -> None:
    assert isinstance(alloc.portfolio_equity(), pd.Series)


def test_portfolio_returns_length_matches_equity(
    alloc: SignalAllocationPerformance, bars_df: pd.DataFrame
) -> None:
    n_dates = bars_df.index.get_level_values("timestamp").nunique()
    assert len(alloc.portfolio_returns()) == n_dates


def test_portfolio_metrics_returns_performance_metrics(
    alloc: SignalAllocationPerformance,
) -> None:
    pm = alloc.portfolio_metrics()
    assert isinstance(pm, PerformanceMetrics)


def test_portfolio_metrics_sharpe_is_finite(alloc: SignalAllocationPerformance) -> None:
    sharpe = alloc.portfolio_metrics().sharpe
    assert isinstance(sharpe, float)
    assert sharpe == sharpe  # not NaN


# ----------------------------------------------------------------- trade_stats


def test_trade_stats_columns(trades: SignalTradePerformance) -> None:
    cols = set(trades.trade_stats().columns)
    assert {"symbol", "label", "entry_date", "exit_date", "duration",
            "return_net", "return_mtc", "return_conservative",
            "max_intra_drawdown_net", "max_intra_drawdown_mtc",
            "max_intra_drawdown_conservative",
            "d5_net", "d5_mtc", "d5_conservative",
            "d5_to_exit_net", "d5_to_exit_mtc", "d5_to_exit_conservative"}.issubset(cols)


def test_trade_stats_labels_unique(trades: SignalTradePerformance) -> None:
    labels = trades.trade_stats()["label"]
    assert labels.is_unique
    assert (labels.str.contains(r"^\S+ T\d+$", regex=True)).all()


def test_trade_stats_drawdown_bounded(trades: SignalTradePerformance) -> None:
    ts = trades.trade_stats()
    assert (ts["max_intra_drawdown_net"] <= 1e-9).all()
    assert (ts["max_intra_drawdown_mtc"] <= 1e-9).all()
    assert (ts["max_intra_drawdown_conservative"] <= 1e-9).all()


def test_trade_stats_duration_positive(trades: SignalTradePerformance) -> None:
    assert (trades.trade_stats()["duration"] > 0).all()


def test_trade_stats_row_count_matches_entries(
    trades: SignalTradePerformance, alloc: SignalAllocationPerformance
) -> None:
    entries_total = int(alloc.summary()["entries"].sum())
    assert len(trades.trade_stats()) == entries_total


# ----------------------------------------------------------------- trade_summary


def test_trade_summary_index_is_symbols(
    trades: SignalTradePerformance, bars_df: pd.DataFrame
) -> None:
    symbols = bars_df.index.get_level_values("symbol").unique()
    assert set(trades.trade_summary().index) == set(symbols)


def test_trade_summary_columns(trades: SignalTradePerformance) -> None:
    cols = set(trades.trade_summary().columns)
    assert {"n_trades", "win_rate", "avg_win", "avg_loss",
            "expectancy", "expectancy_ex_top", "median_return",
            "profit_factor", "max_win", "max_loss", "skewness",
            "avg_duration", "avg_intra_drawdown", "max_intra_drawdown"}.issubset(cols)


def test_trade_summary_win_rate_bounded(trades: SignalTradePerformance) -> None:
    wr = trades.trade_summary()["win_rate"]
    assert (wr >= 0).all() and (wr <= 1).all()


def test_trade_summary_expectancy_formula(trades: SignalTradePerformance) -> None:
    ts = trades.trade_summary()
    expected = ts["win_rate"] * ts["avg_win"] + (1 - ts["win_rate"]) * ts["avg_loss"]
    pd.testing.assert_series_equal(ts["expectancy"], expected, check_names=False, atol=1e-10)


def test_trade_summary_drawdown_bounded(trades: SignalTradePerformance) -> None:
    ts = trades.trade_summary()
    assert (ts["avg_intra_drawdown"] <= 1e-9).all()
    assert (ts["max_intra_drawdown"] <= 1e-9).all()


# ----------------------------------------------------------------- quality_table


def test_quality_table_multiindex_is_symbol_method(trades: SignalTradePerformance) -> None:
    qt = trades.quality_table()
    assert qt.index.names == ["symbol", "method"]
    methods = qt.index.get_level_values("method").unique().tolist()
    assert set(methods) == {"net", "conservative"}


def test_quality_table_has_d5_and_flag_columns(trades: SignalTradePerformance) -> None:
    cols = set(trades.quality_table().columns)
    assert {"wr_d5", "wr_d5_n", "wr_d5_nwin",
            "wr_tail", "wr_tail_n", "wr_tail_nwin",
            "flags"}.issubset(cols)


def test_quality_table_flags_are_dicts(trades: SignalTradePerformance) -> None:
    qt = trades.quality_table()
    for flags in qt["flags"]:
        assert isinstance(flags, dict)
        assert {"median_negative", "top_trade_outlier", "skewed_right",
                "skewed_left", "edge_reversed", "fill_halves_edge",
                "median_flips"}.issubset(flags.keys())


def test_quality_table_net_rows_have_no_cross_fill_flags(
    trades: SignalTradePerformance,
) -> None:
    qt = trades.quality_table()
    for sym in qt.index.get_level_values("symbol").unique():
        flags = qt.loc[(sym, "net"), "flags"]
        assert flags["edge_reversed"] is False
        assert flags["fill_halves_edge"] is False
        assert flags["median_flips"] is False


# ----------------------------------------------------------------- trade_paths


def test_trade_paths_count_matches_trade_stats(trades: SignalTradePerformance) -> None:
    assert len(trades.trade_paths()) == len(trades.trade_stats())


def test_trade_paths_accepts_precomputed_trade_stats(
    trades: SignalTradePerformance,
) -> None:
    ts = trades.trade_stats()
    assert len(trades.trade_paths(trade_stats=ts)) == len(ts)


def test_trade_paths_cum_starts_at_zero(trades: SignalTradePerformance) -> None:
    for p in trades.trade_paths():
        assert float(p["cum_net"].iloc[0]) == 0.0
        assert float(p["cum_con"].iloc[0]) == 0.0


# ----------------------------------------------------------------- fixed_stake mode


def test_fixed_stake_equity_is_series(alloc: SignalAllocationPerformance) -> None:
    assert isinstance(alloc.portfolio_equity(mode="fixed_stake"), pd.Series)


def test_fixed_stake_equity_starts_at_one(alloc: SignalAllocationPerformance) -> None:
    eq = alloc.portfolio_equity(mode="fixed_stake")
    assert abs(float(eq.iloc[0]) - 1.0) < 1e-9


def test_fixed_stake_equity_no_nan(alloc: SignalAllocationPerformance) -> None:
    eq = alloc.portfolio_equity(mode="fixed_stake")
    assert not eq.isna().any()


def test_fixed_stake_conservative_lte_mtc(alloc: SignalAllocationPerformance) -> None:
    eq_mtc = alloc.portfolio_equity(method="mtc", mode="fixed_stake")
    eq_con = alloc.portfolio_equity(method="conservative", mode="fixed_stake")
    assert float(eq_con.iloc[-1]) <= float(eq_mtc.iloc[-1]) + 1e-9


def test_fixed_stake_scale_linearity(alloc: SignalAllocationPerformance) -> None:
    eq_1k = alloc.portfolio_equity(mode="fixed_stake", amount_per_entry=1_000.0)
    eq_5k = alloc.portfolio_equity(mode="fixed_stake", amount_per_entry=5_000.0)
    pd.testing.assert_series_equal(eq_1k, eq_5k, check_names=False)


def test_fixed_stake_returns_length(
    alloc: SignalAllocationPerformance, bars_df: pd.DataFrame
) -> None:
    n_dates = bars_df.index.get_level_values("timestamp").nunique()
    assert len(alloc.portfolio_returns(mode="fixed_stake")) == n_dates


def test_fixed_stake_metrics_is_performance_metrics(
    alloc: SignalAllocationPerformance,
) -> None:
    pm = alloc.portfolio_metrics(mode="fixed_stake")
    assert isinstance(pm, PerformanceMetrics)
