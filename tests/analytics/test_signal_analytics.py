"""Tests for SignalAnalytics."""

from __future__ import annotations

import pandas as pd
import pytest

from hailmary.analytics.metrics import PerformanceMetrics
from hailmary.analytics.signal_analytics import SignalAnalytics
from hailmary.backtest.signal_backtest import BarBacktest, BarBacktestResult
from hailmary.models.signals import TrendSignal


@pytest.fixture
def bt_result(bars_df: pd.DataFrame) -> BarBacktestResult:
    signal_df = TrendSignal(ma_window=200).run(bars_df)
    return BarBacktest().run(signal_df)


@pytest.fixture
def analytics(bt_result: BarBacktestResult) -> SignalAnalytics:
    return SignalAnalytics(bt_result)


# ----------------------------------------------------------------- per-symbol


def test_summary_index_is_symbols(
    analytics: SignalAnalytics, bars_df: pd.DataFrame
) -> None:
    symbols = bars_df.index.get_level_values("symbol").unique()
    assert set(analytics.summary().index) == set(symbols)


def test_summary_columns(analytics: SignalAnalytics) -> None:
    cols = set(analytics.summary().columns)
    assert {
        "return_mtc", "return_conservative", "return_net",
        "max_drawdown_mtc", "avg_drawdown_mtc",
        "max_drawdown_conservative", "avg_drawdown_conservative",
        "entries", "exits", "invested_days", "pct_invested",
    }.issubset(cols)


def test_summary_drawdown_bounded(analytics: SignalAnalytics) -> None:
    s = analytics.summary()
    assert (s["max_drawdown_mtc"] <= 0).all()
    assert (s["max_drawdown_conservative"] <= 0).all()
    assert (s["avg_drawdown_mtc"] <= 0).all()
    assert (s["avg_drawdown_conservative"] <= 0).all()


def test_summary_conservative_dd_lte_mtc(analytics: SignalAnalytics) -> None:
    s = analytics.summary()
    assert (s["max_drawdown_conservative"] <= s["max_drawdown_mtc"] + 1e-9).all()


def test_equity_mtc_shape(analytics: SignalAnalytics, bars_df: pd.DataFrame) -> None:
    eq = analytics.equity(method="mtc")
    n_symbols = bars_df.index.get_level_values("symbol").nunique()
    assert eq.shape[1] == n_symbols
    assert (eq.iloc[0] == 1.0).all()


def test_equity_conservative_lte_mtc(analytics: SignalAnalytics) -> None:
    eq_mtc = analytics.equity(method="mtc")
    eq_con = analytics.equity(method="conservative")
    assert (eq_con.iloc[-1] <= eq_mtc.iloc[-1]).all()


def test_pct_invested_bounded(analytics: SignalAnalytics) -> None:
    pct = analytics.summary()["pct_invested"]
    assert (pct >= 0).all() and (pct <= 100).all()


# ----------------------------------------------------------------- portfolio


def test_portfolio_equity_starts_at_one(analytics: SignalAnalytics) -> None:
    eq = analytics.portfolio_equity(method="mtc")
    assert abs(float(eq.iloc[0]) - 1.0) < 1e-9


def test_portfolio_equity_conservative_lte_mtc(analytics: SignalAnalytics) -> None:
    eq_mtc = analytics.portfolio_equity(method="mtc")
    eq_con = analytics.portfolio_equity(method="conservative")
    assert float(eq_con.iloc[-1]) <= float(eq_mtc.iloc[-1]) + 1e-9


def test_portfolio_equity_net_lte_mtc(analytics: SignalAnalytics) -> None:
    eq_mtc = analytics.portfolio_equity(method="mtc")
    eq_net = analytics.portfolio_equity(method="net")
    assert float(eq_net.iloc[-1]) <= float(eq_mtc.iloc[-1]) + 1e-9


def test_portfolio_equity_is_series(analytics: SignalAnalytics) -> None:
    assert isinstance(analytics.portfolio_equity(), pd.Series)


def test_portfolio_returns_length_matches_equity(
    analytics: SignalAnalytics, bars_df: pd.DataFrame
) -> None:
    n_dates = bars_df.index.get_level_values("timestamp").nunique()
    assert len(analytics.portfolio_returns()) == n_dates


def test_portfolio_metrics_returns_performance_metrics(analytics: SignalAnalytics) -> None:
    pm = analytics.portfolio_metrics()
    assert isinstance(pm, PerformanceMetrics)


def test_portfolio_metrics_sharpe_is_finite(analytics: SignalAnalytics) -> None:
    sharpe = analytics.portfolio_metrics().sharpe
    assert isinstance(sharpe, float)
    assert sharpe == sharpe  # not NaN


# ----------------------------------------------------------------- trade_stats


def test_trade_stats_columns(analytics: SignalAnalytics) -> None:
    cols = set(analytics.trade_stats().columns)
    assert {"symbol", "entry_date", "exit_date", "duration",
            "return_net", "return_mtc", "return_conservative",
            "max_intra_drawdown_net", "max_intra_drawdown_mtc",
            "max_intra_drawdown_conservative"}.issubset(cols)


def test_trade_stats_drawdown_bounded(analytics: SignalAnalytics) -> None:
    ts = analytics.trade_stats()
    assert (ts["max_intra_drawdown_net"] <= 1e-9).all()
    assert (ts["max_intra_drawdown_mtc"] <= 1e-9).all()
    assert (ts["max_intra_drawdown_conservative"] <= 1e-9).all()


def test_trade_stats_duration_positive(analytics: SignalAnalytics) -> None:
    assert (analytics.trade_stats()["duration"] > 0).all()


def test_trade_stats_row_count_matches_entries(analytics: SignalAnalytics) -> None:
    entries_total = int(analytics.summary()["entries"].sum())
    assert len(analytics.trade_stats()) == entries_total


# ----------------------------------------------------------------- trade_summary


def test_trade_summary_index_is_symbols(
    analytics: SignalAnalytics, bars_df: pd.DataFrame
) -> None:
    symbols = bars_df.index.get_level_values("symbol").unique()
    assert set(analytics.trade_summary().index) == set(symbols)


def test_trade_summary_columns(analytics: SignalAnalytics) -> None:
    cols = set(analytics.trade_summary().columns)
    assert {"n_trades", "win_rate", "avg_win", "avg_loss",
            "expectancy", "expectancy_ex_top", "median_return",
            "profit_factor", "max_win", "max_loss", "skewness",
            "avg_duration", "avg_intra_drawdown", "max_intra_drawdown"}.issubset(cols)


def test_trade_summary_win_rate_bounded(analytics: SignalAnalytics) -> None:
    wr = analytics.trade_summary()["win_rate"]
    assert (wr >= 0).all() and (wr <= 1).all()


def test_trade_summary_expectancy_formula(analytics: SignalAnalytics) -> None:
    ts = analytics.trade_summary()
    expected = ts["win_rate"] * ts["avg_win"] + (1 - ts["win_rate"]) * ts["avg_loss"]
    pd.testing.assert_series_equal(ts["expectancy"], expected, check_names=False, atol=1e-10)


def test_trade_summary_drawdown_bounded(analytics: SignalAnalytics) -> None:
    ts = analytics.trade_summary()
    assert (ts["avg_intra_drawdown"] <= 1e-9).all()
    assert (ts["max_intra_drawdown"] <= 1e-9).all()


# ----------------------------------------------------------------- fixed_stake mode


def test_fixed_stake_equity_is_series(analytics: SignalAnalytics) -> None:
    assert isinstance(analytics.portfolio_equity(mode="fixed_stake"), pd.Series)


def test_fixed_stake_equity_starts_at_one(analytics: SignalAnalytics) -> None:
    eq = analytics.portfolio_equity(mode="fixed_stake")
    assert abs(float(eq.iloc[0]) - 1.0) < 1e-9


def test_fixed_stake_equity_no_nan(analytics: SignalAnalytics) -> None:
    eq = analytics.portfolio_equity(mode="fixed_stake")
    assert not eq.isna().any()


def test_fixed_stake_conservative_lte_mtc(analytics: SignalAnalytics) -> None:
    eq_mtc = analytics.portfolio_equity(method="mtc", mode="fixed_stake")
    eq_con = analytics.portfolio_equity(method="conservative", mode="fixed_stake")
    assert float(eq_con.iloc[-1]) <= float(eq_mtc.iloc[-1]) + 1e-9


def test_fixed_stake_scale_linearity(analytics: SignalAnalytics) -> None:
    eq_1k = analytics.portfolio_equity(mode="fixed_stake", amount_per_entry=1_000.0)
    eq_5k = analytics.portfolio_equity(mode="fixed_stake", amount_per_entry=5_000.0)
    pd.testing.assert_series_equal(eq_1k, eq_5k, check_names=False)


def test_fixed_stake_returns_length(analytics: SignalAnalytics, bars_df: pd.DataFrame) -> None:
    n_dates = bars_df.index.get_level_values("timestamp").nunique()
    assert len(analytics.portfolio_returns(mode="fixed_stake")) == n_dates


def test_fixed_stake_metrics_is_performance_metrics(analytics: SignalAnalytics) -> None:
    pm = analytics.portfolio_metrics(mode="fixed_stake")
    assert isinstance(pm, PerformanceMetrics)
