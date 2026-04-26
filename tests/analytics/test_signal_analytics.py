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
    assert {"return_mtc", "return_conservative", "return_net", "entries", "exits",
            "invested_days", "pct_invested"}.issubset(cols)


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
