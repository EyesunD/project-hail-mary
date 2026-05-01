"""Tests for SignalComparison."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from hailmary.analytics.signal_comparison import SignalComparison
from hailmary.backtest.signal_backtest import BarBacktest, BarBacktestResult
from hailmary.models.signals import TrendSignal


@pytest.fixture
def bt_result(bars_df: pd.DataFrame) -> BarBacktestResult:
    signal_df = TrendSignal(ma_window=200).run(bars_df)
    return BarBacktest().run(signal_df)


@pytest.fixture
def comparison(bt_result: BarBacktestResult) -> SignalComparison:
    return SignalComparison({"Base": bt_result, "Copy": bt_result})


# ----------------------------------------------------------------- pooled_table


def test_pooled_table_index(comparison: SignalComparison) -> None:
    assert set(comparison.pooled_table().index) == {"Base", "Copy"}


def test_pooled_table_columns(comparison: SignalComparison) -> None:
    cols = set(comparison.pooled_table().columns)
    assert {"n_trades", "win_rate", "expectancy", "profit_factor",
            "avg_intra_drawdown", "max_intra_drawdown"}.issubset(cols)


def test_pooled_table_identical_variants_match(comparison: SignalComparison) -> None:
    tbl = comparison.pooled_table()
    pd.testing.assert_series_equal(tbl.loc["Base"], tbl.loc["Copy"], check_names=False)


def test_pooled_table_conservative_lte_net(comparison: SignalComparison) -> None:
    net = comparison.pooled_table("net")
    con = comparison.pooled_table("conservative")
    assert float(con.loc["Base", "expectancy"]) <= float(net.loc["Base", "expectancy"]) + 1e-9


# ----------------------------------------------------------------- symbol_table


def test_symbol_table_index_levels(
    comparison: SignalComparison, bars_df: pd.DataFrame
) -> None:
    tbl = comparison.symbol_table()
    assert tbl.index.names == ["variant", "symbol"]
    symbols = set(bars_df.index.get_level_values("symbol").unique())
    assert set(tbl.index.get_level_values("symbol")) == symbols


def test_symbol_table_variants(comparison: SignalComparison) -> None:
    tbl = comparison.symbol_table()
    assert set(tbl.index.get_level_values("variant")) == {"Base", "Copy"}


def test_symbol_table_identical_variants_match(comparison: SignalComparison) -> None:
    tbl = comparison.symbol_table()
    for sym in tbl.index.get_level_values("symbol").unique():
        pd.testing.assert_series_equal(
            tbl.loc[("Base", sym)], tbl.loc[("Copy", sym)], check_names=False
        )


# ----------------------------------------------------------------- tearsheet


def test_tearsheet_returns_figure(comparison: SignalComparison) -> None:
    assert isinstance(comparison.tearsheet(), go.Figure)


def test_tearsheet_single_variant(bt_result: BarBacktestResult) -> None:
    fig = SignalComparison({"Only": bt_result}).tearsheet()
    assert isinstance(fig, go.Figure)
