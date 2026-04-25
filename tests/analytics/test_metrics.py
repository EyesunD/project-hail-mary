"""Tests for PerformanceMetrics."""

import numpy as np
import pandas as pd
import pytest

from hailmary.analytics.metrics import PerformanceMetrics


def test_total_return_positive(return_series: pd.Series) -> None:
    m = PerformanceMetrics(return_series)
    # With positive mean return, total return should be positive
    assert m.total_return > 0


def test_max_drawdown_negative(return_series: pd.Series) -> None:
    m = PerformanceMetrics(return_series)
    assert m.max_drawdown <= 0


def test_sharpe_reasonable(return_series: pd.Series) -> None:
    m = PerformanceMetrics(return_series)
    # Sharpe should be finite and within a reasonable range for random data
    assert -10 < m.sharpe < 10


def test_var_less_than_cvar(return_series: pd.Series) -> None:
    m = PerformanceMetrics(return_series)
    assert m.cvar_95 <= m.var_95  # CVaR is more extreme


def test_summary_returns_series(return_series: pd.Series) -> None:
    m = PerformanceMetrics(return_series)
    s = m.summary()
    assert isinstance(s, pd.Series)
    assert "Sharpe Ratio" in s.index
    assert "Max Drawdown" in s.index


def test_rolling_sharpe_length(return_series: pd.Series) -> None:
    m = PerformanceMetrics(return_series)
    rs = m.rolling_sharpe(window=63)
    assert len(rs) == len(return_series)


def test_monthly_returns_shape(return_series: pd.Series) -> None:
    m = PerformanceMetrics(return_series)
    monthly = m.monthly_returns()
    assert monthly.columns.tolist() == list(range(1, 13)) or all(isinstance(c, int) for c in monthly.columns)
