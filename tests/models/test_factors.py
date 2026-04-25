"""Tests for factor implementations."""

import numpy as np
import pandas as pd
import pytest

from hailmary.models.factors import MomentumFactor, VolatilityFactor
from hailmary.models.factors.base import FactorScore
from hailmary.models.multi_factor import MultiFactorModel


def test_momentum_factor(price_df: pd.DataFrame) -> None:
    factor = MomentumFactor(lookback=252, skip=21)
    score = factor.compute(price_df)
    assert isinstance(score, FactorScore)
    assert len(score.raw) == price_df.shape[1]
    assert not score.z_scores.isna().all()


def test_momentum_insufficient_data() -> None:
    small_df = pd.DataFrame({"A": [100, 101, 102]}, index=pd.bdate_range("2020-01-01", periods=3))
    with pytest.raises(ValueError):
        MomentumFactor().compute(small_df)


def test_volatility_factor(price_df: pd.DataFrame) -> None:
    factor = VolatilityFactor(window=63)
    score = factor.compute(price_df)
    assert score.raw.lt(0).all(), "Low-vol factor should be negative of vol"


def test_factor_score_winsorise(price_df: pd.DataFrame) -> None:
    score = MomentumFactor(lookback=252, skip=21).compute(price_df)
    ws = score.winsorize(0.05, 0.95)
    assert ws.raw.min() >= score.raw.quantile(0.05) - 1e-9
    assert ws.raw.max() <= score.raw.quantile(0.95) + 1e-9


def test_multi_factor_score(price_df: pd.DataFrame) -> None:
    model = MultiFactorModel([MomentumFactor(lookback=252, skip=21), VolatilityFactor(window=63)])
    composite = model.score(price_df)
    assert len(composite) == price_df.shape[1]
    assert not composite.isna().all()


def test_multi_factor_panel(price_df: pd.DataFrame) -> None:
    model = MultiFactorModel([MomentumFactor(lookback=252, skip=21)])
    panel = model.score_panel(price_df, frequency="ME")
    assert panel.shape[1] == price_df.shape[1]
    assert len(panel) > 0
