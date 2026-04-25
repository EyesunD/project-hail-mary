"""Momentum factors: 12-1, residual momentum, short-term reversal."""

from __future__ import annotations

import pandas as pd

from hailmary.models.factors.base import Factor, FactorScore


class MomentumFactor(Factor):
    """Classic cross-sectional momentum: trailing return skipping most recent month.

    Default lookback=252 trading days (12 months), skip=21 (1 month).
    """

    name = "momentum"
    description = "12-1 month price momentum"

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        self.lookback = lookback
        self.skip = skip

    def compute(self, prices: pd.DataFrame, **kwargs: pd.DataFrame) -> FactorScore:
        if len(prices) < self.lookback + self.skip:
            raise ValueError(f"Need at least {self.lookback + self.skip} bars.")
        end = prices.iloc[-(self.skip + 1)]
        start = prices.iloc[-self.lookback]
        raw = (end / start - 1).rename(self.name)
        return FactorScore(name=self.name, as_of=prices.index[-1], raw=raw)


class ShortTermReversalFactor(Factor):
    """1-month return reversal (negative momentum at short horizon)."""

    name = "st_reversal"
    description = "1-month short-term reversal"

    def __init__(self, lookback: int = 21) -> None:
        self.lookback = lookback

    def compute(self, prices: pd.DataFrame, **kwargs: pd.DataFrame) -> FactorScore:
        raw = -(prices.iloc[-1] / prices.iloc[-self.lookback] - 1).rename(self.name)
        return FactorScore(name=self.name, as_of=prices.index[-1], raw=raw)


class ResidualMomentumFactor(Factor):
    """Residual momentum: momentum orthogonal to market (Blitz et al.)."""

    name = "residual_momentum"
    description = "Market-beta-adjusted momentum"

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        self.lookback = lookback
        self.skip = skip

    def compute(self, prices: pd.DataFrame, **kwargs: pd.DataFrame) -> FactorScore:
        from sklearn.linear_model import LinearRegression  # noqa: PLC0415

        rets = prices.pct_change().dropna()
        if len(rets) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} return observations.")

        window = rets.iloc[-self.lookback : -self.skip if self.skip else None]
        mkt = window.mean(axis=1)  # equal-weight market proxy
        residuals: dict[str, float] = {}
        for sym in window.columns:
            y = window[sym].values.reshape(-1, 1)
            X = mkt.values.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            resid = y.flatten() - reg.predict(X).flatten()
            # cumulative residual return as signal
            residuals[sym] = resid.sum()

        raw = pd.Series(residuals, name=self.name)
        return FactorScore(name=self.name, as_of=prices.index[-1], raw=raw)
