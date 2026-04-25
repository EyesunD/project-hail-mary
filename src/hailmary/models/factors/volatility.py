"""Volatility / low-risk factors: realised vol, beta, idiosyncratic risk."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from hailmary.models.factors.base import Factor, FactorScore


class VolatilityFactor(Factor):
    """Low-volatility factor: negative realised volatility (lower is better).

    The negative is intentional — lower vol stocks tend to outperform
    on a risk-adjusted basis (the low-vol anomaly).
    """

    name = "low_volatility"
    description = "Negative trailing realised volatility (low-vol anomaly)"

    def __init__(self, window: int = 63, annualise: bool = True) -> None:
        self.window = window
        self.annualise = annualise

    def compute(self, prices: pd.DataFrame, **kwargs: pd.DataFrame) -> FactorScore:
        rets = prices.pct_change().tail(self.window + 1).dropna()
        vol = rets.std()
        if self.annualise:
            vol = vol * np.sqrt(252)
        return FactorScore(name=self.name, as_of=prices.index[-1], raw=-vol)


class MarketBetaFactor(Factor):
    """Low-beta factor: negative OLS beta vs equal-weight market proxy."""

    name = "low_beta"
    description = "Negative market beta (low-beta anomaly)"

    def __init__(self, window: int = 252) -> None:
        self.window = window

    def compute(self, prices: pd.DataFrame, **kwargs: pd.DataFrame) -> FactorScore:
        rets = prices.pct_change().dropna().tail(self.window)
        mkt = rets.mean(axis=1)
        betas: dict[str, float] = {}
        X = mkt.values.reshape(-1, 1)
        for sym in rets.columns:
            y = rets[sym].values
            mask = ~np.isnan(y)
            if mask.sum() < 30:
                betas[sym] = float("nan")
                continue
            reg = LinearRegression().fit(X[mask], y[mask])
            betas[sym] = float(reg.coef_[0])
        raw = pd.Series(betas, name=self.name)
        return FactorScore(name=self.name, as_of=prices.index[-1], raw=-raw)


class IdiosyncraticVolFactor(Factor):
    """Idiosyncratic (residual) volatility — after stripping market beta."""

    name = "idio_vol"
    description = "Negative idiosyncratic volatility (AHXZ)"

    def __init__(self, window: int = 63) -> None:
        self.window = window

    def compute(self, prices: pd.DataFrame, **kwargs: pd.DataFrame) -> FactorScore:
        rets = prices.pct_change().dropna().tail(self.window)
        mkt = rets.mean(axis=1)
        X = mkt.values.reshape(-1, 1)
        idio_vols: dict[str, float] = {}
        for sym in rets.columns:
            y = rets[sym].values
            mask = ~np.isnan(y)
            if mask.sum() < 20:
                idio_vols[sym] = float("nan")
                continue
            reg = LinearRegression().fit(X[mask], y[mask])
            resid = y[mask] - reg.predict(X[mask]).flatten()
            idio_vols[sym] = float(resid.std())
        raw = pd.Series(idio_vols, name=self.name)
        return FactorScore(name=self.name, as_of=prices.index[-1], raw=-raw)
