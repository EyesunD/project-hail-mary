"""Factor-based portfolio construction: quintile, long-short, optimised."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class FactorPortfolio:
    """Translate factor scores into portfolio weights.

    Supports:
    * Quintile / n-tile long-only or long-short portfolios
    * Score-proportional weighting
    * Mean-variance optimisation with factor score as alpha signal
    """

    def __init__(
        self,
        n_quantiles: int = 5,
        construction: Literal["quantile", "score_weighted", "optimised"] = "quantile",
        long_short: bool = False,
        max_weight: float = 0.10,
    ) -> None:
        self.n_quantiles = n_quantiles
        self.construction = construction
        self.long_short = long_short
        self.max_weight = max_weight

    # ------------------------------------------------------------------ API

    def get_weights(
        self,
        scores: pd.Series,
        *,
        cov: pd.DataFrame | None = None,
        risk_aversion: float = 1.0,
    ) -> pd.Series:
        """Return portfolio weights given cross-sectional factor scores.

        Args:
            scores:         Factor z-scores (index=symbol).
            cov:            Covariance matrix for optimised construction.
            risk_aversion:  Lambda for mean-variance objective (higher = less risk).

        Returns:
            Weights summing to 1.0 (long-only) or long leg and short leg
            each summing to ±0.5 if long_short=True.
        """
        scores = scores.dropna()
        if self.construction == "quantile":
            return self._quantile_weights(scores)
        if self.construction == "score_weighted":
            return self._score_weights(scores)
        if self.construction == "optimised":
            if cov is None:
                raise ValueError("Optimised construction requires a covariance matrix.")
            return self._mv_weights(scores, cov, risk_aversion)
        raise ValueError(f"Unknown construction method: {self.construction}")

    def get_quantile_returns(
        self,
        scores: pd.Series,
        forward_returns: pd.Series,
    ) -> pd.Series:
        """Return mean forward return for each quantile bucket."""
        quantile = pd.qcut(scores, self.n_quantiles, labels=False) + 1
        return forward_returns.groupby(quantile).mean().rename("mean_return")

    # ----------------------------------------------------------------- private

    def _quantile_weights(self, scores: pd.Series) -> pd.Series:
        quantile = pd.qcut(scores, self.n_quantiles, labels=False) + 1
        top = quantile == self.n_quantiles
        bottom = quantile == 1
        if self.long_short:
            longs = top.astype(float) / top.sum()
            shorts = -bottom.astype(float) / bottom.sum()
            w = (longs + shorts) * 0.5
        else:
            w = top.astype(float) / top.sum()
        return w.rename("weight")

    def _score_weights(self, scores: pd.Series) -> pd.Series:
        if self.long_short:
            pos = scores.clip(lower=0)
            neg = scores.clip(upper=0)
            longs = pos / pos.sum() * 0.5 if pos.sum() > 0 else pos
            shorts = neg / neg.abs().sum() * -0.5 if neg.sum() < 0 else neg
            w = longs + shorts
        else:
            clipped = scores.clip(lower=0)
            w = clipped / clipped.sum() if clipped.sum() > 0 else clipped
        return w.clip(-self.max_weight, self.max_weight).rename("weight")

    def _mv_weights(
        self,
        scores: pd.Series,
        cov: pd.DataFrame,
        risk_aversion: float,
    ) -> pd.Series:
        symbols = scores.index.intersection(cov.index)
        alpha = scores[symbols].values
        sigma = cov.loc[symbols, symbols].values
        n = len(symbols)

        def objective(w: np.ndarray) -> float:
            return -float(alpha @ w) + risk_aversion * float(w @ sigma @ w)

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(0, self.max_weight)] * n if not self.long_short else [(-self.max_weight, self.max_weight)] * n
        w0 = np.ones(n) / n
        result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        return pd.Series(result.x, index=symbols, name="weight")
