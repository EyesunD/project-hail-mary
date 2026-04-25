"""Multi-factor model: combine signals, run IC analysis, build factor portfolios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from hailmary.models.factors.base import Factor, FactorScore


@dataclass
class ICStats:
    """Information coefficient summary statistics over a backtest period."""

    factor_name: str
    mean_ic: float
    ic_std: float
    icir: float          # IC / IC_std
    t_stat: float
    p_value: float
    pct_positive: float
    hit_rate: float      # fraction of periods with |IC| > 0.05

    @classmethod
    def from_series(cls, name: str, ic_series: pd.Series) -> "ICStats":
        s = ic_series.dropna()
        t, p = stats.ttest_1samp(s, 0)
        return cls(
            factor_name=name,
            mean_ic=float(s.mean()),
            ic_std=float(s.std()),
            icir=float(s.mean() / s.std()) if s.std() > 0 else 0.0,
            t_stat=float(t),
            p_value=float(p),
            pct_positive=float((s > 0).mean()),
            hit_rate=float((s.abs() > 0.05).mean()),
        )

    def __str__(self) -> str:
        return (
            f"{self.factor_name}: IC={self.mean_ic:.3f}  ICIR={self.icir:.2f}  "
            f"t={self.t_stat:.2f}  p={self.p_value:.3f}  hit={self.hit_rate:.1%}"
        )


class MultiFactorModel:
    """Combine multiple :class:`~hailmary.models.factors.base.Factor` signals.

    Supports:
    * Equal-weight and custom-weight combination
    * Rank-based and z-score-based aggregation
    * Trailing IC-weighted combination (dynamic weighting)
    * IC analysis and signal decay curves
    """

    def __init__(
        self,
        factors: list[Factor],
        weights: list[float] | None = None,
        combination: Literal["equal", "custom", "ic_weighted"] = "equal",
        ic_window: int = 12,
    ) -> None:
        self.factors = factors
        self._base_weights = weights or [1.0] * len(factors)
        self.combination = combination
        self.ic_window = ic_window
        self._ic_history: dict[str, list[float]] = {f.name: [] for f in factors}

    # ------------------------------------------------------------------ API

    def score(
        self,
        prices: pd.DataFrame,
        method: Literal["zscore", "rank"] = "zscore",
        **factor_kwargs: pd.DataFrame,
    ) -> pd.Series:
        """Compute composite score for all symbols at the latest date.

        Returns a Series (index=symbol) with composite factor scores.
        """
        factor_scores: list[pd.Series] = []
        for factor in self.factors:
            try:
                fs: FactorScore = factor.compute(prices, **factor_kwargs)
                signal = fs.z_scores if method == "zscore" else fs.ranks
                factor_scores.append(signal.rename(factor.name))
            except Exception as exc:
                logger.warning("Factor '{}' failed: {}", factor.name, exc)

        if not factor_scores:
            raise RuntimeError("All factors failed to compute.")

        df = pd.concat(factor_scores, axis=1)
        weights = self._resolve_weights(df.columns.tolist())
        w = pd.Series(weights, index=df.columns)
        composite = df.mul(w).sum(axis=1) / w.sum()
        composite.name = "composite"
        return composite

    def score_panel(
        self,
        prices: pd.DataFrame,
        frequency: str = "ME",
        method: Literal["zscore", "rank"] = "zscore",
        **factor_kwargs: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute composite scores at each rebalance date.

        Returns a DataFrame (index=date, columns=symbol).
        """
        dates = prices.resample(frequency).last().index
        records: dict[pd.Timestamp, pd.Series] = {}
        for dt in dates:
            hist = prices.loc[:dt]
            if len(hist) < 50:
                continue
            try:
                records[dt] = self.score(hist, method=method, **factor_kwargs)
            except Exception as exc:
                logger.debug("Skipping {}: {}", dt, exc)
        return pd.DataFrame(records).T

    def compute_ic(
        self,
        score_panel: pd.DataFrame,
        forward_returns: pd.DataFrame,
        horizon: int = 21,
    ) -> dict[str, pd.Series]:
        """Compute IC time series for each individual factor.

        Args:
            score_panel: (date × symbol) composite score panel.
            forward_returns: (date × symbol) forward-return panel.
            horizon: Forward return horizon in *rows* (trading days).

        Returns:
            Dict mapping factor name → IC time series.
        """
        fwd = forward_returns.shift(-horizon)  # align with score dates
        ic_series: dict[str, pd.Series] = {}
        for factor in self.factors:
            factor_panel = factor.compute_panel(
                prices=score_panel,  # crude — callers should pass full prices
                frequency="ME",
            )
            ics = []
            for dt in factor_panel.index:
                if dt not in fwd.index:
                    continue
                s = factor_panel.loc[dt].dropna()
                r = fwd.loc[dt].dropna()
                common = s.index.intersection(r.index)
                if len(common) < 5:
                    ics.append(float("nan"))
                    continue
                ic, _ = stats.spearmanr(s[common], r[common])
                ics.append(ic)
            ic_series[factor.name] = pd.Series(ics, index=factor_panel.index[:len(ics)])
        return ic_series

    def ic_stats(self, ic_series: dict[str, pd.Series]) -> list[ICStats]:
        return [ICStats.from_series(name, s) for name, s in ic_series.items()]

    # ----------------------------------------------------------------- private

    def _resolve_weights(self, factor_names: list[str]) -> list[float]:
        if self.combination == "equal":
            return [1.0] * len(factor_names)
        if self.combination == "custom":
            name_to_w = dict(zip([f.name for f in self.factors], self._base_weights))
            return [name_to_w.get(n, 1.0) for n in factor_names]
        # ic_weighted: use trailing mean IC as weight
        weights = []
        for n in factor_names:
            hist = self._ic_history.get(n, [])
            w = float(np.mean(hist[-self.ic_window:])) if hist else 1.0
            weights.append(max(w, 0))  # no negative weights
        return weights if any(w > 0 for w in weights) else [1.0] * len(factor_names)
