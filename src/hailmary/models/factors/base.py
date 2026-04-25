"""Abstract Factor base class and FactorScore container."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FactorScore:
    """Cross-sectional factor scores at a single point in time.

    Attributes:
        name:      Factor name.
        as_of:     The date these scores apply to.
        raw:       Raw, unnormalised signal values keyed by symbol.
        z_scores:  Cross-sectionally z-scored values (mean=0, std=1).
        ranks:     Percentile ranks [0, 1].
        ic:        Information coefficient vs forward returns (filled after evaluation).
    """

    name: str
    as_of: pd.Timestamp
    raw: pd.Series
    z_scores: pd.Series = field(default_factory=pd.Series)
    ranks: pd.Series = field(default_factory=pd.Series)
    ic: float | None = None

    def __post_init__(self) -> None:
        if self.raw is not None and len(self.raw) > 0:
            self.z_scores = self._zscore(self.raw)
            self.ranks = self.raw.rank(pct=True)

    @staticmethod
    def _zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std(ddof=1)

    def winsorize(self, lower: float = 0.01, upper: float = 0.99) -> FactorScore:
        """Return a new FactorScore with raw values winsorized at [lower, upper] quantiles."""
        lo, hi = self.raw.quantile(lower), self.raw.quantile(upper)
        winsorized = self.raw.clip(lo, hi)
        return FactorScore(name=self.name, as_of=self.as_of, raw=winsorized)

    def neutralize(self, groups: pd.Series) -> FactorScore:
        """Group-demean the raw scores (e.g. sector-neutralize)."""
        demeaned = self.raw - self.raw.groupby(groups).transform("mean")
        return FactorScore(name=self.name, as_of=self.as_of, raw=demeaned)

    def compute_ic(self, forward_returns: pd.Series) -> float:
        """Spearman rank IC vs forward returns (in-place and returned)."""
        aligned = pd.concat([self.raw, forward_returns], axis=1).dropna()
        if len(aligned) < 5:
            return float("nan")
        ic, _ = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        self.ic = ic
        return ic


class Factor(abc.ABC):
    """Abstract factor.

    Subclass and implement :meth:`compute` to define a new alpha signal.
    The method receives a wide close-price DataFrame and any additional
    data needed, and must return a :class:`FactorScore`.
    """

    name: str = "unnamed_factor"
    description: str = ""

    # ------------------------------------------------------------------ API

    @abc.abstractmethod
    def compute(
        self,
        prices: pd.DataFrame,
        **kwargs: pd.DataFrame,
    ) -> FactorScore:
        """Compute cross-sectional factor scores.

        Args:
            prices: Wide DataFrame of adjusted close prices
                    (index=date, columns=symbol).
            **kwargs: Supplemental DataFrames the factor may need
                      (e.g. ``fundamentals``, ``volume``).

        Returns:
            FactorScore for the *last* date in *prices*.
        """

    def compute_panel(
        self,
        prices: pd.DataFrame,
        frequency: str = "ME",
        **kwargs: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute scores at each rebalance date and return a panel.

        Returns a DataFrame (index=date, columns=symbol) of z-scores.
        """
        dates = prices.resample(frequency).last().index
        records: dict[pd.Timestamp, pd.Series] = {}
        for dt in dates:
            hist = prices.loc[:dt]
            if len(hist) < 2:
                continue
            score = self.compute(hist, **kwargs)
            records[dt] = score.z_scores
        return pd.DataFrame(records).T

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        return prices.pct_change(periods)

    @staticmethod
    def _log_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        return np.log(prices / prices.shift(periods))

    def __repr__(self) -> str:
        return f"<Factor name={self.name!r}>"
