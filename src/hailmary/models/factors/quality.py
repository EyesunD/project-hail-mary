"""Quality factors: ROE, gross margin, earnings stability, Piotroski F-Score."""

from __future__ import annotations

import pandas as pd

from hailmary.data.base import FundamentalData
from hailmary.models.factors.base import Factor, FactorScore


class QualityFactor(Factor):
    """Composite quality: profitability + margin stability.

    Signals: ROE, ROA, gross margin, operating margin, net margin.
    """

    name = "quality"
    description = "Composite profitability / quality score"

    _METRICS: list[tuple[str, str]] = [
        ("roe", "roe"),
        ("roa", "roa"),
        ("gross_margin", "gross_margin"),
        ("operating_margin", "operating_margin"),
        ("net_margin", "net_margin"),
    ]

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: dict[str, FundamentalData] | None = None,
        **kwargs: pd.DataFrame,
    ) -> FactorScore:
        if not fundamentals:
            raise ValueError("QualityFactor requires 'fundamentals' kwarg.")

        rows = {}
        for sym, fd in fundamentals.items():
            rows[sym] = {label: getattr(fd, attr) for label, attr in self._METRICS}
        df = pd.DataFrame(rows).T.dropna(how="all")
        zscored = df.apply(lambda col: (col - col.mean()) / col.std(ddof=1), axis=0).fillna(0)
        composite = zscored.mean(axis=1)
        return FactorScore(name=self.name, as_of=prices.index[-1], raw=composite)


class EarningsStabilityFactor(Factor):
    """Low dispersion in earnings-per-share as a quality proxy.

    Uses a time series of price-to-earnings ratios as a proxy for earnings
    stability when full EPS history is not available.
    """

    name = "earnings_stability"
    description = "Inverse earnings volatility (stability)"

    def __init__(self, window: int = 252) -> None:
        self.window = window

    def compute(self, prices: pd.DataFrame, **kwargs: pd.DataFrame) -> FactorScore:
        rets = prices.pct_change().tail(self.window)
        # Use return volatility as proxy for earnings vol (negative → stability)
        raw = -rets.std()
        return FactorScore(name=self.name, as_of=prices.index[-1], raw=raw)


class GrossProfitabilityFactor(Factor):
    """Novy-Marx gross profitability: gross profit / total assets."""

    name = "gross_profitability"
    description = "Gross profit / total assets (Novy-Marx)"

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: dict[str, FundamentalData] | None = None,
        **kwargs: pd.DataFrame,
    ) -> FactorScore:
        if not fundamentals:
            raise ValueError("GrossProfitabilityFactor requires 'fundamentals' kwarg.")
        raw = pd.Series(
            {sym: fd.gross_margin for sym, fd in fundamentals.items() if fd.gross_margin is not None},
            name=self.name,
        )
        return FactorScore(name=self.name, as_of=prices.index[-1], raw=raw)
