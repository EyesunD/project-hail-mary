"""Value factors: P/B, P/E, P/S, EV/EBITDA composite."""

from __future__ import annotations

import pandas as pd

from hailmary.data.base import FundamentalData
from hailmary.models.factors.base import Factor, FactorScore


def _invert(s: pd.Series) -> pd.Series:
    """Flip valuation ratio to a value score (lower ratio → higher score)."""
    return -s


class BookToMarketFactor(Factor):
    """Book-to-Market (inverse of P/B) — the canonical HML value factor."""

    name = "book_to_market"
    description = "Book-to-market ratio (1 / P/B)"

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: dict[str, FundamentalData] | None = None,
        **kwargs: pd.DataFrame,
    ) -> FactorScore:
        if not fundamentals:
            raise ValueError("BookToMarketFactor requires 'fundamentals' kwarg.")
        raw = pd.Series(
            {sym: 1.0 / fd.pb_ratio for sym, fd in fundamentals.items() if fd.pb_ratio},
            name=self.name,
        )
        return FactorScore(name=self.name, as_of=prices.index[-1], raw=raw)


class EarningsYieldFactor(Factor):
    """Earnings yield (1 / P/E) — a forward-earnings value proxy."""

    name = "earnings_yield"
    description = "Earnings yield (1 / trailing P/E)"

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: dict[str, FundamentalData] | None = None,
        **kwargs: pd.DataFrame,
    ) -> FactorScore:
        if not fundamentals:
            raise ValueError("EarningsYieldFactor requires 'fundamentals' kwarg.")
        raw = pd.Series(
            {sym: 1.0 / fd.pe_ratio for sym, fd in fundamentals.items() if fd.pe_ratio},
            name=self.name,
        )
        return FactorScore(name=self.name, as_of=prices.index[-1], raw=raw)


class ValueFactor(Factor):
    """Composite value factor: equal-weight average of available value signals."""

    name = "value"
    description = "Multi-metric composite value score (B/M, E/P, S/P)"

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {"book_to_market": 1.0, "earnings_yield": 1.0, "sales_yield": 1.0}

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: dict[str, FundamentalData] | None = None,
        **kwargs: pd.DataFrame,
    ) -> FactorScore:
        if not fundamentals:
            raise ValueError("ValueFactor requires 'fundamentals' kwarg.")

        signals: dict[str, pd.Series] = {}
        for sym, fd in fundamentals.items():
            row: dict[str, float] = {}
            if fd.pb_ratio and fd.pb_ratio > 0:
                row["book_to_market"] = 1.0 / fd.pb_ratio
            if fd.pe_ratio and fd.pe_ratio > 0:
                row["earnings_yield"] = 1.0 / fd.pe_ratio
            if fd.ps_ratio and fd.ps_ratio > 0:
                row["sales_yield"] = 1.0 / fd.ps_ratio
            signals[sym] = pd.Series(row)

        df = pd.DataFrame(signals).T  # (sym × metric)
        # Z-score each metric then combine
        zscored = df.apply(lambda col: (col - col.mean()) / col.std(ddof=1), axis=0)
        metric_weights = pd.Series(self.weights)
        available = metric_weights.reindex(zscored.columns).fillna(0)
        composite = zscored.mul(available).sum(axis=1) / available.sum()
        return FactorScore(name=self.name, as_of=prices.index[-1], raw=composite)
