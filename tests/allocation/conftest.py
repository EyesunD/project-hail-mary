"""Synthetic fixtures for allocation tests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pytest

from hailmary.allocation.portfolios import Holding, Portfolio, Role
from hailmary.allocation.universe import STASHAWAY_UNIVERSE, AssetMetadata


def _meta(ticker: str, asset_class: str, region: str, sector: str | None = None) -> AssetMetadata:
    return AssetMetadata(ticker=ticker, asset_class=asset_class, region=region, sector=sector)


@pytest.fixture
def seeded_universe() -> dict[str, AssetMetadata]:
    """Seed STASHAWAY_UNIVERSE with a small synthetic set; restore on teardown."""
    backup = dict(STASHAWAY_UNIVERSE)
    STASHAWAY_UNIVERSE.clear()
    STASHAWAY_UNIVERSE.update(
        {
            "VTI": _meta("VTI", "Equity", "US", "Broad Market"),
            "VEA": _meta("VEA", "Equity", "Developed ex-US", "Broad Market"),
            "VWO": _meta("VWO", "Equity", "Emerging Markets", "Broad Market"),
            "BND": _meta("BND", "Bond", "US", "Aggregate"),
            "BNDX": _meta("BNDX", "Bond", "Developed ex-US", "Aggregate"),
            "GLD": _meta("GLD", "Commodity", "Global", "Gold"),
        }
    )
    yield dict(STASHAWAY_UNIVERSE)
    STASHAWAY_UNIVERSE.clear()
    STASHAWAY_UNIVERSE.update(backup)


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    """Wide DataFrame of synthetic daily returns (3 years × 6 tickers)."""
    rng = np.random.default_rng(123)
    n = 252 * 3
    idx = pd.bdate_range("2021-01-04", periods=n, name="date")
    tickers = ["VTI", "VEA", "VWO", "BND", "BNDX", "GLD"]
    means = {
        "VTI": 0.0005, "VEA": 0.0003, "VWO": 0.0004,
        "BND": 0.00005, "BNDX": 0.00003, "GLD": 0.00015,
    }
    vols = {
        "VTI": 0.011, "VEA": 0.012, "VWO": 0.014,
        "BND": 0.003, "BNDX": 0.0035, "GLD": 0.009,
    }
    data = {}
    for t in tickers:
        data[t] = rng.normal(means[t], vols[t], size=n)
    return pd.DataFrame(data, index=idx)


def _portfolio(
    name: str,
    weights: dict[str, float],
    *,
    roles: set[Role],
    total_value: float = 100_000.0,
    statement_date: date = date(2024, 9, 30),
) -> Portfolio:
    holdings = [
        Holding(ticker=t, weight=w, value=w * total_value, metadata=STASHAWAY_UNIVERSE[t])
        for t, w in weights.items()
    ]
    return Portfolio(
        name=name,
        statement_date=statement_date,
        total_value=total_value,
        currency="USD",
        holdings=holdings,
        roles=roles,
    )


@pytest.fixture
def synthetic_book(seeded_universe: object) -> list[Portfolio]:
    """A small, role-tagged book for diagnostic tests."""
    return [
        _portfolio(
            "Custom Growth",
            {"VTI": 0.7, "VEA": 0.2, "VWO": 0.1},
            roles={Role.CUSTOM, Role.HOLDING},
            total_value=200_000,
        ),
        _portfolio(
            "Custom Defensive",
            {"BND": 0.6, "BNDX": 0.3, "GLD": 0.1},
            roles={Role.CUSTOM, Role.HOLDING, Role.PROTECTED},
            total_value=150_000,
        ),
        _portfolio(
            "General Investing",
            {"VTI": 0.5, "VEA": 0.2, "BND": 0.3},
            roles={Role.MANAGED_BENCHMARK, Role.HOLDING},
            total_value=300_000,
        ),
    ]


@pytest.fixture
def make_portfolio(seeded_universe: object) -> Callable[..., Portfolio]:
    """Factory fixture for ad-hoc portfolios in individual tests."""
    return _portfolio
