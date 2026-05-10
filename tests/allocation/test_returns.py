"""Return reconstruction."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from hailmary.allocation.portfolios import Holding, Portfolio, Role
from hailmary.allocation.returns import InsufficientHistoryError, portfolio_returns
from hailmary.allocation.universe import STASHAWAY_UNIVERSE


def test_known_weights_known_returns(seeded_universe: object) -> None:
    idx = pd.bdate_range("2022-01-03", periods=5)
    returns = pd.DataFrame(
        {
            "VTI": [0.01, 0.02, -0.01, 0.005, 0.0],
            "BND": [0.001, 0.0, 0.002, -0.001, 0.0005],
        },
        index=idx,
    )
    p = Portfolio(
        name="60/40",
        statement_date=date(2024, 9, 30),
        total_value=100_000,
        currency="USD",
        holdings=[
            Holding("VTI", 0.6, 60_000, STASHAWAY_UNIVERSE["VTI"]),
            Holding("BND", 0.4, 40_000, STASHAWAY_UNIVERSE["BND"]),
        ],
        roles={Role.HOLDING},
    )
    s = portfolio_returns(p, returns=returns)
    expected = 0.6 * returns["VTI"] + 0.4 * returns["BND"]
    np.testing.assert_allclose(s.values, expected.values, rtol=1e-12)
    assert s.name == "60/40"


def test_truncates_when_holding_has_gaps(seeded_universe: object) -> None:
    idx = pd.bdate_range("2022-01-03", periods=5)
    returns = pd.DataFrame(
        {
            "VTI": [0.01, 0.02, -0.01, 0.005, 0.0],
            "BND": [np.nan, np.nan, 0.002, -0.001, 0.0005],
        },
        index=idx,
    )
    p = Portfolio(
        name="Truncated",
        statement_date=date(2024, 9, 30),
        total_value=100_000,
        currency="USD",
        holdings=[
            Holding("VTI", 0.6, 60_000, STASHAWAY_UNIVERSE["VTI"]),
            Holding("BND", 0.4, 40_000, STASHAWAY_UNIVERSE["BND"]),
        ],
        roles={Role.HOLDING},
    )
    with pytest.warns(UserWarning, match="common-history window"):
        s = portfolio_returns(p, returns=returns)
    assert len(s) == 3  # only the days where BND has data


def test_strict_mode_raises_on_gaps(seeded_universe: object) -> None:
    idx = pd.bdate_range("2022-01-03", periods=5)
    returns = pd.DataFrame(
        {
            "VTI": [0.01, 0.02, -0.01, 0.005, 0.0],
            "BND": [np.nan, np.nan, 0.002, -0.001, 0.0005],
        },
        index=idx,
    )
    p = Portfolio(
        name="Strict",
        statement_date=date(2024, 9, 30),
        total_value=100_000,
        currency="USD",
        holdings=[
            Holding("VTI", 0.6, 60_000, STASHAWAY_UNIVERSE["VTI"]),
            Holding("BND", 0.4, 40_000, STASHAWAY_UNIVERSE["BND"]),
        ],
        roles={Role.HOLDING},
    )
    with pytest.raises(InsufficientHistoryError):
        portfolio_returns(p, returns=returns, strict=True)


def test_missing_ticker_raises(seeded_universe: object) -> None:
    idx = pd.bdate_range("2022-01-03", periods=5)
    returns = pd.DataFrame({"VTI": [0.01] * 5}, index=idx)
    p = Portfolio(
        name="Missing",
        statement_date=date(2024, 9, 30),
        total_value=100_000,
        currency="USD",
        holdings=[
            Holding("VTI", 0.6, 60_000, STASHAWAY_UNIVERSE["VTI"]),
            Holding("BND", 0.4, 40_000, STASHAWAY_UNIVERSE["BND"]),
        ],
        roles={Role.HOLDING},
    )
    with pytest.raises(InsufficientHistoryError):
        portfolio_returns(p, returns=returns)
