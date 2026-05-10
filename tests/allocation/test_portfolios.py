"""Portfolio model + role tagging."""

from __future__ import annotations

from datetime import date

import pytest

from hailmary.allocation.portfolios import Holding, Portfolio, Role, from_parsed
from hailmary.allocation.statements import ParsedHolding, ParsedPortfolio
from hailmary.allocation.universe import STASHAWAY_UNIVERSE


def test_portfolio_rejects_invalid_weights(seeded_universe: object) -> None:
    holdings = [
        Holding("VTI", 0.5, 50_000, STASHAWAY_UNIVERSE["VTI"]),
        Holding("VEA", 0.3, 30_000, STASHAWAY_UNIVERSE["VEA"]),
    ]
    with pytest.raises(ValueError, match="weights sum to"):
        Portfolio(
            name="Bad",
            statement_date=date(2024, 9, 30),
            total_value=80_000,
            currency="USD",
            holdings=holdings,
            roles={Role.HOLDING},
        )


def test_portfolio_multi_role(seeded_universe: object) -> None:
    h = [Holding("VTI", 1.0, 100_000, STASHAWAY_UNIVERSE["VTI"])]
    p = Portfolio(
        name="General Investing",
        statement_date=date(2024, 9, 30),
        total_value=100_000,
        currency="USD",
        holdings=h,
        roles={Role.HOLDING, Role.MANAGED_BENCHMARK},
    )
    assert p.has_role(Role.HOLDING)
    assert p.has_role(Role.MANAGED_BENCHMARK)
    assert not p.has_role(Role.PROTECTED)


def test_protected_role_carries_through(seeded_universe: object) -> None:
    h = [Holding("BND", 1.0, 100_000, STASHAWAY_UNIVERSE["BND"])]
    p = Portfolio(
        name="Cash + Bonds",
        statement_date=date(2024, 9, 30),
        total_value=100_000,
        currency="USD",
        holdings=h,
        roles={Role.PROTECTED, Role.HOLDING},
    )
    assert Role.PROTECTED in p.roles
    assert Role.HOLDING in p.roles


def test_from_parsed_resolves_universe(seeded_universe: object) -> None:
    parsed = ParsedPortfolio(
        name="Custom Growth",
        statement_date=date(2024, 9, 30),
        currency="USD",
        total_value=100_000,
        holdings=[
            ParsedHolding("VTI", 0.7, 70_000),
            ParsedHolding("VEA", 0.3, 30_000),
        ],
    )
    p = from_parsed(parsed, roles={Role.CUSTOM, Role.HOLDING})
    assert len(p.holdings) == 2
    assert p.holdings[0].metadata.region == "US"
    assert p.tickers == ["VTI", "VEA"]
