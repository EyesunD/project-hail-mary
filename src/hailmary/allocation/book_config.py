"""User-specific configuration for the Stashaway book.

Centralises the role mapping so notebooks don't duplicate it. Edit this file
when adding a new portfolio or re-tagging an existing one.
"""

from __future__ import annotations

from hailmary.allocation.portfolios import Role

ROLES: dict[str, set[Role]] = {
    # Cash management — invisible to diagnostic (no HOLDING)
    "Simple USD": {Role.PROTECTED},
    "Simple SGD": {Role.PROTECTED},
    "Guitsa": {Role.PROTECTED},
    # Tax-locked
    "General SRS": {Role.HOLDING, Role.PROTECTED},
    # Managed-benchmark sleeves — comparison targets
    "BlackRock": {Role.HOLDING, Role.MANAGED_BENCHMARK},
    "General Investing": {Role.HOLDING, Role.MANAGED_BENCHMARK},
    "Singapore Investing": {Role.HOLDING, Role.MANAGED_BENCHMARK},
    "Income Investing": {Role.HOLDING, Role.MANAGED_BENCHMARK},
    # Custom thematic bets — measured against the benchmarks above
    "Energy": {Role.HOLDING, Role.CUSTOM},
    "Utilities": {Role.HOLDING, Role.CUSTOM},
    "High Dividend Yield": {Role.HOLDING, Role.CUSTOM},
    "Ex-US Large-cap": {Role.HOLDING, Role.CUSTOM},
    "SG ETF": {Role.HOLDING, Role.CUSTOM},
    "Nasdaq Covered Call": {Role.HOLDING, Role.CUSTOM},
    "Crypto": {Role.HOLDING, Role.CUSTOM},
}
"""Map of Stashaway portfolio name → role set.

Multi-role portfolios are supported (a portfolio is a *set* of roles, not a
single label). See :class:`hailmary.allocation.portfolios.Role` for semantics.
"""
