"""User-specific configuration for the Stashaway book.

Centralises the role mapping so notebooks don't duplicate it. Edit this file
when adding a new portfolio or re-tagging an existing one.
"""

from __future__ import annotations

from hailmary.allocation.portfolios import Role

ROLES: dict[str, set[Role]] = {
    # Cash-yield buckets — capital is movable, not strategic. Just HOLDING.
    "Simple USD": {Role.HOLDING},
    "Simple SGD": {Role.HOLDING},
    # PROTECTED — capital is structurally untouchable (tax-locked or held for
    # someone else). The redundancy detector won't propose dissolving these.
    "Guitsa": {Role.HOLDING, Role.PROTECTED},  # mum's account
    "General SRS": {Role.HOLDING, Role.PROTECTED},  # tax-locked CPF-SRS
    # Stashaway-managed sleeves — auto-rebalanced by Stashaway. Used as the
    # comparison target for CUSTOM sleeves. Capital is movable; can be
    # restructured if shown redundant.
    "BlackRock": {Role.HOLDING, Role.MANAGED_BENCHMARK},
    "General Investing": {Role.HOLDING, Role.MANAGED_BENCHMARK},
    "Singapore Investing": {Role.HOLDING, Role.MANAGED_BENCHMARK},
    "Income Investing": {Role.HOLDING, Role.MANAGED_BENCHMARK},
    # CUSTOM — user picks the weights. CUSTOM is a label only (no code branches
    # on it today); kept for mental classification.
    "Energy": {Role.HOLDING, Role.CUSTOM},
    "Utilities": {Role.HOLDING, Role.CUSTOM},
    "High Dividend Yield": {Role.HOLDING, Role.CUSTOM},
    "Ex-US Large-cap": {Role.HOLDING, Role.CUSTOM},
    "SG ETF": {Role.HOLDING, Role.CUSTOM},
    "Nasdaq Covered Call": {Role.HOLDING, Role.CUSTOM},
    "Crypto": {Role.HOLDING, Role.CUSTOM},
    "Asia ex-Japan": {Role.HOLDING, Role.CUSTOM},
    "Global Floating Rate USD": {Role.HOLDING, Role.CUSTOM},
    "Investment-Grade SGD": {Role.HOLDING, Role.CUSTOM},
    "Japan Currency-hedged": {Role.HOLDING, Role.CUSTOM},
    "Longevity Stack": {Role.HOLDING, Role.CUSTOM},
    "The AI Power Stack": {Role.HOLDING, Role.CUSTOM},
}
"""Map of Stashaway portfolio name → role set.

Multi-role portfolios are supported (a portfolio is a *set* of roles, not a
single label). See :class:`hailmary.allocation.portfolios.Role` for semantics.
"""


MGMT_FEES_ANNUAL: dict[str, float] = {
    # Stashaway management fee (annualised, decimal). Subtracted daily from
    # reconstructed returns in `returns.py:portfolio_returns` for portfolios
    # whose return series is built from gross-of-fee market data (the LSE/SGX
    # ETFs Stashaway holds quote price-only, not net-of-Stashaway-fee).
    #
    # Back-computed 2026-05 from app transactions: monthly_fee_charge × 12 / AUM.
    "Simple USD": 0.0044,             # 99.75 USD on 274,198 USD AUM (Stashaway page advertises 0.30%; USD variant bills 0.44%)
    "Simple SGD": 0.0015,             # 0.15% p.a. wrapper on top of LionGlobal SGD MMF + Enhanced Liquidity NAVs
    "Guitsa": 0.0015,                 # 0.15% p.a. (mum's account; same Simple-SGD composition)
    "Singapore Investing": 0.0032,    # 5.36 SGD on 19,979 SGD AUM
    "General Investing": 0.0058,      # 180.35 USD on 371,548 USD AUM
    "Crypto": 0.0041,                 # 18.80 USD on 54,803 USD AUM
    "Income Investing": 0.0048,       # 1.99 SGD on 5,016 SGD AUM
    "General SRS": 0.0059,            # 50.19 USD on 101,300 USD AUM
    "BlackRock": 0.0059,              # 0.63 × 365 / 39,180 = 0.587% (matches managed-sleeve tier; 1 calendar day in Apr)
    # Missing entries (intentional, not yet known):
    #   - 6 sleeves seeded in May 2026 (Asia ex-Japan, Global Floating Rate USD,
    #     Investment-Grade SGD, Japan Currency-hedged, Longevity Stack, AI Power
    #     Stack): too new to back-compute fee from a month of activity. Re-add
    #     after a full statement cycle.
    # Add entries when you have a clean fee transaction to back-compute from.
}
"""Per-portfolio Stashaway management fee (annualised, decimal)."""
