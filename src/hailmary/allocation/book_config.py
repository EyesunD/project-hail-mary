"""User-specific configuration for the Stashaway book.

Centralises the role mapping so notebooks don't duplicate it. Edit this file
when adding a new portfolio or re-tagging an existing one.
"""

from __future__ import annotations

from hailmary.allocation.portfolios import Role

ROLES: dict[str, set[Role]] = {
    # Stashaway Simple cash family — visible to diagnostic as replacement candidates.
    # Underlying yield modelled via synthetic CASH_USD / CASH_SGD return series
    # (see returns.py _CASH_ANNUAL_YIELDS). Simple SGD + Guitsa share the same
    # composition; kept separate for statement audit (Guitsa = mum's label).
    "Simple USD": {Role.HOLDING, Role.PROTECTED},
    "Simple SGD": {Role.HOLDING, Role.PROTECTED},
    "Guitsa": {Role.HOLDING, Role.PROTECTED},
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


MGMT_FEES_ANNUAL: dict[str, float] = {
    # Stashaway-charged management fees on top of underlying. Subtracted daily from
    # reconstructed returns in `returns.py:portfolio_returns`. Used when the
    # underlying ticker series is gross of Stashaway's fee (e.g. BIL for Simple
    # USD). Skip when the synthetic CASH_* series is already net of fees (e.g.
    # Simple SGD/Guitsa modelled at the 1.5% net rate from stashaway.sg/simple-var3).
    # Back-computed 2026-05 from app transactions: monthly fee charge / AUM × 12.
    # Stashaway page advertised 0.30% for Simple but the USD variant actually
    # bills at 0.44% (Apr fee 99.75 USD on 274,198 AUM).
    "Simple USD": 0.0044,
    "Singapore Investing": 0.0032,    # 5.36 SGD on 19,979 SGD AUM
    "General Investing": 0.0058,      # 180.35 USD on 371,548 USD AUM
    "Crypto": 0.0041,                 # 18.80 USD on 54,803 USD AUM
    "Income Investing": 0.0048,       # 1.99 SGD on 5,016 SGD AUM
    "General SRS": 0.0059,            # 50.19 USD on 101,300 USD AUM
    # BlackRock funded 30 Apr → April fee covered just 1 calendar day.
    # 0.63 × 365 / 39,180 = 0.587% p.a. — matches the other managed-sleeve tier.
    "BlackRock": 0.0059,
}
"""Per-portfolio Stashaway management fee (annualised, decimal).

Used for portfolios whose return series is built from gross-of-fee market data.
Add entries when extending coverage to other managed Stashaway portfolios
(General Investing, Singapore Investing, etc. — each carries 0.2–0.8% depending
on tier and AUM).
"""
