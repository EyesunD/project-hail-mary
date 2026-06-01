"""Portfolio data model with role tagging.

A :class:`Portfolio` represents one Stashaway portfolio (custom or managed)
after parsing and universe resolution. Roles are a *set*, not a hierarchy —
a managed Stashaway portfolio is typically both ``HOLDING`` and
``MANAGED_BENCHMARK``. Diagnostic functions filter by role membership.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

from hailmary.allocation.statements import ParsedPortfolio
from hailmary.allocation.universe import AssetMetadata, resolve

WEIGHT_TOLERANCE = 1e-4


class Role(StrEnum):
    """Tagged roles a portfolio can play in the diagnostic.

    Multiple roles are allowed; see ``D3`` in design.md.
    """

    CUSTOM = "custom"
    MANAGED_BENCHMARK = "managed_benchmark"
    PROTECTED = "protected"
    HOLDING = "holding"


@dataclass(frozen=True, slots=True)
class Holding:
    """One resolved holding inside a :class:`Portfolio`."""

    ticker: str
    weight: float
    value: float
    metadata: AssetMetadata


@dataclass(slots=True)
class Portfolio:
    """A Stashaway portfolio with current holdings and diagnostic role tags."""

    name: str
    statement_date: date
    total_value: float
    currency: str
    holdings: list[Holding]
    roles: set[Role] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.holdings:
            return
        s = sum(h.weight for h in self.holdings)
        if abs(s - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(
                f"Portfolio {self.name!r} weights sum to {s:.6f}, expected 1.0 ± {WEIGHT_TOLERANCE}"
            )

    @property
    def tickers(self) -> list[str]:
        return [h.metadata.ticker for h in self.holdings]

    def has_role(self, role: Role) -> bool:
        return role in self.roles


def from_parsed(
    parsed: ParsedPortfolio,
    roles: set[Role],
    *,
    metadata: dict[str, Any] | None = None,
) -> Portfolio:
    """Build a :class:`Portfolio` from a :class:`ParsedPortfolio`.

    Each holding is resolved against the universe map; an unknown identifier
    raises :class:`hailmary.allocation.universe.UnknownAssetError` so the gap
    surfaces immediately rather than silently dropping the row. Optional
    ``metadata`` is attached to ``Portfolio.metadata`` — commonly used to
    carry a per-portfolio ``management_fee_annual`` for return-series
    deduction (see ``returns.py``).
    """
    holdings = [
        Holding(
            ticker=h.ticker,
            weight=h.weight,
            value=h.value,
            metadata=resolve(h.ticker, source=parsed.name),
        )
        for h in parsed.holdings
    ]
    md: dict[str, Any] = dict(metadata) if metadata else {}
    if parsed.statement_fx_usd_sgd is not None:
        md.setdefault("statement_fx_usd_sgd", parsed.statement_fx_usd_sgd)
    return Portfolio(
        name=parsed.name,
        statement_date=parsed.statement_date,
        total_value=parsed.total_value,
        currency=parsed.currency,
        holdings=holdings,
        roles=set(roles),
        metadata=md,
    )
