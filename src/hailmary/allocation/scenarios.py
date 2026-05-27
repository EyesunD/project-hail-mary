"""Phase 2 scenario comparison: express a proposed book and diff it vs current.

User-driven (not optimizer-driven) what-if rebalancing on top of Phase 1
diagnostics. Helpers (drop_portfolio, set_weights, rebalance_into,
merge_into) return a new tuple of portfolios — they never mutate the input.
``scenario_compare`` (chunk 2) runs every Phase 1 diagnostic on both books
and computes deltas.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from hailmary.allocation.portfolios import (
    WEIGHT_TOLERANCE,
    Holding,
    Portfolio,
    Role,
)


class ScenarioEditError(Exception):
    """Raised when an edit helper receives invalid input."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named book of portfolios for use in scenario comparisons.

    Immutable wrapper around a tuple of :class:`Portfolio` objects. Use the
    edit helpers (``drop_portfolio``, ``set_weights``, ``rebalance_into``,
    ``merge_into``) to construct a 'proposed' Scenario from a 'current' one.
    """

    label: str
    portfolios: tuple[Portfolio, ...]
    note: str | None = None

    def __post_init__(self) -> None:
        # Coerce list inputs to tuple so equality and hashing behave
        if not isinstance(self.portfolios, tuple):
            object.__setattr__(self, "portfolios", tuple(self.portfolios))


@dataclass
class ScenarioDeltas:
    """Computed deltas (proposed − current) across every Phase 1 diagnostic.

    All fields default to empty/zero so the dataclass can be constructed
    incrementally inside :func:`scenario_compare` (chunk 2). Carries:

    - book-level shifts (Sharpe, ann return, vol, max DD, AUM)
    - per-portfolio shifts (Sharpe / vol / max-DD per portfolio name)
    - exposure-bucket shifts per dimension
    - redundancy-pair appearances / disappearances / persistence
    """

    # Whole-book deltas
    book_sharpe_delta: float = 0.0
    book_ann_return_delta: float = 0.0
    book_ann_vol_delta: float = 0.0
    book_max_dd_delta: float = 0.0
    book_aum_delta: float = 0.0

    # Per-portfolio deltas (portfolio name → value)
    per_portfolio_sharpe_delta: dict[str, float] = field(default_factory=dict)
    per_portfolio_vol_delta: dict[str, float] = field(default_factory=dict)
    per_portfolio_max_dd_delta: dict[str, float] = field(default_factory=dict)

    # Exposure shift per dimension (asset_class / region / sector)
    exposure_delta: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Redundancy-pair lifecycle. Each tuple is (name_a, name_b, ...)
    # appeared / disappeared carry the single rho they had at that side.
    # persisted carries (name_a, name_b, rho_current, rho_proposed).
    redundancy_appeared: list[tuple[str, str, float]] = field(default_factory=list)
    redundancy_disappeared: list[tuple[str, str, float]] = field(default_factory=list)
    redundancy_persisted: list[tuple[str, str, float, float]] = field(default_factory=list)


@dataclass
class ScenarioDiff:
    """Result of ``scenario_compare(current, proposed)``.

    Carries both scenarios, paired raw outputs (current, proposed) from every
    Phase 1 diagnostic, and the computed :class:`ScenarioDeltas`. Paired
    outputs are populated in chunk 2; for chunk 1 they default to ``None``.
    """

    current: Scenario
    proposed: Scenario
    deltas: ScenarioDeltas = field(default_factory=ScenarioDeltas)
    book_performance: tuple[dict[str, Any], dict[str, Any]] | None = None
    exposure: tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]] | None = None
    correlation: tuple[pd.DataFrame, pd.DataFrame] | None = None
    redundancy: tuple[list[Any], list[Any]] | None = None
    risk: tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]] | None = None
    benchmarks: tuple[pd.DataFrame, pd.DataFrame] | None = None


# ---------------------------------------------------------------------------
# Edit helpers — all return tuple[Portfolio, ...], all non-mutating
# ---------------------------------------------------------------------------


def _portfolio_index(portfolios: Sequence[Portfolio]) -> dict[str, Portfolio]:
    """Index portfolios by name, raising on duplicate names."""
    out: dict[str, Portfolio] = {}
    for p in portfolios:
        if p.name in out:
            raise ScenarioEditError(f"Duplicate portfolio name in input: {p.name!r}")
        out[p.name] = p
    return out


def _clone_portfolio(
    p: Portfolio,
    *,
    name: str | None = None,
    total_value: float | None = None,
    holdings: list[Holding] | None = None,
    roles: set[Role] | None = None,
) -> Portfolio:
    """Build a new Portfolio from an existing one, optionally overriding fields.

    Always deep-copies the mutable containers (set of roles, dict of metadata,
    list of holdings) so the caller can't unintentionally share state with the
    source portfolio.
    """
    return Portfolio(
        name=name if name is not None else p.name,
        statement_date=p.statement_date,
        total_value=total_value if total_value is not None else p.total_value,
        currency=p.currency,
        holdings=list(holdings) if holdings is not None else list(p.holdings),
        roles=set(roles) if roles is not None else set(p.roles),
        metadata=dict(p.metadata),
    )


def drop_portfolio(
    portfolios: Sequence[Portfolio], name: str
) -> tuple[Portfolio, ...]:
    """Return a new tuple with the named portfolio removed.

    Raises :class:`ScenarioEditError` if no portfolio matches ``name``.
    """
    available = [p.name for p in portfolios]
    if name not in available:
        raise ScenarioEditError(
            f"Portfolio {name!r} not found. Available: {sorted(available)}"
        )
    return tuple(p for p in portfolios if p.name != name)


def set_weights(
    portfolios: Sequence[Portfolio],
    portfolio_name: str,
    weights: dict[str, float],
) -> tuple[Portfolio, ...]:
    """Replace one portfolio's holding weights with the supplied mapping.

    Keys in ``weights`` are matched against existing holdings'
    ``stashaway_id``. The values must sum to ``1.0 ± WEIGHT_TOLERANCE``.
    Each existing holding has its weight overwritten with the value from
    ``weights``; holdings whose ``stashaway_id`` doesn't appear in
    ``weights`` raise — the caller must explicitly assign 0.0 to zero out
    a holding (or omit the helper and construct a new ``Portfolio`` if
    they want to add new tickers).
    """
    index = _portfolio_index(portfolios)
    if portfolio_name not in index:
        raise ScenarioEditError(
            f"Portfolio {portfolio_name!r} not found. Available: {sorted(index)}"
        )
    total = sum(weights.values())
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ScenarioEditError(
            f"Weights for {portfolio_name!r} sum to {total:.6f}, "
            f"expected 1.0 ± {WEIGHT_TOLERANCE}"
        )
    target = index[portfolio_name]
    existing_ids = {h.stashaway_id for h in target.holdings}
    unknown = set(weights) - existing_ids
    if unknown:
        raise ScenarioEditError(
            f"Unknown stashaway_id(s) in weights for {portfolio_name!r}: "
            f"{sorted(unknown)}. Existing: {sorted(existing_ids)}"
        )
    missing = existing_ids - set(weights)
    if missing:
        raise ScenarioEditError(
            f"set_weights requires explicit weights for every existing holding. "
            f"Missing for {portfolio_name!r}: {sorted(missing)}. "
            f"Pass 0.0 to zero them out."
        )

    new_holdings = [
        Holding(
            stashaway_id=h.stashaway_id,
            weight=weights[h.stashaway_id],
            value=weights[h.stashaway_id] * target.total_value,
            metadata=h.metadata,
        )
        for h in target.holdings
    ]
    new_target = _clone_portfolio(target, holdings=new_holdings)
    return tuple(new_target if p.name == portfolio_name else p for p in portfolios)


def rebalance_into(
    portfolios: Sequence[Portfolio], from_name: str, to_name: str
) -> tuple[Portfolio, ...]:
    """Move all of ``from_name``'s ``total_value`` into ``to_name``; drop ``from_name``.

    ``to_name``'s holdings stay the same — only its ``total_value`` grows.
    Models *"sell everything in `from_name` and use the cash to buy more of
    `to_name` at its current composition"*. Currency mismatch raises.
    """
    index = _portfolio_index(portfolios)
    if from_name not in index:
        raise ScenarioEditError(f"Portfolio {from_name!r} not found.")
    if to_name not in index:
        raise ScenarioEditError(f"Portfolio {to_name!r} not found.")
    if from_name == to_name:
        raise ScenarioEditError(
            f"from_name and to_name must differ ({from_name!r})."
        )
    src = index[from_name]
    dst = index[to_name]
    if src.currency.upper() != dst.currency.upper():
        raise ScenarioEditError(
            f"Currency mismatch: {from_name!r} is {src.currency}, "
            f"{to_name!r} is {dst.currency}. Convert FX first."
        )

    new_value = dst.total_value + src.total_value
    new_holdings = [
        Holding(
            stashaway_id=h.stashaway_id,
            weight=h.weight,
            value=h.weight * new_value,
            metadata=h.metadata,
        )
        for h in dst.holdings
    ]
    new_dst = _clone_portfolio(dst, total_value=new_value, holdings=new_holdings)
    return tuple(
        new_dst if p.name == to_name else p
        for p in portfolios
        if p.name != from_name
    )


def merge_into(
    portfolios: Sequence[Portfolio],
    names: Sequence[str],
    into: str,
    *,
    roles: set[Role] | None = None,
) -> tuple[Portfolio, ...]:
    """Combine ``names`` into one value-weighted portfolio named ``into``.

    New ``total_value`` = sum of sources. New holdings = value-weighted union
    of sources' holdings (same ``stashaway_id`` held in multiple sources gets
    summed). Source portfolios are removed. ``into`` may be the name of one
    of the source portfolios (in which case that one is conceptually replaced
    with the merged result) or a new name. All sources must share a currency.

    The merged portfolio's ``roles`` default to the first source's roles
    unless overridden via the ``roles`` keyword argument. ``statement_date``
    and ``metadata`` are inherited from the first source.
    """
    index = _portfolio_index(portfolios)
    sources: list[Portfolio] = []
    for n in names:
        if n not in index:
            raise ScenarioEditError(f"Portfolio {n!r} not found.")
        sources.append(index[n])
    if not sources:
        raise ScenarioEditError("merge_into requires at least one source name.")
    currencies = {p.currency.upper() for p in sources}
    if len(currencies) > 1:
        raise ScenarioEditError(
            f"Cannot merge portfolios with mixed currencies: {sorted(currencies)}. "
            "Convert FX first."
        )
    drop_names = set(names)
    if into not in drop_names and into in index:
        raise ScenarioEditError(
            f"into={into!r} matches an existing portfolio not in names. "
            "Either include it in `names` (to merge into itself) or pick a new name."
        )

    new_value = float(sum(p.total_value for p in sources))
    aggregated: dict[str, dict[str, Any]] = {}
    for p in sources:
        for h in p.holdings:
            entry = aggregated.setdefault(
                h.stashaway_id,
                {"value": 0.0, "metadata": h.metadata},
            )
            entry["value"] += h.weight * p.total_value
    new_holdings = [
        Holding(
            stashaway_id=sid,
            weight=(entry["value"] / new_value) if new_value > 0 else 0.0,
            value=entry["value"],
            metadata=entry["metadata"],
        )
        for sid, entry in aggregated.items()
    ]
    merged = _clone_portfolio(
        sources[0],
        name=into,
        total_value=new_value,
        holdings=new_holdings,
        roles=roles if roles is not None else set(sources[0].roles),
    )

    result: list[Portfolio] = []
    inserted = False
    for p in portfolios:
        if p.name in drop_names:
            if not inserted:
                result.append(merged)
                inserted = True
            continue
        result.append(p)
    if not inserted:  # defensive — shouldn't happen if names is non-empty
        result.append(merged)
    return tuple(result)
