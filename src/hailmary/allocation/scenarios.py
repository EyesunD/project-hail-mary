"""Phase 2 scenario comparison: express a proposed book and diff it vs current.

User-driven (not optimizer-driven) what-if rebalancing on top of Phase 1
diagnostics. Helpers (drop_portfolio, set_weights, rebalance_into,
merge_into) return a new tuple of portfolios — they never mutate the input.
``scenario_compare`` runs every Phase 1 diagnostic on both books and computes
deltas; ``render_scenario_report`` (chunk 3) renders the result to HTML.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
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

    # Per-period scenario deltas — diff book_performance's windowed table
    # (1M / 3M / 6M / 1Y / All / per-calendar-year rows) between current and
    # proposed. Surfaces regime variation: an aggregate +0.2 Sharpe delta
    # might hide a +0.6 in one year and -0.4 in another. Columns include
    # cur_/prop_/delta_ variants of sharpe, ann_return, ann_vol, max_dd.
    by_period_deltas: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())


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


# ---------------------------------------------------------------------------
# scenario_compare engine
# ---------------------------------------------------------------------------


def _exposure_delta(
    cur: dict[str, pd.DataFrame], prop: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Per-dimension delta of exposure frames.

    For each dimension (asset_class / region / sector), join current and
    proposed on bucket, then compute weight_delta and value_delta. Returns
    one DataFrame per dimension keyed by bucket: columns
    ``bucket, weight_cur, weight_prop, weight_delta, value_cur, value_prop, value_delta``.
    """
    out: dict[str, pd.DataFrame] = {}
    for dim in set(cur) | set(prop):
        c = cur.get(dim, pd.DataFrame(columns=["bucket", "value", "weight"])).set_index("bucket")
        p = prop.get(dim, pd.DataFrame(columns=["bucket", "value", "weight"])).set_index("bucket")
        merged = c.add_suffix("_cur").join(p.add_suffix("_prop"), how="outer").fillna(0.0)
        merged["weight_delta"] = merged["weight_prop"] - merged["weight_cur"]
        merged["value_delta"] = merged["value_prop"] - merged["value_cur"]
        out[dim] = merged.reset_index().sort_values("weight_delta", key=abs, ascending=False)
    return out


def _by_period_delta(
    cur_windowed: pd.DataFrame | None, prop_windowed: pd.DataFrame | None
) -> pd.DataFrame:
    """Per-period diff of the windowed-metrics tables.

    Each row = one period (1M / 3M / 6M / 1Y / All / a calendar year),
    columns = ``period``, ``cur_<metric>``, ``prop_<metric>``, ``delta_<metric>``
    for each of sharpe / ann_return / ann_vol / max_dd. Periods present in
    only one side get NaNs on the missing side.
    """
    if cur_windowed is None or prop_windowed is None:
        return pd.DataFrame()
    if cur_windowed.empty and prop_windowed.empty:
        return pd.DataFrame()
    metrics = ["sharpe", "ann_return", "ann_vol", "max_dd"]
    c = cur_windowed.set_index("period")[metrics].add_prefix("cur_")
    p = prop_windowed.set_index("period")[metrics].add_prefix("prop_")
    merged = c.join(p, how="outer")
    for m in metrics:
        merged[f"delta_{m}"] = merged.get(f"prop_{m}") - merged.get(f"cur_{m}")
    # Preserve the canonical period order (1M, 3M, 6M, 1Y, All, then years).
    canonical = list(cur_windowed["period"]) + [
        p for p in prop_windowed["period"] if p not in list(cur_windowed["period"])
    ]
    merged = merged.reindex([p for p in canonical if p in merged.index])
    return merged.reset_index()


def _redundancy_delta(
    cur_pairs: list[tuple[str, str, float, str]],
    prop_pairs: list[tuple[str, str, float, str]],
) -> tuple[
    list[tuple[str, str, float]],
    list[tuple[str, str, float]],
    list[tuple[str, str, float, float]],
]:
    """Split into appeared (only in proposed), disappeared (only in current),
    persisted (both — current rho, proposed rho)."""

    def _key(p: tuple[str, str, float, str]) -> tuple[str, str]:
        return tuple(sorted([p[0], p[1]]))  # type: ignore[return-value]

    cur_map = {_key(p): p for p in cur_pairs}
    prop_map = {_key(p): p for p in prop_pairs}
    appeared = [
        (prop_map[k][0], prop_map[k][1], prop_map[k][2])
        for k in prop_map.keys() - cur_map.keys()
    ]
    disappeared = [
        (cur_map[k][0], cur_map[k][1], cur_map[k][2])
        for k in cur_map.keys() - prop_map.keys()
    ]
    persisted = [
        (cur_map[k][0], cur_map[k][1], cur_map[k][2], prop_map[k][2])
        for k in cur_map.keys() & prop_map.keys()
    ]
    return appeared, disappeared, persisted


def scenario_compare(
    current: Scenario,
    proposed: Scenario,
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    price_source: Any | None = None,
    returns: pd.DataFrame | None = None,
    fx_series_usd_sgd: pd.Series | None = None,
    fx_rate_usd_sgd: float | None = None,
    redundancy_threshold: float = 0.85,
    risk_free_rate: float = 0.0,
    align_window: bool = True,
    target_ann_return: float = 0.05,
) -> ScenarioDiff:
    """Run every Phase 1 diagnostic on both books and compute deltas.

    Returns a :class:`ScenarioDiff` containing the paired raw outputs and a
    fully-populated :class:`ScenarioDeltas`. Identical kwargs are passed to
    both runs so the comparison is apples-to-apples.

    Phase 1 diagnostics invoked:
    ``book_performance`` · ``combined_exposure`` · ``correlation_matrix``
    · ``redundancy_pairs`` · ``risk_contribution`` · ``benchmark_comparison``
    """
    # Imported inside the function to keep scenarios.py free of a top-level
    # dependency on diagnostic.py (avoids any future circular-import risk).
    from hailmary.allocation.diagnostic import (
        benchmark_comparison,
        book_performance,
        combined_exposure,
        correlation_matrix,
        redundancy_pairs,
        risk_contribution,
    )

    cur_books = list(current.portfolios)
    prop_books = list(proposed.portfolios)

    panel_kwargs: dict[str, Any] = {
        "start": start,
        "end": end,
        "price_source": price_source,
        "returns": returns,
        "fx_series_usd_sgd": fx_series_usd_sgd,
    }

    cur_book_perf = book_performance(
        cur_books,
        **panel_kwargs,
        fx_rate_usd_sgd=fx_rate_usd_sgd,
        risk_free_rate=risk_free_rate,
        align_window=align_window,
    )
    prop_book_perf = book_performance(
        prop_books,
        **panel_kwargs,
        fx_rate_usd_sgd=fx_rate_usd_sgd,
        risk_free_rate=risk_free_rate,
        align_window=align_window,
    )

    cur_exposure = combined_exposure(cur_books)
    prop_exposure = combined_exposure(prop_books)

    cur_corr = correlation_matrix(cur_books, **panel_kwargs)
    prop_corr = correlation_matrix(prop_books, **panel_kwargs)

    cur_pairs = redundancy_pairs(cur_corr, threshold=redundancy_threshold, portfolios=cur_books)
    prop_pairs = redundancy_pairs(prop_corr, threshold=redundancy_threshold, portfolios=prop_books)

    cur_risk = risk_contribution(
        cur_books,
        start=start,
        end=end,
        price_source=price_source,
        returns=returns,
        fx_rate_usd_sgd=fx_rate_usd_sgd,
    )
    prop_risk = risk_contribution(
        prop_books,
        start=start,
        end=end,
        price_source=price_source,
        returns=returns,
        fx_rate_usd_sgd=fx_rate_usd_sgd,
    )

    cur_bench = benchmark_comparison(
        cur_books,
        **panel_kwargs,
        fx_rate_usd_sgd=fx_rate_usd_sgd,
        risk_free_rate=risk_free_rate,
        align_window=align_window,
        target_ann_return=target_ann_return,
    )
    prop_bench = benchmark_comparison(
        prop_books,
        **panel_kwargs,
        fx_rate_usd_sgd=fx_rate_usd_sgd,
        risk_free_rate=risk_free_rate,
        align_window=align_window,
        target_ann_return=target_ann_return,
    )

    # ----- compute deltas -----
    deltas = ScenarioDeltas()
    deltas.book_sharpe_delta = float(prop_book_perf.get("sharpe", 0.0)) - float(
        cur_book_perf.get("sharpe", 0.0)
    )
    deltas.book_ann_return_delta = float(prop_book_perf.get("ann_return", 0.0)) - float(
        cur_book_perf.get("ann_return", 0.0)
    )
    deltas.book_ann_vol_delta = float(prop_book_perf.get("ann_vol", 0.0)) - float(
        cur_book_perf.get("ann_vol", 0.0)
    )
    deltas.book_max_dd_delta = float(prop_book_perf.get("max_dd", 0.0)) - float(
        cur_book_perf.get("max_dd", 0.0)
    )
    deltas.book_aum_delta = float(prop_book_perf.get("aum", 0.0)) - float(
        cur_book_perf.get("aum", 0.0)
    )

    # Per-portfolio deltas — only names present in both
    for name in set(cur_bench.index) & set(prop_bench.index):
        if name == "Combined book":
            continue
        deltas.per_portfolio_sharpe_delta[name] = float(prop_bench.loc[name, "sharpe"]) - float(
            cur_bench.loc[name, "sharpe"]
        )
        deltas.per_portfolio_vol_delta[name] = float(
            prop_bench.loc[name, "annualised_vol"]
        ) - float(cur_bench.loc[name, "annualised_vol"])
        deltas.per_portfolio_max_dd_delta[name] = float(
            prop_bench.loc[name, "max_dd"]
        ) - float(cur_bench.loc[name, "max_dd"])

    deltas.exposure_delta = _exposure_delta(cur_exposure, prop_exposure)
    deltas.redundancy_appeared, deltas.redundancy_disappeared, deltas.redundancy_persisted = (
        _redundancy_delta(cur_pairs, prop_pairs)
    )
    deltas.by_period_deltas = _by_period_delta(
        cur_book_perf.get("windowed"), prop_book_perf.get("windowed")
    )

    return ScenarioDiff(
        current=current,
        proposed=proposed,
        deltas=deltas,
        book_performance=(cur_book_perf, prop_book_perf),
        exposure=(cur_exposure, prop_exposure),
        correlation=(cur_corr, prop_corr),
        redundancy=(cur_pairs, prop_pairs),
        risk=(cur_risk, prop_risk),
        benchmarks=(cur_bench, prop_bench),
    )
