"""Phase 2 scenario edit helpers + data model."""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from hailmary.allocation.portfolios import Holding, Portfolio, Role
from hailmary.allocation.scenarios import (
    Scenario,
    ScenarioDeltas,
    ScenarioDiff,
    ScenarioEditError,
    drop_portfolio,
    merge_into,
    rebalance_into,
    set_weights,
)
from hailmary.allocation.universe import STASHAWAY_UNIVERSE


def _portfolio(
    name: str,
    weights: dict[str, float],
    *,
    total_value: float = 100_000.0,
    currency: str = "USD",
    roles: set[Role] | None = None,
) -> Portfolio:
    holdings = [
        Holding(
            stashaway_id=sid,
            weight=w,
            value=w * total_value,
            metadata=STASHAWAY_UNIVERSE[sid],
        )
        for sid, w in weights.items()
    ]
    return Portfolio(
        name=name,
        statement_date=date(2026, 4, 30),
        total_value=total_value,
        currency=currency,
        holdings=holdings,
        roles=roles or {Role.HOLDING, Role.CUSTOM},
    )


# ---------------------------------------------------------------------------
# Scenario / ScenarioDeltas / ScenarioDiff data types
# ---------------------------------------------------------------------------


def test_scenario_is_frozen(seeded_universe: object) -> None:
    p = _portfolio("A", {"VTI": 1.0})
    s = Scenario(label="current", portfolios=(p,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.label = "modified"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.portfolios = (p, p)  # type: ignore[misc]


def test_scenario_coerces_list_to_tuple(seeded_universe: object) -> None:
    p = _portfolio("A", {"VTI": 1.0})
    s = Scenario(label="L", portfolios=[p])  # type: ignore[arg-type]
    assert isinstance(s.portfolios, tuple)
    assert s.portfolios == (p,)


def test_scenario_deltas_defaults() -> None:
    d = ScenarioDeltas()
    assert d.book_sharpe_delta == 0.0
    assert d.per_portfolio_sharpe_delta == {}
    assert d.redundancy_appeared == []


def test_scenario_diff_construction(seeded_universe: object) -> None:
    p = _portfolio("A", {"VTI": 1.0})
    cur = Scenario(label="cur", portfolios=(p,))
    prop = Scenario(label="prop", portfolios=(p,))
    diff = ScenarioDiff(current=cur, proposed=prop)
    assert diff.current is cur
    assert diff.proposed is prop
    assert isinstance(diff.deltas, ScenarioDeltas)
    assert diff.book_performance is None


# ---------------------------------------------------------------------------
# drop_portfolio
# ---------------------------------------------------------------------------


def test_drop_portfolio_happy_path(seeded_universe: object) -> None:
    book = [
        _portfolio("A", {"VTI": 1.0}),
        _portfolio("B", {"BND": 1.0}),
        _portfolio("C", {"GLD": 1.0}),
    ]
    out = drop_portfolio(book, "B")
    assert [p.name for p in out] == ["A", "C"]
    # Input untouched
    assert [p.name for p in book] == ["A", "B", "C"]


def test_drop_portfolio_unknown_raises(seeded_universe: object) -> None:
    book = [_portfolio("A", {"VTI": 1.0})]
    with pytest.raises(ScenarioEditError, match="not found"):
        drop_portfolio(book, "Z")


# ---------------------------------------------------------------------------
# set_weights
# ---------------------------------------------------------------------------


def test_set_weights_happy_path(seeded_universe: object) -> None:
    book = [_portfolio("A", {"VTI": 0.6, "BND": 0.4})]
    out = set_weights(book, "A", {"VTI": 0.3, "BND": 0.7})
    new_a = out[0]
    weights = {h.stashaway_id: h.weight for h in new_a.holdings}
    assert weights == {"VTI": 0.3, "BND": 0.7}
    # Values rebuilt
    values = {h.stashaway_id: h.value for h in new_a.holdings}
    assert values == {"VTI": 30_000.0, "BND": 70_000.0}
    # Input untouched
    orig_weights = {h.stashaway_id: h.weight for h in book[0].holdings}
    assert orig_weights == {"VTI": 0.6, "BND": 0.4}


def test_set_weights_zero_out_holding(seeded_universe: object) -> None:
    book = [_portfolio("A", {"VTI": 0.5, "BND": 0.5})]
    out = set_weights(book, "A", {"VTI": 1.0, "BND": 0.0})
    weights = {h.stashaway_id: h.weight for h in out[0].holdings}
    assert weights == {"VTI": 1.0, "BND": 0.0}


def test_set_weights_unknown_portfolio_raises(seeded_universe: object) -> None:
    book = [_portfolio("A", {"VTI": 1.0})]
    with pytest.raises(ScenarioEditError, match="not found"):
        set_weights(book, "Z", {"VTI": 1.0})


def test_set_weights_unknown_ticker_raises(seeded_universe: object) -> None:
    book = [_portfolio("A", {"VTI": 0.6, "BND": 0.4})]
    with pytest.raises(ScenarioEditError, match="Unknown stashaway_id"):
        set_weights(book, "A", {"VTI": 0.5, "GLD": 0.5})


def test_set_weights_missing_existing_raises(seeded_universe: object) -> None:
    book = [_portfolio("A", {"VTI": 0.6, "BND": 0.4})]
    with pytest.raises(ScenarioEditError, match="requires explicit weights"):
        set_weights(book, "A", {"VTI": 1.0})


def test_set_weights_bad_sum_raises(seeded_universe: object) -> None:
    book = [_portfolio("A", {"VTI": 0.6, "BND": 0.4})]
    with pytest.raises(ScenarioEditError, match="sum to"):
        set_weights(book, "A", {"VTI": 0.7, "BND": 0.4})


# ---------------------------------------------------------------------------
# rebalance_into
# ---------------------------------------------------------------------------


def test_rebalance_into_happy_path(seeded_universe: object) -> None:
    book = [
        _portfolio("Crypto", {"VTI": 1.0}, total_value=50_000),
        _portfolio("Equity", {"BND": 1.0}, total_value=200_000),
    ]
    out = rebalance_into(book, "Crypto", "Equity")
    assert [p.name for p in out] == ["Equity"]
    eq = out[0]
    assert eq.total_value == 250_000
    # Holdings unchanged structurally — just larger value
    assert {h.stashaway_id for h in eq.holdings} == {"BND"}
    assert eq.holdings[0].value == 250_000


def test_rebalance_into_currency_mismatch_raises(seeded_universe: object) -> None:
    book = [
        _portfolio("USD-port", {"VTI": 1.0}, currency="USD"),
        _portfolio("SGD-port", {"BND": 1.0}, currency="SGD"),
    ]
    with pytest.raises(ScenarioEditError, match="Currency mismatch"):
        rebalance_into(book, "USD-port", "SGD-port")


def test_rebalance_into_self_raises(seeded_universe: object) -> None:
    book = [_portfolio("A", {"VTI": 1.0})]
    with pytest.raises(ScenarioEditError, match="must differ"):
        rebalance_into(book, "A", "A")


def test_rebalance_into_unknown_raises(seeded_universe: object) -> None:
    book = [_portfolio("A", {"VTI": 1.0})]
    with pytest.raises(ScenarioEditError, match="not found"):
        rebalance_into(book, "A", "Z")


# ---------------------------------------------------------------------------
# merge_into
# ---------------------------------------------------------------------------


def test_merge_into_happy_path_new_name(seeded_universe: object) -> None:
    book = [
        _portfolio("Energy", {"VTI": 1.0}, total_value=10_000),
        _portfolio("Utilities", {"BND": 1.0}, total_value=20_000),
        _portfolio("Other", {"GLD": 1.0}, total_value=5_000),
    ]
    out = merge_into(book, ["Energy", "Utilities"], "Custom Sleeve")
    names = [p.name for p in out]
    assert "Energy" not in names and "Utilities" not in names
    assert "Custom Sleeve" in names
    merged = next(p for p in out if p.name == "Custom Sleeve")
    assert merged.total_value == 30_000
    # Value-weighted union: Energy contributed 10K VTI, Utilities 20K BND
    weights = {h.stashaway_id: h.weight for h in merged.holdings}
    assert weights["VTI"] == pytest.approx(10_000 / 30_000)
    assert weights["BND"] == pytest.approx(20_000 / 30_000)


def test_merge_into_happy_path_replace_source(seeded_universe: object) -> None:
    book = [
        _portfolio("Energy", {"VTI": 1.0}, total_value=10_000),
        _portfolio("Utilities", {"BND": 1.0}, total_value=20_000),
    ]
    out = merge_into(book, ["Energy", "Utilities"], "Energy")
    assert [p.name for p in out] == ["Energy"]
    assert out[0].total_value == 30_000


def test_merge_into_same_ticker_across_sources_sums(seeded_universe: object) -> None:
    book = [
        _portfolio("A", {"VTI": 1.0}, total_value=10_000),
        _portfolio("B", {"VTI": 1.0}, total_value=20_000),
    ]
    out = merge_into(book, ["A", "B"], "AB")
    merged = next(p for p in out if p.name == "AB")
    assert len(merged.holdings) == 1
    assert merged.holdings[0].weight == pytest.approx(1.0)


def test_merge_into_currency_mismatch_raises(seeded_universe: object) -> None:
    book = [
        _portfolio("A", {"VTI": 1.0}, currency="USD"),
        _portfolio("B", {"BND": 1.0}, currency="SGD"),
    ]
    with pytest.raises(ScenarioEditError, match="mixed currencies"):
        merge_into(book, ["A", "B"], "AB")


def test_merge_into_into_name_clashes_with_non_source(seeded_universe: object) -> None:
    book = [
        _portfolio("Energy", {"VTI": 1.0}),
        _portfolio("Utilities", {"BND": 1.0}),
        _portfolio("Crypto", {"GLD": 1.0}),
    ]
    with pytest.raises(ScenarioEditError, match="matches an existing portfolio"):
        merge_into(book, ["Energy", "Utilities"], "Crypto")


def test_merge_into_unknown_source_raises(seeded_universe: object) -> None:
    book = [_portfolio("A", {"VTI": 1.0})]
    with pytest.raises(ScenarioEditError, match="not found"):
        merge_into(book, ["A", "Z"], "AB")


def test_merge_into_empty_names_raises(seeded_universe: object) -> None:
    book = [_portfolio("A", {"VTI": 1.0})]
    with pytest.raises(ScenarioEditError, match="at least one"):
        merge_into(book, [], "AB")


# ---------------------------------------------------------------------------
# Non-mutation invariant — applies to every helper
# ---------------------------------------------------------------------------


def test_helpers_do_not_mutate_input(seeded_universe: object) -> None:
    book = [
        _portfolio("A", {"VTI": 0.6, "BND": 0.4}, total_value=100_000),
        _portfolio("B", {"GLD": 1.0}, total_value=50_000),
    ]
    snapshot = {
        p.name: {
            "total_value": p.total_value,
            "weights": tuple((h.stashaway_id, h.weight) for h in p.holdings),
            "roles": frozenset(p.roles),
        }
        for p in book
    }
    drop_portfolio(book, "B")
    set_weights(book, "A", {"VTI": 0.5, "BND": 0.5})
    rebalance_into(book, "B", "A")
    merge_into(book, ["A", "B"], "AB")
    for p in book:
        assert p.total_value == snapshot[p.name]["total_value"]
        assert (
            tuple((h.stashaway_id, h.weight) for h in p.holdings)
            == snapshot[p.name]["weights"]
        )
        assert frozenset(p.roles) == snapshot[p.name]["roles"]
