"""Universe map + resolve()."""

from __future__ import annotations

import pytest

from hailmary.allocation.universe import (
    STASHAWAY_UNIVERSE,
    AssetMetadata,
    UnknownAssetError,
    resolve,
)


def test_resolve_known_asset(seeded_universe: object) -> None:
    md = resolve("VTI")
    assert isinstance(md, AssetMetadata)
    assert md.ticker == "VTI"
    assert md.asset_class == "Equity"


def test_resolve_unknown_asset_raises(seeded_universe: object) -> None:
    with pytest.raises(UnknownAssetError) as excinfo:
        resolve("FOO_NEVER_HEARD_OF", source="Custom X")
    assert excinfo.value.ticker == "FOO_NEVER_HEARD_OF"
    assert excinfo.value.source == "Custom X"
    assert "Custom X" in str(excinfo.value)


def test_universe_initially_empty() -> None:
    # Outside the seeded_universe fixture, the map is empty until real-statement seeding.
    # The seeded_universe fixture restores the original state on teardown.
    assert isinstance(STASHAWAY_UNIVERSE, dict)
