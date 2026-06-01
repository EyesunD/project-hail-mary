"""Statement parser + cache."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from hailmary.allocation.statements import (
    ParsedHolding,
    ParsedPortfolio,
    StatementCache,
    StatementParseError,
    _CacheKey,
    load_holdings_from_json,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_load_holdings_from_json_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "manual.json"
    _write_json(
        p,
        {
            "statement_date": "2024-09-30",
            "currency": "SGD",
            "portfolios": [
                {
                    "name": "Custom Growth",
                    "total_value": 100_000,
                    "holdings": [
                        {"ticker": "VTI", "weight": 0.7, "value": 70_000},
                        {"ticker": "VEA", "weight": 0.3, "value": 30_000},
                    ],
                }
            ],
        },
    )
    portfolios = load_holdings_from_json(p)
    assert len(portfolios) == 1
    pf = portfolios[0]
    assert pf.name == "Custom Growth"
    assert pf.currency == "SGD"
    assert len(pf.holdings) == 2
    assert abs(pf.weight_sum() - 1.0) < 1e-9


def test_load_holdings_from_json_invalid_weights(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    _write_json(
        p,
        {
            "statement_date": "2024-09-30",
            "currency": "USD",
            "portfolios": [
                {
                    "name": "Bad Portfolio",
                    "total_value": 100_000,
                    "holdings": [
                        {"ticker": "VTI", "weight": 0.6, "value": 60_000},
                        {"ticker": "VEA", "weight": 0.3, "value": 30_000},
                        # sums to 0.9 — should fail validation
                    ],
                }
            ],
        },
    )
    with pytest.raises(StatementParseError):
        load_holdings_from_json(p)


def test_load_holdings_from_json_missing_file(tmp_path: Path) -> None:
    with pytest.raises(StatementParseError):
        load_holdings_from_json(tmp_path / "nope.json")


def test_load_holdings_from_json_malformed(tmp_path: Path) -> None:
    p = tmp_path / "malformed.json"
    p.write_text("{not valid json")
    with pytest.raises(StatementParseError):
        load_holdings_from_json(p)


def test_load_anonymised_book_fixture() -> None:
    """Regression: the committed anonymised fixture parses cleanly."""
    fixture = Path("tests/fixtures/statements/example_book.json")
    if not fixture.exists():
        pytest.skip("Anonymised fixture not generated yet")
    portfolios = load_holdings_from_json(fixture)
    assert len(portfolios) == 15
    expected_names = {
        "BlackRock", "Energy", "General Investing", "Simple USD",
        "Singapore Investing", "Utilities", "Income Investing",
        "High Dividend Yield", "Ex-US Large-cap", "SG ETF",
        "Nasdaq Covered Call", "Cash Pool A", "Simple SGD",
        "General SRS", "Crypto",
    }
    assert {p.name for p in portfolios} == expected_names
    for pf in portfolios:
        assert abs(pf.weight_sum() - 1.0) < 1e-4, f"{pf.name} weights drift"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _sample_portfolios() -> list[ParsedPortfolio]:
    return [
        ParsedPortfolio(
            name="Custom Growth",
            statement_date=__import__("datetime").date(2024, 9, 30),
            currency="USD",
            total_value=100_000,
            holdings=[
                ParsedHolding(ticker="VTI", weight=0.7, value=70_000),
                ParsedHolding(ticker="VEA", weight=0.3, value=30_000),
            ],
        )
    ]


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = StatementCache(cache_dir=tmp_path)
    src = tmp_path / "src.pdf"
    src.write_bytes(b"placeholder")
    key = _CacheKey(name=src.name, mtime_ns=src.stat().st_mtime_ns)
    portfolios = _sample_portfolios()
    cache.set(key, portfolios)

    loaded = cache.get(key)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].name == "Custom Growth"
    assert loaded[0].currency == "USD"
    assert len(loaded[0].holdings) == 2


def test_cache_miss_after_touch(tmp_path: Path) -> None:
    cache = StatementCache(cache_dir=tmp_path)
    src = tmp_path / "src.pdf"
    src.write_bytes(b"placeholder")

    key1 = _CacheKey(name=src.name, mtime_ns=src.stat().st_mtime_ns)
    cache.set(key1, _sample_portfolios())
    assert cache.get(key1) is not None

    # Touch the file so mtime_ns changes
    time.sleep(0.05)
    new_mtime = time.time_ns()
    os.utime(src, ns=(new_mtime, new_mtime))
    key2 = _CacheKey(name=src.name, mtime_ns=src.stat().st_mtime_ns)
    assert key2.mtime_ns != key1.mtime_ns
    assert cache.get(key2) is None  # different key → miss
