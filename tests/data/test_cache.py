"""Tests for DataCache."""

import pandas as pd
import pytest

from hailmary.data.cache import DataCache


def test_cache_roundtrip(tmp_path: object) -> None:
    cache = DataCache(cache_dir=tmp_path, ttl_hours=1)  # type: ignore[arg-type]
    df = pd.DataFrame({"a": [1, 2, 3]})
    key = DataCache.make_key(test="roundtrip")
    assert cache.get(key) is None
    cache.set(key, df)
    result = cache.get(key)
    assert result is not None
    pd.testing.assert_frame_equal(df, result)


def test_cache_expiry(tmp_path: object) -> None:
    cache = DataCache(cache_dir=tmp_path, ttl_hours=0)  # TTL=0 → always expired
    df = pd.DataFrame({"x": [10]})
    key = DataCache.make_key(test="expiry")
    cache.set(key, df)
    result = cache.get(key)
    assert result is None


def test_cache_invalidate(tmp_path: object) -> None:
    cache = DataCache(cache_dir=tmp_path, ttl_hours=24)
    df = pd.DataFrame({"v": [1]})
    key = DataCache.make_key(test="invalidate")
    cache.set(key, df)
    cache.invalidate(key)
    assert cache.get(key) is None
