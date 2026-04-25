"""Transparent disk cache for market-data requests (Parquet + metadata)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
from loguru import logger


class DataCache:
    """File-system cache backed by Parquet files.

    Cache keys are SHA-256 hashes of the serialised request parameters so
    the same data is never fetched twice within the TTL window.
    """

    def __init__(self, cache_dir: str | Path = "data/cache", ttl_hours: float = 24.0) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_hours * 3600
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public

    def get(self, key: str) -> pd.DataFrame | None:
        parquet, meta = self._paths(key)
        if not parquet.exists() or not meta.exists():
            return None
        metadata = json.loads(meta.read_text())
        if time.time() - metadata["created_at"] > self.ttl_seconds:
            logger.debug("Cache expired for key={}", key[:12])
            parquet.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
            return None
        logger.debug("Cache hit key={}", key[:12])
        return pd.read_parquet(parquet)

    def set(self, key: str, df: pd.DataFrame) -> None:
        parquet, meta = self._paths(key)
        df.to_parquet(parquet)
        meta.write_text(json.dumps({"created_at": time.time(), "rows": len(df)}))
        logger.debug("Cached {} rows key={}", len(df), key[:12])

    def invalidate(self, key: str) -> None:
        for p in self._paths(key):
            p.unlink(missing_ok=True)

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        for f in self.cache_dir.glob("*.meta.json"):
            f.unlink()
        logger.info("Cache cleared: {}", self.cache_dir)

    # ------------------------------------------------------------------ util

    @staticmethod
    def make_key(**kwargs: object) -> str:
        payload = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path]:
        base = self.cache_dir / key
        return base.with_suffix(".parquet"), base.with_suffix(".meta.json")
