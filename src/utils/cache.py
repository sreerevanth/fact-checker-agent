"""
Cache Layer
===========
Redis-backed cache for verification results and rate limiting.
Falls back to in-memory diskcache if Redis is unavailable.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from src.utils.logging import get_logger
from src.utils.settings import get_settings

log = get_logger(__name__)


def _claim_key(claim_text: str) -> str:
    h = hashlib.sha256(claim_text.strip().lower().encode()).hexdigest()[:16]
    return f"factcheck:result:{h}"


class CacheBackend:
    """Thin async wrapper — Redis preferred, diskcache fallback."""

    def __init__(self) -> None:
        self._redis: Any = None
        self._disk: Any = None
        self._ttl = get_settings().cache_ttl_seconds

    async def initialize(self) -> None:
        settings = get_settings()
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
            await self._redis.ping()
            log.info("Cache: Redis connected", url=settings.redis_url)
        except Exception as exc:
            log.warning("Redis unavailable, using diskcache", error=str(exc))
            import diskcache
            self._disk = diskcache.Cache("./data/diskcache")

    async def get(self, key: str) -> Optional[str]:
        try:
            if self._redis:
                return await self._redis.get(key)
            if self._disk:
                return self._disk.get(key)
        except Exception as exc:
            log.debug("Cache get failed", key=key, error=str(exc))
        return None

    async def set(self, key: str, value: str) -> None:
        try:
            if self._redis:
                await self._redis.setex(key, self._ttl, value)
            elif self._disk:
                self._disk.set(key, value, expire=self._ttl)
        except Exception as exc:
            log.debug("Cache set failed", key=key, error=str(exc))

    async def close(self) -> None:
        try:
            if self._redis:
                await self._redis.close()
            if self._disk:
                self._disk.close()
        except Exception:
            pass


_cache: Optional[CacheBackend] = None


async def get_cache() -> CacheBackend:
    global _cache
    if _cache is None:
        _cache = CacheBackend()
        await _cache.initialize()
    return _cache


async def cache_result(claim_text: str, result_json: str) -> None:
    c = await get_cache()
    await c.set(_claim_key(claim_text), result_json)


async def get_cached_result(claim_text: str) -> Optional[str]:
    c = await get_cache()
    return await c.get(_claim_key(claim_text))
