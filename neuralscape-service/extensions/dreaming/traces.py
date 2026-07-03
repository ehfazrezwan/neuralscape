"""Recall traces — the reinforcement signal for dream scoring.

Every ``MemoryService.search()`` fire-and-forgets a trace here: which
memory ids were returned, for which query. Aggregates power the deep-phase
promotion score and the Ebbinghaus retention strength (scoring.py).

Redis layout (all keys carry a rolling TTL, refreshed on write):

- ``dreaming:tr:count``  — HASH  memory_id → total recall count
- ``dreaming:tr:last``   — HASH  memory_id → unix ts of last recall
- ``dreaming:tr:q:<id>`` — HLL   distinct query hashes for that memory

The write path must never block or fail a recall: writes run on a single
daemon worker thread and every Redis op is wrapped. Keys are global (not
per pool) — memory ids are unique, so pool separation adds nothing.
"""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_COUNT_KEY = "dreaming:tr:count"
_LAST_KEY = "dreaming:tr:last"
_QUERY_KEY = "dreaming:tr:q:{mid}"

# One background thread: traces are tiny pipelined writes; ordering across
# recalls is irrelevant. The executor is process-global and daemonized so
# it never blocks interpreter shutdown.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dream-trace")

_redis_client = None


def _get_redis():
    """Lazy sync Redis client from core settings (best-effort)."""
    global _redis_client
    if _redis_client is None:
        import redis as redis_lib

        from config import settings

        _redis_client = redis_lib.Redis.from_url(
            settings.redis_url, socket_timeout=2, socket_connect_timeout=2
        )
    return _redis_client


def query_hash(query: str) -> str:
    """Stable short hash of a recall query (for the uniqueness HLL)."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def log_recall(memory_ids: list[str], query: str, *, ttl_days: int = 30) -> None:
    """Fire-and-forget a recall trace. NEVER raises; never blocks the read."""
    if not memory_ids:
        return
    try:
        _executor.submit(_write_trace, list(memory_ids), query, ttl_days)
    except Exception:  # executor shut down at interpreter exit — drop it
        pass


def _write_trace(memory_ids: list[str], query: str, ttl_days: int) -> None:
    try:
        r = _get_redis()
        qh = query_hash(query)
        now = time.time()
        ttl = ttl_days * 86400
        pipe = r.pipeline(transaction=False)
        for mid in memory_ids:
            pipe.hincrby(_COUNT_KEY, mid, 1)
            pipe.hset(_LAST_KEY, mid, now)
            pipe.pfadd(_QUERY_KEY.format(mid=mid), qh)
            pipe.expire(_QUERY_KEY.format(mid=mid), ttl)
        pipe.expire(_COUNT_KEY, ttl)
        pipe.expire(_LAST_KEY, ttl)
        pipe.execute()
    except Exception:
        # Best-effort by contract: a dead Redis must not affect recall.
        logger.debug("recall trace write dropped", exc_info=True)


def read_aggregates(redis, memory_ids: list[str]) -> dict[str, dict]:
    """Read trace aggregates for the given ids (used at dream time).

    Returns ``{memory_id: {recall_count, unique_query_count, last_recalled_at}}``
    with zeros for never-recalled ids. Errors degrade to all-zeros — a dead
    trace store makes scoring coarser, never breaks a sweep.
    """
    out = {
        mid: {"recall_count": 0, "unique_query_count": 0, "last_recalled_at": 0.0}
        for mid in memory_ids
    }
    if not memory_ids:
        return out
    try:
        pipe = redis.pipeline(transaction=False)
        pipe.hmget(_COUNT_KEY, memory_ids)
        pipe.hmget(_LAST_KEY, memory_ids)
        for mid in memory_ids:
            pipe.pfcount(_QUERY_KEY.format(mid=mid))
        results = pipe.execute()
        counts, lasts, uniques = results[0], results[1], results[2:]
        for i, mid in enumerate(memory_ids):
            out[mid] = {
                "recall_count": int(counts[i] or 0),
                "unique_query_count": int(uniques[i] or 0),
                "last_recalled_at": float(lasts[i] or 0.0),
            }
    except Exception:
        logger.warning("trace aggregate read failed — scoring degrades", exc_info=True)
    return out
