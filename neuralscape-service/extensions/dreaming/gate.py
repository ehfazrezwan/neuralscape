"""Dream gate — per-pool activity gates and a distributed lock.

A Python/Redis port of the gate economy in the vendored TS reference
(``mem0/openclaw/dream-gate.ts``), adapted to Neuralscape's multi-process
deploy shape: state lives in Redis (not a local file) so the API and the
graph worker agree, and the lock is ``SET NX PX`` (not an exclusive-create
file) so concurrent sweeps — or a sweep racing the dedup cron on the same
pool — are impossible.

Gate order is cheap → expensive:

1. **Time** (one Redis read): hours since the pool's last completed dream.
2. **Volume** (counted during the LIGHT scroll, passed in by the caller):
   new/changed memories since the last dream.

The session gate from the reference (``minSessions``) has no analog here —
Neuralscape's service layer has no session concept — so volume carries that
weight alone.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_GATE_KEY = "dreaming:gate:{pool}"
_LOCK_KEY = "dreaming:lock:{pool}"
_LOCK_STALE_MS = 60 * 60 * 1000  # 1 hour, matching the reference


@dataclass(slots=True)
class GateDecision:
    proceed: bool
    reason: str = ""


def get_gate_state(redis, pool: str) -> dict:
    """Return the pool's gate state: ``{last_dreamt_at: float|0}``."""
    try:
        raw = redis.get(_GATE_KEY.format(pool=pool))
        if raw:
            return json.loads(raw)
    except Exception:
        logger.warning("dream gate state read failed for %s", pool, exc_info=True)
    return {"last_dreamt_at": 0.0}


def check_time_gate(redis, pool: str, *, min_hours: float, now: float | None = None) -> GateDecision:
    """Cheap gate: enough wall-clock since this pool's last dream?"""
    state = get_gate_state(redis, pool)
    now = time.time() if now is None else now
    hours_since = (now - float(state.get("last_dreamt_at") or 0.0)) / 3600.0
    if hours_since < min_hours:
        return GateDecision(False, f"time: {hours_since:.1f}h < {min_hours}h")
    return GateDecision(True)


def check_volume_gate(new_memory_count: int, *, min_new_memories: int) -> GateDecision:
    """Expensive gate: enough new/changed material to justify an LLM pass?

    The count comes from the LIGHT scroll — the caller already paid for it,
    so this is pure arithmetic.
    """
    if new_memory_count < min_new_memories:
        return GateDecision(
            False, f"volume: {new_memory_count} < {min_new_memories} new memories"
        )
    return GateDecision(True)


def acquire_lock(redis, pool: str) -> bool:
    """Take the pool's dream lock. Stale locks (>1h) are reclaimed.

    ``SET NX PX`` is atomic across processes; the PX expiry doubles as the
    stale-reclaim so a crashed sweep can never wedge a pool forever.
    """
    key = _LOCK_KEY.format(pool=pool)
    try:
        return bool(redis.set(key, str(time.time()), nx=True, px=_LOCK_STALE_MS))
    except Exception:
        logger.warning("dream lock acquire failed for %s", pool, exc_info=True)
        return False


def release_lock(redis, pool: str) -> None:
    try:
        redis.delete(_LOCK_KEY.format(pool=pool))
    except Exception:
        logger.warning("dream lock release failed for %s", pool, exc_info=True)


def is_locked(redis, pool: str) -> bool:
    """True while a sweep (or a coordinated cron) holds the pool's lock."""
    try:
        return bool(redis.exists(_LOCK_KEY.format(pool=pool)))
    except Exception:
        return False


def record_completion(redis, pool: str, *, now: float | None = None) -> None:
    """Stamp a successful sweep; resets the time gate."""
    state = {"last_dreamt_at": time.time() if now is None else now}
    try:
        redis.set(_GATE_KEY.format(pool=pool), json.dumps(state))
    except Exception:
        logger.warning("dream gate completion write failed for %s", pool, exc_info=True)
