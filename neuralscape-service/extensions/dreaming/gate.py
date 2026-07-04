"""Dream gate — per-pool activity gates and a distributed lock.

A Python/Redis port of the gate economy in the vendored TS reference
(``mem0/openclaw/dream-gate.ts``), adapted to Neuralscape's multi-process
deploy shape: state lives in Redis (not a local file) so the API and the
graph worker agree, and the lock is ``SET NX PX`` (not an exclusive-create
file) so concurrent sweeps — or a sweep racing the dedup cron on the same
pool — are impossible.

Gate order is cheap → expensive:

1. **Time** (one Redis read): hours since the pool's last completed dream.
2. **Settling** (A3-lite, pure arithmetic over the already-scrolled pool):
   a pool written to in the last N minutes is mid-conversation — defer.
3. **Volume** (counted during the LIGHT scroll, passed in by the caller):
   new/changed memories since the last dream.

The session gate from the reference (``minSessions``) has no analog here —
Neuralscape's service layer has no session concept — so volume carries that
weight alone.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_GATE_KEY = "dreaming:gate:{pool}"
_LOCK_KEY = "dreaming:lock:{pool}"
_LOCK_STALE_MS = 60 * 60 * 1000  # 1 hour, matching the reference

# Compare-and-delete: only the owner's token may clear the lock, so a sweep
# that outlives the stale window can't release a lock another worker has
# since reclaimed.
_RELEASE_SCRIPT = (
    "if redis.call('get', KEYS[1]) == ARGV[1] "
    "then return redis.call('del', KEYS[1]) else return 0 end"
)


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


def check_settling_gate(
    memories: list[dict], *, settling_minutes: float, now: float | None = None
) -> GateDecision:
    """A3-lite settling guard: defer pools that took a write moments ago.

    "Consolidate settled data" (Honcho): a pool whose newest
    ``created_at``/``updated_at`` is younger than ``settling_minutes`` is
    likely mid-conversation — dreaming over it would consolidate a
    thought while it's still forming. Pure arithmetic over the batch the
    caller already scrolled; the sweep reports status ``"settling"`` and
    retries next pass. ``force=true`` bypasses (the caller simply doesn't
    invoke this); ``settling_minutes <= 0`` disables the guard.
    """
    from .scoring import _parse_ts

    if settling_minutes <= 0:
        return GateDecision(True)
    now = time.time() if now is None else now
    last_write = 0.0
    for mem in memories or []:
        for field in ("created_at", "updated_at"):
            last_write = max(last_write, _parse_ts(mem.get(field)))
    if last_write <= 0:
        return GateDecision(True)
    minutes_since = (now - last_write) / 60.0
    if minutes_since < settling_minutes:
        return GateDecision(
            False,
            f"settling: last write {minutes_since:.1f}m ago < {settling_minutes:g}m",
        )
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


def acquire_lock(redis, pool: str) -> str | None:
    """Take the pool's dream lock. Returns an owner token, or None if held.

    ``SET NX PX`` is atomic across processes; the PX expiry doubles as the
    stale-reclaim (>1h) so a crashed sweep can never wedge a pool forever.
    Pass the token back to :func:`release_lock` — release is a no-op unless
    the token still owns the key.
    """
    key = _LOCK_KEY.format(pool=pool)
    token = uuid.uuid4().hex
    try:
        if redis.set(key, token, nx=True, px=_LOCK_STALE_MS):
            return token
        return None
    except Exception:
        logger.warning("dream lock acquire failed for %s", pool, exc_info=True)
        return None


def release_lock(redis, pool: str, token: str | None) -> None:
    """Release the pool lock iff ``token`` still owns it (compare-and-delete)."""
    if not token:
        return
    key = _LOCK_KEY.format(pool=pool)
    try:
        if hasattr(redis, "eval"):
            redis.eval(_RELEASE_SCRIPT, 1, key, token)
        else:  # test doubles without scripting: non-atomic compare-and-delete
            current = redis.get(key)
            if isinstance(current, (bytes, bytearray)):
                current = current.decode()
            if current == token:
                redis.delete(key)
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
