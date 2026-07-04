"""Recall traces — the reinforcement signal for dream scoring.

Every ``MemoryService.search()`` fire-and-forgets a trace here: which
memory ids were returned, for which query. Aggregates power the deep-phase
promotion score and the Ebbinghaus retention strength (scoring.py).

Redis layout (all keys carry a rolling TTL, refreshed on write):

- ``dreaming:tr:count``  — HASH  memory_id → total recall count
- ``dreaming:tr:last``   — HASH  memory_id → unix ts of last recall
- ``dreaming:tr:q:<id>`` — HLL   distinct query hashes for that memory
- ``dreaming:dyn``       — HASH  memory_id → JSON salience-dynamics state
  (A4; strength / stability / last activation / recall + co-recall
  counts — see dynamics.py for the math, this module only does the I/O)

The write path must never block or fail a recall: writes run on a single
daemon worker thread and every Redis op is wrapped. Keys are global (not
per pool) — memory ids are unique, so pool separation adds nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_COUNT_KEY = "dreaming:tr:count"
_LAST_KEY = "dreaming:tr:last"
_QUERY_KEY = "dreaming:tr:q:{mid}"
_DYN_KEY = "dreaming:dyn"

# One background thread: traces are tiny pipelined writes; ordering across
# recalls is irrelevant. The executor is process-global and daemonized so
# it never blocks interpreter shutdown.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dream-trace")

# Audit 27 #13: bound the queue. Traces are best-effort by contract — when
# the worker can't keep up (dead/slow Redis), new traces are DROPPED with a
# debug log instead of growing an unbounded backlog in process memory.
MAX_PENDING_TRACES = 1000

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
    """Fire-and-forget a recall trace. NEVER raises; never blocks the read.

    Drops the trace (debug log) when the queue is at MAX_PENDING_TRACES —
    a backlog that deep means Redis is down/stalled, and traces are a
    best-effort reinforcement signal, never worth unbounded memory.
    """
    if not memory_ids:
        return
    try:
        if _executor._work_queue.qsize() >= MAX_PENDING_TRACES:
            logger.debug(
                "recall-trace queue full (>= %d) — dropping trace",
                MAX_PENDING_TRACES,
            )
            return
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
        results = pipe.execute()
        _write_dynamics(r, memory_ids, results, now, ttl)
    except Exception:
        # Best-effort by contract: a dead Redis must not affect recall.
        logger.debug("recall trace write dropped", exc_info=True)


def _write_dynamics(r, memory_ids: list[str], trace_results: list, now: float, ttl: int) -> None:
    """A4: fold this recall into each memory's salience-dynamics state.

    Same contract as the trace write it piggybacks on: fire-and-forget on
    the daemon thread, best-effort, never raises. The PFADD replies inside
    ``trace_results`` (position ``i*4 + 2`` per memory) say whether this
    query hash was NEW for that memory — the novelty flag that damps
    single-query hammering (guardrail 3). Co-recall = this one query
    returned ≥ 2 memories together (Hebbian co-activation).
    """
    try:
        from .config import dreaming_settings

        if not dreaming_settings.dynamics_enabled:
            return
        from . import dynamics

        # The trace pipeline queues exactly 4 ops per memory (hincrby, hset,
        # pfadd, expire) plus 2 trailing expires; PFADD replies sit at stride
        # 4, offset 2. Guard the shape so a future pipeline change degrades
        # to novel=True (no damping) instead of silently mis-damping.
        expected_len = len(memory_ids) * 4 + 2
        if isinstance(trace_results, (list, tuple)) and len(trace_results) == expected_len:
            pfadd_replies = trace_results[2 : len(memory_ids) * 4 : 4]
        else:
            pfadd_replies = [None] * len(memory_ids)
        novel: list[bool] = []
        for reply in pfadd_replies:
            try:
                novel.append(bool(int(reply)))
            except (TypeError, ValueError):
                novel.append(True)  # unknown → don't damp
        co_recalled = len(memory_ids) >= 2
        raw_states = r.hmget(_DYN_KEY, memory_ids)
        pipe = r.pipeline(transaction=False)
        for mid, raw, is_novel in zip(memory_ids, raw_states, novel):
            state = dynamics.reinforce(
                dynamics.from_dict(raw),
                now=now,
                co_recalled=co_recalled,
                novel_query=is_novel,
                delta=dreaming_settings.dynamics_strength_delta,
                cap=dreaming_settings.dynamics_strength_cap,
            )
            pipe.hset(_DYN_KEY, mid, json.dumps(dynamics.to_dict(state)))
        pipe.expire(_DYN_KEY, ttl)
        pipe.execute()
    except Exception:
        logger.debug("salience dynamics write dropped", exc_info=True)


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


def read_dynamics(redis, memory_ids: list[str]) -> dict:
    """Read per-memory dynamics states for a staged batch (dream time).

    Returns ``{memory_id: DynamicsState}`` for ids that HAVE a persisted
    state; never-recalled ids are simply absent so scoring can fall back
    to the legacy single-half-life path. Errors degrade to ``{}`` — a dead
    trace store makes retention coarser, never breaks a sweep.
    """
    from . import dynamics

    out: dict = {}
    if not memory_ids:
        return out
    try:
        raw_states = redis.hmget(_DYN_KEY, memory_ids)
        for mid, raw in zip(memory_ids, raw_states):
            if raw:
                out[mid] = dynamics.from_dict(raw)
    except Exception:
        logger.warning(
            "dynamics state read failed — retention falls back to half-life",
            exc_info=True,
        )
    return out


def get_strength_signals(memory_ids: list[str]) -> dict[str, float]:
    """Best-effort strength signals for the bounded recall tie-breaker.

    Called from the search path ONLY when ``DREAMING_SALIENCE_RECALL_K > 0``
    — at the k=0 default the search path never touches this module and
    stays byte-identical. One HMGET on the module's lazy client; any
    failure returns ``{}`` (no boost), never an error into the read.
    """
    from . import dynamics

    out: dict[str, float] = {}
    if not memory_ids:
        return out
    try:
        r = _get_redis()
        raw_states = r.hmget(_DYN_KEY, memory_ids)
        for mid, raw in zip(memory_ids, raw_states):
            if raw:
                out[mid] = dynamics.strength_signal(dynamics.from_dict(raw))
    except Exception:
        logger.debug("strength-signal read dropped (no recall boost)", exc_info=True)
    return out
