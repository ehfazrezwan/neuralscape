"""Phase G — cross-process persistence for code-index metadata + project config.

Two things must survive the API↔ingest-worker process boundary and outlive a
single request:

  1. **Per-code_space index metadata** — ``{system, repo_source, repo_sha,
     indexed_at, engine_version, symbols, edges}`` recorded when a corpus is
     indexed through NS (PLAN §5: "Index task records {repo_sha, indexed_at,
     engine_version} per code_space"). Drives staleness hints.
  2. **Per-project routing config** — the ``ProjectKnowledgeConfig`` the router's
     layer 2 reads (``code_systems`` / ``default_engine`` / ``fuse_code_into_recall``).
     Set at index time so subsequent recalls route to the indexed engine.

Backed by Redis (already a hard dependency for ARQ), **best-effort**: every call
degrades to an in-memory dict when Redis is unavailable (unit tests, Redis down).
The in-memory dict is also an L1 cache so the router's hot path pays at most one
bounded Redis GET on a config miss — output stays byte-identical (a miss returns
None, exactly as before Phase G).
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_KEY_PREFIX = "ns:code-index:"          # + code_space  → metadata hash (JSON)
_CFG_PREFIX = "ns:code-projcfg:"        # + project_id  → project config (JSON)

# Bounded so a wedged Redis can't stall the request path (router <1ms budget).
# Payloads are tiny JSON blobs, so a healthy Redis answers in well under a
# millisecond; 0.25 s is the worst-case cost of ONE failed read/connect.
_SOCKET_TIMEOUT = 0.25

# In-memory L1 (also the sole store when Redis is unavailable). Entries carry a
# monotonic write time; when Redis is up, a positive hit older than _L1_TTL is
# re-read from Redis so a reindex that changed default_engine/code_space becomes
# visible to a long-running process (Fable should-fix a).
_mem_meta: dict[str, dict] = {}
_mem_cfg: dict[str, dict] = {}
_mem_cfg_at: dict[str, float] = {}
# project_id -> monotonic time of the last authoritative "no config" answer.
# Without it every recall for an unconfigured project (the common case) pays a
# Redis GET. Valid for one config_ttl(); cleared by a local save.
_mem_cfg_miss_at: dict[str, float] = {}
# Process-wide: after a failed Redis config read (down, timeout, error), skip
# Redis for config reads until this monotonic time, so a wedged Redis costs at
# most ONE bounded read per config_ttl() per process, not one per project.
_cfg_down_until = 0.0
_L1_TTL = 30.0  # fallback when settings are unavailable


def config_ttl() -> float:
    """Seconds a cached project config may be served before a Redis re-read.

    Settings-driven (``knowledge_config_cache_ttl_s``) so the router mirror and
    this L1 share one staleness bound; falls back to _L1_TTL.
    """
    try:
        from config import settings

        return max(0.0, float(settings.knowledge_config_cache_ttl_s))
    except Exception:  # noqa: BLE001
        return _L1_TTL

_client = None
_client_probe_at = 0.0
_CLIENT_RETRY = 30.0


def _redis():
    """Lazy best-effort sync Redis client; None when unavailable.

    Retries the connection at most every _CLIENT_RETRY seconds rather than
    latching a single failure forever (Fable should-fix b), so a Redis that was
    briefly down at first touch becomes usable without a process restart.
    """
    global _client, _client_probe_at
    import time as _t

    if _client is not None:
        return _client
    now = _t.monotonic()
    if now - _client_probe_at < _CLIENT_RETRY:
        return None
    _client_probe_at = now
    try:
        import redis  # redis-py (dependency of arq)

        from config import settings

        _client = redis.Redis.from_url(
            settings.redis_url,
            socket_timeout=_SOCKET_TIMEOUT,
            socket_connect_timeout=_SOCKET_TIMEOUT,
            decode_responses=True,
        )
        # Cheap liveness ping so a dead Redis falls back to memory immediately.
        _client.ping()
    except Exception:  # noqa: BLE001 — no Redis ⇒ in-memory only
        logger.info("code index_store: Redis unavailable, using in-memory store")
        _client = None
    return _client


# ── Index metadata ──────────────────────────────────────────────────


def record_index(code_space: str, meta: dict) -> None:
    """Record index metadata for a code_space (best-effort; never raises)."""
    _mem_meta[code_space] = dict(meta)
    r = _redis()
    if r is None:
        return
    try:
        r.set(_KEY_PREFIX + code_space, json.dumps(meta, default=str))
    except Exception:  # noqa: BLE001
        logger.debug("record_index Redis write failed (non-fatal)", exc_info=True)


def get_index(code_space: str) -> dict | None:
    """Fetch index metadata for a code_space (L1 → Redis). None if unknown."""
    hit = _mem_meta.get(code_space)
    if hit is not None:
        return hit
    r = _redis()
    if r is None:
        return None
    try:
        raw = r.get(_KEY_PREFIX + code_space)
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    try:
        meta = json.loads(raw)
    except (ValueError, TypeError):
        return None
    _mem_meta[code_space] = meta
    return meta


# ── Project routing config ──────────────────────────────────────────


def save_project_config(project_id: str, config: dict) -> None:
    """Persist a project's routing config (best-effort; never raises)."""
    import time as _t

    _mem_cfg[project_id] = dict(config)
    _mem_cfg_at[project_id] = _t.monotonic()
    _mem_cfg_miss_at.pop(project_id, None)
    r = _redis()
    if r is None:
        return
    try:
        r.set(_CFG_PREFIX + project_id, json.dumps(config, default=str))
    except Exception:  # noqa: BLE001
        logger.debug("save_project_config Redis write failed (non-fatal)", exc_info=True)


def load_project_config(project_id: str, max_age: float | None = None) -> dict | None:
    """Fetch a project's routing config (L1 → Redis). None if unset.

    ``max_age`` overrides the L1 freshness bound for this call (0 forces a
    Redis read; used by the router's refresh so one bound governs both layers).

    On the router hot path: a fresh L1 hit is free; a miss (or an L1 entry older
    than _L1_TTL) costs at most one bounded Redis GET (then cached). A None result
    keeps recall output byte-identical. The TTL re-read lets a reindex that
    changed default_engine/code_space become visible to a long-running process
    (Fable should-fix a).
    """
    return _load_project_config(project_id, max_age)[1]


def refresh_project_config(project_id: str) -> tuple[bool, dict | None]:
    """Force a Redis read, bypassing L1. Returns (authoritative, config).

    ``authoritative`` is True only when Redis answered. Then a missing key means
    the config is gone (another process deleted or cleared it): the L1 entry is
    dropped and (True, None) returned. Redis down/erroring or an unparseable
    value is (False, <L1 value or None>): the caller keeps what it has.
    """
    return _load_project_config(project_id, 0.0)


def _load_project_config(project_id: str, max_age: float | None) -> tuple[bool, dict | None]:
    """Worst case per call: one Redis read bounded by ``_SOCKET_TIMEOUT``; after a
    failure no further Redis read for ``config_ttl()`` seconds (process-wide)."""
    global _cfg_down_until
    import time as _t

    ttl = config_ttl() if max_age is None else max_age
    now = _t.monotonic()
    hit = _mem_cfg.get(project_id)
    if hit is not None and (now - _mem_cfg_at.get(project_id, 0.0)) < ttl:
        return False, hit
    if hit is None and max_age is None:
        missed_at = _mem_cfg_miss_at.get(project_id)
        if missed_at is not None and (now - missed_at) < ttl:
            return False, None  # negative-cached: no Redis I/O
    if now < _cfg_down_until:
        return False, hit  # Redis failed recently: serve L1 without retrying

    def _down() -> tuple[bool, dict | None]:
        global _cfg_down_until
        _cfg_down_until = _t.monotonic() + max(config_ttl(), 1.0)
        return False, hit

    r = _redis()
    if r is None:
        return _down()  # Redis down → serve the (possibly stale) L1 value, if any
    try:
        raw = r.get(_CFG_PREFIX + project_id)
    except Exception:  # noqa: BLE001
        return _down()
    if not raw:
        # Redis answered and the key is absent: authoritative, drop L1.
        _mem_cfg.pop(project_id, None)
        _mem_cfg_at.pop(project_id, None)
        _mem_cfg_miss_at[project_id] = _t.monotonic()
        return True, None
    try:
        cfg = json.loads(raw)
    except (ValueError, TypeError):
        return False, hit
    _mem_cfg[project_id] = cfg
    _mem_cfg_at[project_id] = _t.monotonic()
    _mem_cfg_miss_at.pop(project_id, None)
    return True, cfg


def _reset_for_tests() -> None:
    """Clear in-memory caches + force client re-resolution (test hook)."""
    global _client, _client_probe_at, _cfg_down_until
    _mem_meta.clear()
    _mem_cfg.clear()
    _mem_cfg_at.clear()
    _mem_cfg_miss_at.clear()
    _cfg_down_until = 0.0
    _client = None
    _client_probe_at = 0.0
