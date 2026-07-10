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
_SOCKET_TIMEOUT = 1.0

# In-memory L1 (also the sole store when Redis is unavailable).
_mem_meta: dict[str, dict] = {}
_mem_cfg: dict[str, dict] = {}

_client = None
_client_tried = False


def _redis():
    """Lazy best-effort sync Redis client; None when unavailable."""
    global _client, _client_tried
    if _client_tried:
        return _client
    _client_tried = True
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
    _mem_cfg[project_id] = dict(config)
    r = _redis()
    if r is None:
        return
    try:
        r.set(_CFG_PREFIX + project_id, json.dumps(config, default=str))
    except Exception:  # noqa: BLE001
        logger.debug("save_project_config Redis write failed (non-fatal)", exc_info=True)


def load_project_config(project_id: str) -> dict | None:
    """Fetch a project's routing config (L1 → Redis). None if unset.

    On the router hot path: an L1 hit is free; a miss costs at most one bounded
    Redis GET (then cached). A None result keeps recall output byte-identical.
    """
    hit = _mem_cfg.get(project_id)
    if hit is not None:
        return hit
    r = _redis()
    if r is None:
        return None
    try:
        raw = r.get(_CFG_PREFIX + project_id)
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    try:
        cfg = json.loads(raw)
    except (ValueError, TypeError):
        return None
    _mem_cfg[project_id] = cfg
    return cfg


def _reset_for_tests() -> None:
    """Clear in-memory caches + force client re-resolution (test hook)."""
    global _client, _client_tried
    _mem_meta.clear()
    _mem_cfg.clear()
    _client = None
    _client_tried = False
