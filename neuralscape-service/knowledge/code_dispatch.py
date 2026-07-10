"""Phase G — bind a knowledge_system name to a REAL per-code_space engine.

The registry (`knowledge/__init__.py`) holds one CAPABILITY-placeholder entry per
code backend (`code-cbm`, `code-graphify-lib`): its `_engine` is not bound to any
project (CBM `project=None`; graphify `code_space="__registry_capability__"`,
`G=None`). Those placeholders answer health/eligibility, but they cannot serve a
real query for a specific corpus.

This module is the missing seam: given an explicit `knowledge_system` name + a
`code_space` (the `graph_id` ref the caller already passes), build/resolve the
real `CodeIntelEngine` for that space and wrap it in a fresh
`CodeKnowledgeSystem` so the REST/MCP code tools and the recall-fusion path can
call `system.recall(...)` uniformly.

Cross-process safety (index runs on the ingest worker; recall runs on the API):
  - CBM: state lives in the bridge service. The engine binds its project slug via
    the bridge's `/index_status` `root_path` match (``resolve_project_from_source``)
    — no shared memory / Redis needed on the query path.
  - graphify-lib: the in-process NetworkX graph is rebuilt from source (resolved
    from ``settings.code_repos``) on first use in each process; ~0.4s cold, then
    warm in ``query._ctx_cache``.
  - native: reads come straight from Neo4j/Qdrant by code_space.

Nothing here branches on ``transport`` (DECISIONS.md cross-cutting rule) — the
transport is a declared info field carried through from the placeholder entry.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def repo_path_for_code_space(code_space: str, settings) -> str | None:
    """Map a ``code--owner--repo`` code_space to a filesystem repo path.

    Uses ``settings.code_repos`` (the same CODE_REPOS dict the native/graphify
    factories use). Returns None when the space is malformed or the repo isn't
    configured (caller degrades gracefully).
    """
    parts = code_space.split("--")
    if len(parts) < 3 or parts[0] != "code":
        return None
    repo_name = parts[-1]
    repos = getattr(settings, "code_repos", {}) or {}
    repo_path = repos.get(repo_name)
    if not repo_path:
        return None
    return str(Path(os.path.expanduser(repo_path)))


def resolve_code_engine(
    system_name: str,
    code_space: str,
    user_id: str,
    settings,
    *,
    for_index: bool = False,
):
    """Return a real ``CodeIntelEngine`` bound to ``code_space`` for ``system_name``.

    Args:
        system_name: Registry key ("code-cbm" | "code-graphify-lib" | "code-native").
        code_space: Partition key ``code--owner--repo`` (the ``graph_id`` ref).
        user_id: Owner (for repo-path / native resolution).
        settings: App settings (CODE_REPOS, CBM bridge url, ...).
        for_index: True on the index path (skip lazy read-time warmups; the
            caller will call ``engine.index()`` itself).

    Returns:
        A bound engine, or None when it can't be resolved (unknown system, repo
        not configured, bridge down) — callers degrade to the base answer.
    """
    from adapters.code_graph import query as cg_query

    if system_name == "code-cbm":
        return _resolve_cbm_engine(code_space, settings, for_index=for_index)

    if system_name == "code-graphify-lib":
        # Reuse the query.py factory (cached per code_space in _ctx_cache).
        try:
            engine = cg_query._get_graphify_lib_engine(code_space, user_id, settings)
        except Exception:  # noqa: BLE001
            logger.warning(
                "resolve_code_engine: graphify-lib factory failed for %s", code_space,
                exc_info=True,
            )
            return None
        # Read path: the in-process graph must be built. Lazy-build once from
        # source (each process keeps its own warm copy). Index path skips this —
        # the caller invokes index() explicitly.
        if not for_index and getattr(engine, "G", None) is None:
            src = str(getattr(engine, "source_root", "") or "")
            if src:
                logger.info("graphify-lib lazy build for %s from %s", code_space, src)
                try:
                    engine.index(source=src)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "graphify-lib lazy build failed for %s", code_space, exc_info=True
                    )
            # Still no graph → unbindable; degrade (Fable must-fix #2) instead of
            # returning an engine that answers "No graph loaded" as a 200.
            if getattr(engine, "G", None) is None:
                return None
        return engine

    if system_name == "code-native":
        try:
            return cg_query._get_native_engine_by_code_space(code_space, settings)
        except Exception:  # noqa: BLE001
            logger.warning(
                "resolve_code_engine: native factory failed for %s", code_space,
                exc_info=True,
            )
            return None

    logger.warning("resolve_code_engine: unknown system '%s'", system_name)
    return None


# CBM engines are cached per code_space so the bound project slug + http client
# are reused across requests (mirrors query._ctx_cache).
_cbm_cache: dict[str, object] = {}

# Negative cache: code_space → monotonic expiry, for read-path binds that failed
# (bridge down / repo not indexed). Prevents a per-request /index_status probe on
# an unbindable space from recurring (Fable must-fix #1). Short TTL so a
# just-indexed space becomes bindable quickly.
_cbm_unbound_until: dict[str, float] = {}
_CBM_NEG_TTL = 15.0


def _resolve_cbm_engine(code_space: str, settings, *, for_index: bool):
    """Build/get a CBMEngine bound to ``code_space`` (project slug via the bridge).

    Read path: returns None when the project can't be bound (bridge down / repo
    not indexed / not in CODE_REPOS) so the caller degrades cleanly instead of
    later raising a RuntimeError from ``_ensure_project`` (Fable must-fix #2).
    """
    import time

    from adapters.code_graph.cbm_engine import CBMEngine

    ent = _cbm_cache.get(code_space)
    if ent is not None:
        engine = ent
    else:
        bridge_url = os.getenv("CBM_BRIDGE_URL", "http://cbm-bridge:8200")
        engine = CBMEngine(bridge_url=bridge_url, code_space=code_space)
        _cbm_cache[code_space] = engine

    if for_index:
        # The caller's engine.index() sets the slug from the index_repository
        # response; no read-path bind needed.
        return engine

    # Read path: bind the CBM project slug from the bridge if not yet known
    # (index runs in another process).
    if getattr(engine, "project", None):
        return engine

    # Negative cache: skip the probe if we recently failed to bind this space.
    exp = _cbm_unbound_until.get(code_space)
    if exp is not None and time.monotonic() < exp:
        return None

    repo_path = repo_path_for_code_space(code_space, settings)
    slug = engine.resolve_project_from_source(repo_path) if repo_path else None
    if not slug:
        # Unbindable (bridge down / repo not indexed / not configured). Degrade:
        # return None so the caller answers from base / returns a clean error.
        _cbm_unbound_until[code_space] = time.monotonic() + _CBM_NEG_TTL
        return None
    _cbm_unbound_until.pop(code_space, None)
    return engine


def _evict_engine_cache(system_name: str, code_space: str) -> None:
    """Evict any cached engine for (system, code_space) so the next bind is cold."""
    from adapters.code_graph import query as cg_query

    if system_name == "code-native":
        cg_query._ctx_cache.pop(f"native:{code_space}", None)
    elif system_name == "code-graphify-lib":
        cg_query._ctx_cache.pop(f"lib:{code_space}", None)
    elif system_name == "code-cbm":
        _cbm_cache.pop(code_space, None)
        _cbm_unbound_until.pop(code_space, None)


def teardown_code_space(
    system_name: str,
    code_space: str,
    user_id: str,
    settings,
) -> dict:
    """R-C: reset a code system's index for one ``code_space`` (true cold-delete).

    Binds the engine WITHOUT a read-time warmup (``for_index=True`` — graphify
    must not lazily rebuild a graph just to drop it) and delegates to the
    engine's ``teardown()``:
      - native: drop the code_space label-space (CodeRepo/CodeFile/CodeSymbol +
        edges) + code_index symbol cards; CodeAnchor + memory graph preserved.
      - graphify-lib: drop the in-process NetworkX graph + evict the cache.
      - cbm: bridge ``/delete_project``.

    Scoped strictly to the code_space; NEVER touches the memory graph or the
    memory↔code anchors (they join on the memory's ``source_ref`` in Qdrant).
    Idempotent. Returns a structured result; ``deleted`` is False (with a
    reason) when the space is unbindable or the engine has no teardown.
    """
    engine = resolve_code_engine(
        system_name, code_space, user_id, settings, for_index=True
    )
    if engine is None:
        return {
            "deleted": False,
            "system": system_name,
            "code_space": code_space,
            "reason": "unbindable (unknown system / repo not configured / bridge down)",
        }

    # CBM's delete_project needs the project slug; the index ran in another
    # process, so bind it from the bridge if this instance hasn't resolved it.
    if system_name == "code-cbm" and not getattr(engine, "project", None):
        repo_path = repo_path_for_code_space(code_space, settings)
        if repo_path:
            try:
                engine.resolve_project_from_source(repo_path)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "teardown_code_space: could not resolve CBM slug for %s",
                    code_space, exc_info=True,
                )

    teardown = getattr(engine, "teardown", None)
    if not callable(teardown):
        return {
            "deleted": False,
            "system": system_name,
            "code_space": code_space,
            "reason": "engine has no teardown",
        }

    engine_result = teardown() or {}
    _evict_engine_cache(system_name, code_space)
    return {
        "deleted": True,
        "system": system_name,
        "code_space": code_space,
        "engine_result": engine_result,
    }


def resolve_bound_code_system(
    system_name: str,
    code_space: str,
    user_id: str,
    settings,
    *,
    for_index: bool = False,
):
    """Wrap the real per-code_space engine as a ``CodeKnowledgeSystem``.

    Capabilities/transport are carried from the registered placeholder entry so
    the bound system declares exactly what the backend supports (honest N/A) and
    routing/fusion stay transport-agnostic.

    Returns None when the engine can't be resolved (caller degrades to base).
    """
    from knowledge.code_system import CodeKnowledgeSystem
    from knowledge.registry import get_system

    registered = get_system(system_name)

    # If the registry already holds a REAL bound system for THIS EXACT code_space
    # (not a capability placeholder, not a different space), use it directly —
    # this is the production path once per-space systems are registered, and it's
    # what test fakes rely on. Comparing the space (not just "is it real") is the
    # moat guard: registry keys are per-backend, so a per-space entry for
    # code--b--Y must NOT satisfy a request for code--a--X (Fable must-fix #3).
    if registered is not None:
        eng = getattr(registered, "_engine", None)
        cs = getattr(eng, "code_space", None) if eng is not None else None
        if cs and cs != "__registry_capability__" and cs == code_space:
            return registered

    engine = resolve_code_engine(
        system_name, code_space, user_id, settings, for_index=for_index
    )
    if engine is None:
        return None

    # Carry capabilities/transport/version verbatim from the registered
    # placeholder. This is a declaration carry-through into a wrapper constructor,
    # NEVER a routing branch on transport (DECISIONS.md cross-cutting rule).
    # code_dispatch.py is on the transport-invariant test's allowlist as a
    # system-construction file (peer to __init__.py / code_system.py).
    if registered is not None:
        capabilities = registered.info.capabilities
        transport = registered.info.transport
        version = getattr(registered, "_version", None)
    else:
        # System not registered (e.g. CBM disabled) — permissive capability set
        # so an explicit request still dispatches; the engine's own
        # EngineCapabilityError is the honest backstop.
        capabilities = frozenset(
            {"query", "neighbors", "path", "locate", "impact", "index"}
        )
        transport = "in-process"
        version = None

    return CodeKnowledgeSystem(
        name=system_name,
        engine=engine,
        capabilities=capabilities,
        transport=transport,
        version=version,
    )
