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


def _resolve_cbm_engine(code_space: str, settings, *, for_index: bool):
    """Build/get a CBMEngine bound to ``code_space`` (project slug via the bridge)."""
    from adapters.code_graph.cbm_engine import CBMEngine

    ent = _cbm_cache.get(code_space)
    if ent is not None:
        engine = ent
    else:
        bridge_url = os.getenv("CBM_BRIDGE_URL", "http://cbm-bridge:8200")
        engine = CBMEngine(bridge_url=bridge_url, code_space=code_space)
        _cbm_cache[code_space] = engine

    # On the read path, bind the CBM project slug from the bridge if not yet
    # known (index runs in another process). On the index path the caller's
    # engine.index() sets the slug from the index_repository response.
    if not for_index and not getattr(engine, "project", None):
        repo_path = repo_path_for_code_space(code_space, settings)
        if repo_path:
            engine.resolve_project_from_source(repo_path)
    return engine


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

    # If the registry already holds a REAL bound system for a usable code_space
    # (not a capability placeholder), use it directly — this is the production
    # path once per-space systems are registered, and it's what test fakes rely
    # on. A placeholder is CBM's unbound engine (code_space None) or graphify's
    # "__registry_capability__" marker.
    if registered is not None:
        eng = getattr(registered, "_engine", None)
        cs = getattr(eng, "code_space", None) if eng is not None else None
        if cs and cs != "__registry_capability__":
            return registered

    engine = resolve_code_engine(
        system_name, code_space, user_id, settings, for_index=for_index
    )
    if engine is None:
        return None

    # Carry capabilities/transport/version verbatim from the registered
    # placeholder (getattr — defensive, and never a routing branch on transport;
    # DECISIONS.md cross-cutting rule: transport is declared, not branched).
    if registered is not None:
        capabilities = registered.info.capabilities
        transport = getattr(registered.info, "transport", "in-process")
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
