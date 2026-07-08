"""NS-surface code-structure queries — route through the CodeIntelEngine protocol.

E1 refactor: the three public tools (query_code_graph, get_code_neighbors,
code_path) now route through the CodeIntelEngine protocol via get_engine(),
which selects GraphifyJsonEngine for .json artifact graph_ids (the only kind
that exists today). Zero behavior change: GraphifyJsonEngine wraps the exact
same graphify library calls that the old direct-call code path used.

**The interaction interface is ALWAYS Neuralscape** (roadmap F2 constraint):
agents call NS's own ``query_code_graph`` / ``get_code_neighbors`` /
``code_path`` MCP tools (or the matching ``/v1/code-graph/*`` REST routes),
and NS resolves them against a ``graph.json`` via the engine. Clients are
never pointed at Graphify's own MCP server; the ``source_ref`` retrieval
handles stamped on code-graph memories point back HERE.

Graph resolution (no arbitrary path reads — the API must not become a file
oracle):

- ``graph_id`` — the artifact id of an ingested graph.json bundle, resolved
  owner-scoped via :func:`ingest.storage.find_artifact` (one user cannot read
  another's graph by guessing an id);
- otherwise the deployment-configured ``settings.code_graph_json_path``.

This module imports ``graphify`` at module level — import it only behind
:func:`adapters.code_graph.code_graph_available`.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import networkx as nx

from adapters.code_graph.engine import CodeIntelEngine
from adapters.code_graph.graphify_engine import GraphifyJsonEngine
from adapters.code_graph.semantic import CodeGraphError, load_code_graph

logger = logging.getLogger(__name__)


class CodeGraphNotConfigured(CodeGraphError):
    """No graph.json to query: no graph_id given and no default path configured."""


# ── Graph resolution + engine cache ────────────────────────────────

# E1: path -> {"key": (mtime_ns, size), "engine": CodeIntelEngine} — mtime+size
# hot-reload, mirroring graphify's own serve-layer cache. E2+ will extend this
# to cache NativeEngine instances keyed by repo:<name> refs.
_ctx_lock = threading.Lock()
_ctx_cache: dict[str, dict] = {}


def resolve_graph_path(graph_id: str | None, user_id: str, settings) -> str:
    """Map (graph_id | default setting) → a concrete graph.json path.

    Raises:
        CodeGraphNotConfigured: neither a graph_id nor a configured default.
        CodeGraphError: graph_id doesn't resolve to a .json artifact of this user.
    """
    if graph_id:
        from ingest.storage import find_artifact

        found = find_artifact(graph_id, user_id, settings)
        if not found:
            raise CodeGraphError(
                f"No ingested code graph with graph_id {graph_id!r} for this user. "
                f"Ingest a Graphify bundle (graph.json + GRAPH_REPORT.md) first."
            )
        abs_path, filename = found
        if not filename.endswith(".json"):
            raise CodeGraphError(
                f"Artifact {graph_id!r} is not a graph.json (got {filename!r})."
            )
        return abs_path
    default = (settings.code_graph_json_path or "").strip()
    if not default:
        raise CodeGraphNotConfigured(
            "No graph_id given and no default code graph configured "
            "(set CODE_GRAPH_JSON_PATH or pass the graph_id from an ingested bundle)."
        )
    return str(Path(os.path.expanduser(default)))


def get_engine(graph_id: str | None, user_id: str, settings) -> CodeIntelEngine:
    """Engine-selection factory: returns the right CodeIntelEngine for the graph ref.

    E2: detects repo:<name> refs and returns NativeEngine; .json artifact paths
    still return GraphifyJsonEngine. Engines are cached per ref (repo: by name,
    .json by mtime+size).

    Args:
        graph_id: Artifact id, repo:<name> ref, or None (uses default path).
        user_id: Owner-scoped resolution.
        settings: Config for default path.

    Returns:
        A CodeIntelEngine (GraphifyJsonEngine or NativeEngine).

    Raises:
        CodeGraphNotConfigured: No graph_id and no default configured.
        CodeGraphError: graph_id doesn't resolve or repo path not found.
    """
    # E2: detect repo:<name> refs
    if graph_id and graph_id.startswith("repo:"):
        repo_name = graph_id.removeprefix("repo:")
        return _get_native_engine(repo_name, user_id, settings)

    # E1 path: .json artifacts
    path = resolve_graph_path(graph_id, user_id, settings)
    try:
        s = Path(path).stat()
    except FileNotFoundError:
        raise CodeGraphError(f"graph.json not found: {path}") from None
    key = (s.st_mtime_ns, s.st_size)
    ent = _ctx_cache.get(path)
    if ent is not None and ent["key"] == key:
        return ent["engine"]
    with _ctx_lock:
        ent = _ctx_cache.get(path)
        if ent is not None and ent["key"] == key:
            return ent["engine"]  # another thread built it
        # Load the graph and wrap it in GraphifyJsonEngine.
        G = load_code_graph(path)
        engine = GraphifyJsonEngine(G, path)
        _ctx_cache[path] = {"key": key, "engine": engine}
        return engine


def _get_native_engine(repo_name: str, user_id: str, settings) -> CodeIntelEngine:
    """Get or create a cached NativeEngine for a repo:<name> ref (E2).

    The repo path is resolved from settings.code_repos[repo_name] (a dict mapping
    repo names to filesystem paths). Engines are cached by code_space key.

    Raises:
        CodeGraphError: repo_name not in configured repos or path doesn't exist.
    """
    from adapters.code_graph.native_engine import NativeEngine

    # Resolve repo path from settings
    repos = getattr(settings, "code_repos", {})
    if not repos:
        raise CodeGraphError(
            "No code_repos configured. Set CODE_REPOS env var (JSON dict) "
            "mapping repo names to filesystem paths."
        )
    repo_path = repos.get(repo_name)
    if not repo_path:
        raise CodeGraphError(
            f"No repo configured with name {repo_name!r}. "
            f"Available repos: {', '.join(repos.keys())}"
        )
    repo_path = Path(os.path.expanduser(repo_path))
    if not repo_path.is_dir():
        raise CodeGraphError(f"Repo path does not exist: {repo_path}")

    # Build the code_space partition key
    code_space = f"code--{user_id}--{repo_name}"

    # Check cache (keyed by code_space, no mtime check — NativeEngine reads from Neo4j)
    cache_key = f"native:{code_space}"
    ent = _ctx_cache.get(cache_key)
    if ent is not None:
        return ent["engine"]

    with _ctx_lock:
        ent = _ctx_cache.get(cache_key)
        if ent is not None:
            return ent["engine"]

        # Get the Graphiti bridge from the shared MemoryService
        from memory_service import get_shared_service
        service = get_shared_service()
        service._get_memory()  # ensure bridge is initialized
        bridge = service._bridge
        if bridge is None:
            raise CodeGraphError("Graphiti bridge not initialized (Neo4j unavailable)")

        # Create NativeEngine. The Neo4j driver lives on the Graphiti client
        # (service._graphiti.driver); the _AsyncBridge only carries the loop.
        driver = getattr(getattr(service, "_graphiti", None), "driver", None)
        engine = NativeEngine(
            repo_path=str(repo_path),
            code_space=code_space,
            bridge=bridge,
            settings=settings,
            driver=driver,
        )
        _ctx_cache[cache_key] = {"engine": engine}
        return engine


# ── The three delegation queries (now route through the protocol) ──


def query_code_graph(
    question: str,
    *,
    user_id: str,
    settings,
    graph_id: str | None = None,
    mode: str = "bfs",
    depth: int = 3,
    token_budget: int = 2000,
) -> str:
    """Search the code graph (BFS/DFS from scored seed nodes) — routes through engine.

    E4: NativeEngine enriches results with attached memories (respects read scope).
    """
    engine = get_engine(graph_id, user_id, settings)
    # Try passing user_id for E4 enrichment; GraphifyJsonEngine ignores extra kwargs
    try:
        return engine.query(question, mode=mode, depth=depth, token_budget=token_budget, user_id=user_id)
    except TypeError:
        # Fallback for engines that don't accept user_id (backward compat)
        return engine.query(question, mode=mode, depth=depth, token_budget=token_budget)


def get_code_neighbors(
    label: str,
    *,
    user_id: str,
    settings,
    graph_id: str | None = None,
    relation_filter: str = "",
) -> str:
    """Direct in/out neighbors of a node — routes through engine.

    E4: NativeEngine enriches results with attached memories.
    """
    engine = get_engine(graph_id, user_id, settings)
    try:
        return engine.neighbors(label, relation_filter=relation_filter, user_id=user_id)
    except TypeError:
        return engine.neighbors(label, relation_filter=relation_filter)


def code_path(
    source: str,
    target: str,
    *,
    user_id: str,
    settings,
    graph_id: str | None = None,
    max_hops: int = 8,
) -> str:
    """Shortest path between two symbols — routes through engine."""
    engine = get_engine(graph_id, user_id, settings)
    return engine.path(source, target, max_hops=max_hops)


def locate_symbols(
    query: str,
    *,
    user_id: str,
    settings,
    graph_id: str | None = None,
    k: int = 10,
):
    """Hybrid code retrieval — routes through engine (E3: NativeEngine only).

    E4: Results enriched with attached memories (respects read scope).

    Args:
        query: Natural-language description or symbol name pattern.
        user_id: Owner-scoped resolution + memory read-scope.
        settings: Config for default path / repo resolution.
        graph_id: Artifact id, repo:<name> ref, or None (uses default).
        k: Max hits to return.

    Returns:
        List of LocateHit dataclasses (E4: with memories field populated).

    Raises:
        EngineCapabilityError: When the engine lacks locate (GraphifyJsonEngine).
        CodeGraphError: graph_id doesn't resolve or repo not configured.
    """
    from adapters.code_graph.engine import EngineCapabilityError

    k = max(1, min(int(k), 50))  # clamp to [1, 50]
    engine = get_engine(graph_id, user_id, settings)
    return engine.locate(query, k=k, user_id=user_id)


def code_impact(
    symbol: str,
    *,
    user_id: str,
    settings,
    graph_id: str | None = None,
    max_hops: int = 4,
) -> str:
    """Blast radius from a symbol — routes through engine (E7: NativeEngine only).

    BFS over CALLS/IMPORTS edges to find all symbols affected by changes to the
    given symbol. Returns a text summary (file:line format).

    Args:
        symbol: FQN or partial match of the epicenter symbol.
        user_id: Owner-scoped resolution.
        settings: Config for default path / repo resolution.
        graph_id: Artifact id, repo:<name> ref, or None (uses default).
        max_hops: Maximum BFS depth (1-16, default 4).

    Returns:
        Text summary of affected symbols.

    Raises:
        EngineCapabilityError: When the engine lacks blast_radius (GraphifyJsonEngine).
        CodeGraphError: graph_id doesn't resolve or repo not configured.
    """
    from adapters.code_graph.engine import EngineCapabilityError

    max_hops = max(1, min(int(max_hops), 16))  # clamp to [1, 16]
    engine = get_engine(graph_id, user_id, settings)
    # blast_radius is native-only. GraphifyJsonEngine (.json artifacts) has no
    # such method — surface a clean EngineCapabilityError (→ 501) rather than
    # letting a bare AttributeError bubble up as an HTTP 500.
    impact = getattr(engine, "blast_radius", None)
    if not callable(impact):
        raise EngineCapabilityError(
            "code_impact/blast_radius requires the native code-intel engine "
            "(repo:<name> refs). GraphifyJsonEngine operates on static graph.json "
            "artifacts and has no blast-radius traversal. Index the repo with the "
            "native engine (native_index_cli) to use code_impact."
        )
    return impact(symbol, max_hops=max_hops)
