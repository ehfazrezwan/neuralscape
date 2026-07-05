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

    E1: always returns GraphifyJsonEngine for .json artifact paths (the only kind
    that exists today). E2+ will detect repo:<name> refs and return NativeEngine.

    The engine is cached per resolved path until (mtime, size) changes — same
    hot-reload discipline as the old load_graph_cached.

    Args:
        graph_id: Artifact id or None (uses default path).
        user_id: Owner-scoped resolution.
        settings: Config for default path.

    Returns:
        A CodeIntelEngine (today: GraphifyJsonEngine).

    Raises:
        CodeGraphNotConfigured: No graph_id and no default configured.
        CodeGraphError: graph_id doesn't resolve or isn't a .json artifact.
    """
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
    """Search the code graph (BFS/DFS from scored seed nodes) — routes through engine."""
    engine = get_engine(graph_id, user_id, settings)
    return engine.query(question, mode=mode, depth=depth, token_budget=token_budget)


def get_code_neighbors(
    label: str,
    *,
    user_id: str,
    settings,
    graph_id: str | None = None,
    relation_filter: str = "",
) -> str:
    """Direct in/out neighbors of a node — routes through engine."""
    engine = get_engine(graph_id, user_id, settings)
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
