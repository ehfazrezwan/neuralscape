"""NS-surface code-structure queries — thin delegations to the graphifyy library.

**The interaction interface is ALWAYS Neuralscape** (roadmap F2 constraint):
agents call NS's own ``query_code_graph`` / ``get_code_neighbors`` /
``code_path`` MCP tools (or the matching ``/v1/code-graph/*`` REST routes),
and NS resolves them against a ``graph.json`` via graphifyy's query layer.
Clients are never pointed at Graphify's own MCP server; the ``source_ref``
retrieval handles stamped on code-graph memories point back HERE.

Graph resolution (no arbitrary path reads — the API must not become a file
oracle):

- ``graph_id`` — the artifact id of an ingested graph.json bundle, resolved
  owner-scoped via :func:`ingest.storage.find_artifact` (one user cannot read
  another's graph by guessing an id);
- otherwise the deployment-configured ``settings.code_graph_json_path``.

Query semantics reuse graphify's own scoring/traversal/rendering
(``_query_graph_text`` / ``_find_node`` / ``_score_nodes``). The neighbor and
shortest-path renderers are small re-implementations of graphify's MCP tool
bodies — those live as closures inside ``serve._build_server`` and aren't
importable, but every non-trivial step (node search, edge access, label
sanitisation, path-finding) still calls the library. Kept behaviorally aligned
with graphify's ``get_neighbors`` / ``shortest_path`` tools.

This module imports ``graphify`` at module level — import it only behind
:func:`adapters.code_graph.code_graph_available`.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import networkx as nx
from graphify.build import edge_data
from graphify.security import sanitize_label
from graphify.serve import _find_node, _query_graph_text, _score_nodes

from adapters.code_graph.semantic import CodeGraphError, load_code_graph

logger = logging.getLogger(__name__)


class CodeGraphNotConfigured(CodeGraphError):
    """No graph.json to query: no graph_id given and no default path configured."""


# ── Graph resolution + cache ────────────────────────────────────────

# path -> {"key": (mtime_ns, size), "G": nx.Graph} — mtime+size hot-reload,
# mirroring graphify's own serve-layer cache.
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


def load_graph_cached(path: str) -> nx.Graph:
    """Load graph.json via the library, cached per path until (mtime, size) changes."""
    try:
        s = Path(path).stat()
    except FileNotFoundError:
        raise CodeGraphError(f"graph.json not found: {path}") from None
    key = (s.st_mtime_ns, s.st_size)
    ent = _ctx_cache.get(path)
    if ent is not None and ent["key"] == key:
        return ent["G"]
    with _ctx_lock:
        ent = _ctx_cache.get(path)
        if ent is not None and ent["key"] == key:
            return ent["G"]  # another thread built it
        G = load_code_graph(path)
        _ctx_cache[path] = {"key": key, "G": G}
        return G


def _resolve_and_load(graph_id: str | None, user_id: str, settings) -> nx.Graph:
    return load_graph_cached(resolve_graph_path(graph_id, user_id, settings))


# ── The three delegation queries ────────────────────────────────────


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
    """Search the code graph (BFS/DFS from scored seed nodes) — graphify's query_graph."""
    G = _resolve_and_load(graph_id, user_id, settings)
    mode = mode if mode in ("bfs", "dfs") else "bfs"
    depth = max(1, min(int(depth), 6))
    token_budget = max(100, min(int(token_budget), 20_000))
    return _query_graph_text(G, question, mode=mode, depth=depth, token_budget=token_budget)


def get_code_neighbors(
    label: str,
    *,
    user_id: str,
    settings,
    graph_id: str | None = None,
    relation_filter: str = "",
) -> str:
    """Direct in/out neighbors of a node — graphify's get_neighbors behavior."""
    G = _resolve_and_load(graph_id, user_id, settings)
    matches = _find_node(G, label)
    if not matches:
        return f"No node matching '{sanitize_label(label)}' found."
    nid = matches[0]
    rel_filter = (relation_filter or "").lower()
    lines = [f"Neighbors of {sanitize_label(G.nodes[nid].get('label', nid))}:"]
    for nb in G.successors(nid):
        d = edge_data(G, nid, nb)
        rel = d.get("relation", "")
        if rel_filter and rel_filter not in rel.lower():
            continue
        lines.append(
            f"  --> {sanitize_label(G.nodes[nb].get('label', nb))} "
            f"[{sanitize_label(str(rel))}] [{sanitize_label(str(d.get('confidence', '')))}]"
        )
    for nb in G.predecessors(nid):
        d = edge_data(G, nb, nid)
        rel = d.get("relation", "")
        if rel_filter and rel_filter not in rel.lower():
            continue
        lines.append(
            f"  <-- {sanitize_label(G.nodes[nb].get('label', nb))} "
            f"[{sanitize_label(str(rel))}] [{sanitize_label(str(d.get('confidence', '')))}]"
        )
    if len(lines) == 1:
        lines.append("  (no neighbors matching the filter)")
    return "\n".join(lines)


def code_path(
    source: str,
    target: str,
    *,
    user_id: str,
    settings,
    graph_id: str | None = None,
    max_hops: int = 8,
) -> str:
    """Shortest path between two symbols — graphify's shortest_path behavior."""
    G = _resolve_and_load(graph_id, user_id, settings)
    src_scored = _score_nodes(G, [t.lower() for t in source.split()])
    tgt_scored = _score_nodes(G, [t.lower() for t in target.split()])
    if not src_scored:
        return f"No node matching source '{sanitize_label(source)}' found."
    if not tgt_scored:
        return f"No node matching target '{sanitize_label(target)}' found."
    src_nid, tgt_nid = src_scored[0][1], tgt_scored[0][1]
    if src_nid == tgt_nid:
        return (
            f"'{sanitize_label(source)}' and '{sanitize_label(target)}' both resolved "
            f"to the same node '{sanitize_label(src_nid)}'. Use a more specific label."
        )
    try:
        path_nodes = nx.shortest_path(G.to_undirected(as_view=True), src_nid, tgt_nid)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return (
            f"No path found between "
            f"'{sanitize_label(G.nodes[src_nid].get('label', src_nid))}' and "
            f"'{sanitize_label(G.nodes[tgt_nid].get('label', tgt_nid))}'."
        )
    hops = len(path_nodes) - 1
    max_hops = max(1, min(int(max_hops), 32))
    if hops > max_hops:
        return f"Path exceeds max_hops={max_hops} ({hops} hops found)."
    segments: list[str] = []
    for i in range(hops):
        u, v = path_nodes[i], path_nodes[i + 1]
        if G.has_edge(u, v):
            edata, forward = edge_data(G, u, v), True
        else:
            edata, forward = edge_data(G, v, u), False
        rel = sanitize_label(str(edata.get("relation", "")))
        conf = edata.get("confidence", "")
        conf_str = f" [{sanitize_label(str(conf))}]" if conf else ""
        if i == 0:
            segments.append(sanitize_label(G.nodes[u].get("label", u)))
        if forward:
            segments.append(f"--{rel}{conf_str}--> {sanitize_label(G.nodes[v].get('label', v))}")
        else:
            segments.append(f"<--{rel}{conf_str}-- {sanitize_label(G.nodes[v].get('label', v))}")
    return f"Shortest path ({hops} hops):\n  " + " ".join(segments)
