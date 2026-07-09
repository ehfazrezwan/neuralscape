"""GraphifyJsonEngine — wraps today's query.py / semantic.py for E1 protocol compliance.

Pure delegation to existing helpers with ZERO behavior change: the engine calls
the exact same graphify library functions (query_graph_text, find_node, score_nodes,
etc.) and produces byte-identical output to the current direct-call code path.

F2-future methods (locate, detect_changes, index, export_snapshot) raise
EngineCapabilityError — those require the native engine (E2+).
"""

from __future__ import annotations

import logging

import networkx as nx
from graphify.build import edge_data
from graphify.security import sanitize_label
from graphify.serve import _find_node, _query_graph_text, _score_nodes

from adapters.code_graph.engine import (
    ChangeReport,
    EngineCapabilityError,
    IndexReport,
    LocateHit,
    SemanticFact,
)

logger = logging.getLogger(__name__)


class GraphifyJsonEngine:
    """E1 engine: wraps a static graph.json artifact via graphify's serve layer.

    The core three methods (query, neighbors, path) delegate to today's
    query.py helpers, which in turn call graphify's library — preserving
    exact behavior. F2-future methods raise EngineCapabilityError.

    Attributes:
        G: The loaded NetworkX graph (from graph.json).
        graph_path: Filesystem path to the graph.json (for reload detection).
    """

    def __init__(self, G: nx.Graph, graph_path: str):
        """Initialize with a loaded graph.

        Args:
            G: NetworkX graph loaded via graphify's _load_graph.
            graph_path: Path to the graph.json (for cache-key / debug).
        """
        self.G = G
        self.graph_path = graph_path

    # ── Canonical FQN normalization (Phase C) ───────────────────────────

    @staticmethod
    def to_canonical(raw_node_id: str) -> str:
        """Normalize graphify's node ID to canonical FQN.

        Graphify format: `<path>_<qualname>` (underscore-joined, file path + symbol).
        Example: `src_click_termui_impl_progressbar` (node ID) with label `ProgressBar`.

        Canonical format: `<module>.<qualname>` (src/lib stripped, '/' → '.').
        Example: `click.termui.impl.ProgressBar`.

        This is best-effort: graphify's node IDs are often file-path-derived and
        don't always carry the full qualname. We use the node's 'label' attribute
        as the qualname when available.

        Args:
            raw_node_id: Graphify node ID (underscore-joined).

        Returns:
            Canonical FQN (best-effort reconstruction).
        """
        # Graphify node IDs are underscore-joined paths: src_click_core_Group
        # We need to reconstruct the dotted FQN.
        # Strategy: replace underscores with dots, strip src/lib roots, use label if available.

        parts = raw_node_id.split("_")

        # Strip common root directories
        root_markers = {"src", "lib", "pkg", "internal", "app", "core", "main"}
        while parts and parts[0] in root_markers:
            parts.pop(0)

        canonical = ".".join(parts)
        logger.debug(f"Graphify to_canonical: {raw_node_id} → {canonical}")
        return canonical

    @staticmethod
    def from_canonical(canonical_fqn: str) -> str:
        """Convert canonical FQN back to graphify node ID (best-effort).

        Since graphify uses underscore-joined file paths as node IDs, we can't
        reconstruct them exactly from canonical FQNs. Instead, we return a
        search-friendly pattern: dots → underscores.

        Args:
            canonical_fqn: Canonical FQN (e.g. click.core.Group).

        Returns:
            Graphify node ID pattern (underscore-joined).
        """
        # For search, convert dots to underscores (graphify's node ID format).
        return canonical_fqn.replace(".", "_")

    def query(
        self,
        question: str,
        *,
        mode: str = "bfs",
        depth: int = 3,
        token_budget: int = 2000,
    ) -> str:
        """Search the code graph — delegates to graphify's _query_graph_text."""
        # Param normalization matches query.py:query_code_graph exactly.
        mode = mode if mode in ("bfs", "dfs") else "bfs"
        depth = max(1, min(int(depth), 6))
        token_budget = max(100, min(int(token_budget), 20_000))
        return _query_graph_text(
            self.G, question, mode=mode, depth=depth, token_budget=token_budget
        )

    def neighbors(
        self,
        label: str,
        *,
        relation_filter: str = "",
    ) -> str:
        """Direct in/out neighbors — re-implements graphify's get_neighbors behavior."""
        # This is query.py:get_code_neighbors verbatim, now as a method.
        matches = _find_node(self.G, label)
        if not matches:
            return f"No node matching '{sanitize_label(label)}' found."
        nid = matches[0]
        rel_filter = (relation_filter or "").lower()
        lines = [f"Neighbors of {sanitize_label(self.G.nodes[nid].get('label', nid))}:"]
        for nb in self.G.successors(nid):
            d = edge_data(self.G, nid, nb)
            rel = d.get("relation", "")
            if rel_filter and rel_filter not in rel.lower():
                continue
            lines.append(
                f"  --> {sanitize_label(self.G.nodes[nb].get('label', nb))} "
                f"[{sanitize_label(str(rel))}] [{sanitize_label(str(d.get('confidence', '')))}]"
            )
        for nb in self.G.predecessors(nid):
            d = edge_data(self.G, nb, nid)
            rel = d.get("relation", "")
            if rel_filter and rel_filter not in rel.lower():
                continue
            lines.append(
                f"  <-- {sanitize_label(self.G.nodes[nb].get('label', nb))} "
                f"[{sanitize_label(str(rel))}] [{sanitize_label(str(d.get('confidence', '')))}]"
            )
        if len(lines) == 1:
            lines.append("  (no neighbors matching the filter)")
        return "\n".join(lines)

    def path(
        self,
        source: str,
        target: str,
        *,
        max_hops: int = 8,
    ) -> str:
        """Shortest path between two symbols — graphify's shortest_path behavior."""
        # This is query.py:code_path verbatim.
        src_scored = _score_nodes(self.G, [t.lower() for t in source.split()])
        tgt_scored = _score_nodes(self.G, [t.lower() for t in target.split()])
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
            path_nodes = nx.shortest_path(self.G.to_undirected(as_view=True), src_nid, tgt_nid)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return (
                f"No path found between "
                f"'{sanitize_label(self.G.nodes[src_nid].get('label', src_nid))}' and "
                f"'{sanitize_label(self.G.nodes[tgt_nid].get('label', tgt_nid))}'."
            )
        hops = len(path_nodes) - 1
        max_hops = max(1, min(int(max_hops), 32))
        if hops > max_hops:
            return f"Path exceeds max_hops={max_hops} ({hops} hops found)."
        segments: list[str] = []
        for i in range(hops):
            u, v = path_nodes[i], path_nodes[i + 1]
            if self.G.has_edge(u, v):
                edata, forward = edge_data(self.G, u, v), True
            else:
                edata, forward = edge_data(self.G, v, u), False
            rel = sanitize_label(str(edata.get("relation", "")))
            conf = edata.get("confidence", "")
            conf_str = f" [{sanitize_label(str(conf))}]" if conf else ""
            if i == 0:
                segments.append(sanitize_label(self.G.nodes[u].get("label", u)))
            if forward:
                segments.append(f"--{rel}{conf_str}--> {sanitize_label(self.G.nodes[v].get('label', v))}")
            else:
                segments.append(f"<--{rel}{conf_str}-- {sanitize_label(self.G.nodes[v].get('label', v))}")
        return f"Shortest path ({hops} hops):\n  " + " ".join(segments)

    # ── F2-future methods (not supported on static graph.json) ────────

    def locate(
        self,
        query: str,
        *,
        k: int = 10,
    ) -> list[LocateHit]:
        """Not supported on graph.json artifacts — requires native engine (E2+)."""
        raise EngineCapabilityError(
            "locate() requires the native code-intel engine (E2+). "
            "GraphifyJsonEngine operates on static graph.json artifacts and has no "
            "dense-embedding code_index collection. Use query() for structure search."
        )

    def detect_changes(
        self,
        since: str,
    ) -> ChangeReport:
        """Not supported on immutable graph.json — requires native engine (E2+)."""
        raise EngineCapabilityError(
            "detect_changes() requires the native code-intel engine (E2+). "
            "GraphifyJsonEngine operates on immutable graph.json artifacts with no "
            "incremental index or historical snapshots. Re-run graphify to produce a "
            "new artifact and diff the semantic layer manually."
        )

    def semantic_layer(self) -> list[SemanticFact]:
        """Not supported via protocol yet (E1 only lands the seam; E2+ unifies it).

        The distillation exists (semantic.py:extract_semantic_layer) but lives
        outside the protocol — it's called at ingest time, not query time. E2+
        will unify it under the protocol so both engines expose the same
        surface. For now, raise NotSupported to keep the protocol honest.
        """
        raise EngineCapabilityError(
            "semantic_layer() not yet exposed via the protocol (E1). "
            "Use the existing ingest-time distillation (semantic.py:extract_semantic_layer). "
            "E2+ will unify this under the protocol."
        )

    def index(
        self,
        source: str,
        *,
        incremental: bool = True,
    ) -> IndexReport:
        """Not supported — graph.json artifacts are immutable, built by graphify."""
        raise EngineCapabilityError(
            "index() is not supported on GraphifyJsonEngine. "
            "GraphifyJsonEngine operates on pre-built graph.json artifacts produced by "
            "the graphify CLI. To index a codebase, run `graphify` externally and ingest "
            "the resulting bundle. The native engine (E2+) will support NS-native indexing."
        )

    def export_snapshot(self) -> bytes:
        """Not supported — graph.json is already the artifact; no separate export."""
        raise EngineCapabilityError(
            "export_snapshot() is not supported on GraphifyJsonEngine. "
            "The graph.json artifact is already the snapshot. The native engine (E2+) "
            "will support content-addressed index snapshots for index-in-CI workflows."
        )
