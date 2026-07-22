"""GraphifyLibEngine — in-process library engine for Phase F.

Per PLAN §3.2: graphify as an IN-PROCESS library inside the API/worker, not a
compose service. The engine:
  - index(source) BUILDS the graph in-process from source via graphify's extract API
  - Keeps the graph WARM per code_space (reuses _ctx_cache idiom; ~5.7MB/repo)
  - neighbors/path/query over the loaded NetworkX graph (latency ~tens of ms)
  - detect_changes/impact via graphify's affected_nodes (reverse BFS for blast_radius)
  - Reuses GraphifyJsonEngine's to_canonical/from_canonical (same graphify node naming)

Decision #2 locked: graphify = in-process library, transport="in-process". Nothing
above the seam branches on transport — this is a first-class KnowledgeSystem.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import networkx as nx
from graphify.affected import affected_nodes, resolve_seed
from graphify.build import build, edge_data
from graphify.extract import collect_files, extract
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


class GraphifyLibEngine:
    """In-process library engine: builds graphs from source using graphify's extract API.

    Implements CodeIntelEngine by importing graphify. Graphs are kept warm per
    code_space (resident in-process; ~5.7MB/repo). Query/neighbors/path delegate
    to graphify's serve layer over the loaded NetworkX graph (tens of ms latency).

    Attributes:
        code_space: Partition key (e.g. "code--owner--repo").
        source_root: Filesystem path to the indexed source (for incremental re-index).
        G: The loaded NetworkX graph (built from source).
    """

    def __init__(self, code_space: str, source_root: str | Path, G: nx.Graph | None = None):
        """Initialize with a code_space and source root.

        Args:
            code_space: Partition key (code--owner--repo).
            source_root: Filesystem path to the source directory.
            G: Pre-loaded graph (optional; otherwise built on first index()).
        """
        self.code_space = code_space
        self.source_root = Path(source_root)
        self.G = G
        self._indexed_at: float | None = None

    # ── Health / availability probe ──────────────────────────────────────

    def health(self) -> bool:
        """Availability probe: is the graphify library importable?

        CRITICAL (PLAN §3.3): a code system's registry entry represents the
        CAPABILITY (graphify library importable / per-code_space factory
        available), NOT a specific loaded graph. This engine is registered as a
        per-code_space FACTORY placeholder (G may be None) — its eligibility must
        reflect that graphify imports cleanly, not that any one graph is resident.

        The module-level ``from graphify.build import build`` at import time means
        that if this class imported at all, graphify is available. We re-confirm
        the import here so a broken/partial install reports unreachable rather
        than falsely "ok".

        Returns:
            True when graphify is importable (system eligible for routing).
        """
        try:
            import graphify.build  # noqa: F401
            import graphify.extract  # noqa: F401
            import graphify.affected  # noqa: F401
            return True
        except Exception:
            logger.warning("GraphifyLibEngine.health: graphify import failed", exc_info=True)
            return False

    # ── Canonical FQN normalization (Phase C) ───────────────────────────
    # Reuse GraphifyJsonEngine's logic (same graphify node naming; don't diverge).

    @staticmethod
    def to_canonical(raw_node_id: str) -> str:
        """Normalize graphify's node ID to canonical FQN.

        Reuses GraphifyJsonEngine.to_canonical verbatim (same graphify node naming).
        Graphify format: `<path>_<qualname>` (underscore-joined).
        Canonical format: `<module>.<qualname>` (src/lib stripped, '/' → '.').

        Args:
            raw_node_id: Graphify node ID (underscore-joined, e.g. src_click_core_Group).

        Returns:
            Canonical FQN (best-effort reconstruction from the node id).
        """
        # Graphify node IDs are underscore-joined paths: src_click_core_Group.
        # Reconstruct the dotted FQN: underscore→dot, then strip leading src/lib.
        parts = raw_node_id.split("_")

        # Strip genuine source-root directories from the start ONLY. Narrow set
        # (src/lib) so real module names like `core`/`app`/`main` survive.
        root_markers = {"src", "lib"}
        while parts and parts[0] in root_markers:
            parts.pop(0)

        canonical = ".".join(parts)
        logger.debug(f"GraphifyLib to_canonical: {raw_node_id} → {canonical}")
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
        if not self.G:
            return "No graph loaded. Run index() first."

        # Param normalization matches GraphifyJsonEngine.query exactly.
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
        if not self.G:
            return "No graph loaded. Run index() first."

        # This is adapted from GraphifyJsonEngine.neighbors (adjusted for undirected graphs).
        matches = _find_node(self.G, label)
        if not matches:
            return f"No node matching '{sanitize_label(label)}' found."
        nid = matches[0]
        rel_filter = (relation_filter or "").lower()
        lines = [f"Neighbors of {sanitize_label(self.G.nodes[nid].get('label', nid))}:"]

        # For undirected graphs, use .neighbors() instead of successors/predecessors
        if isinstance(self.G, nx.DiGraph):
            # Directed graph: show in/out separately
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
        else:
            # Undirected graph: just show neighbors (no direction)
            for nb in self.G.neighbors(nid):
                d = edge_data(self.G, nid, nb)
                rel = d.get("relation", "")
                if rel_filter and rel_filter not in rel.lower():
                    continue
                lines.append(
                    f"  -- {sanitize_label(self.G.nodes[nb].get('label', nb))} "
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
        if not self.G:
            return "No graph loaded. Run index() first."

        # This is GraphifyJsonEngine.path verbatim.
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

    # ── Phase F methods (index + impact/detect_changes) ─────────────────

    def index(
        self,
        source: str,
        *,
        incremental: bool = True,
    ) -> IndexReport:
        """Index a codebase from source using graphify's extract API.

        Builds the graph in-process via graphify.extract + graphify.build. The
        resulting NetworkX graph is kept warm (resident) in this engine instance.

        Args:
            source: Filesystem path to the source directory.
            incremental: Use graphify's caching (mtime-based; default True).

        Returns:
            IndexReport with symbols/edges counts and duration.
        """
        start = time.time()
        source_path = Path(source)
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source}")

        # Update source_root (for incremental re-index)
        self.source_root = source_path

        # Collect files to extract
        files = collect_files(source_path, follow_symlinks=False, root=source_path)
        if not files:
            logger.warning("No files found in source: %s", source)
            self.G = nx.Graph()
            return IndexReport(
                files_indexed=0,
                symbols_indexed=0,
                edges_indexed=0,
                incremental=incremental,
                duration_s=time.time() - start,
                system_version=None,  # Stamped by caller
            )

        # Extract: graphify.extract returns a dict of nodes/edges
        # cache_root controls where graphify stores its mtime cache for incrementals
        cache_root = source_path if incremental else None
        extraction = extract(files, cache_root=cache_root, parallel=True)

        # Build: merge the extraction into a NetworkX graph
        # Wrap extraction in a list (build expects list[dict])
        self.G = build([extraction], directed=False, dedup=True)
        self._indexed_at = time.time()

        duration = time.time() - start
        symbols = self.G.number_of_nodes()
        edges = self.G.number_of_edges()

        logger.info(
            "Graphify in-process index complete: %d symbols, %d edges in %.2fs (code_space=%s)",
            symbols, edges, duration, self.code_space,
        )

        return IndexReport(
            files_indexed=len(files),
            symbols_indexed=symbols,
            edges_indexed=edges,
            incremental=incremental,
            duration_s=duration,
            system_version=None,  # Stamped by caller
        )

    def detect_changes(
        self,
        since: str | bytes | None = None,
    ) -> ChangeReport:
        """Blast-radius via graphify's affected_nodes (reverse BFS).

        For git-less repos, this powers blast_radius: "what is affected if I
        change X?" The `since` param is overloaded: a str is treated as a seed
        symbol (or query) and resolved against the graph; bytes/None (git-based
        diff) is not supported by the in-process library engine.

        The affected node IDs are canonicalized (via to_canonical) so the report
        is engine-agnostic and aligns with anchor keys.

        Args:
            since: str — seed symbol/query for blast-radius analysis.

        Returns:
            ChangeReport with the blast-radius symbols under ``modified_symbols``
            (they may need attention if the seed changes). ``deleted_symbols`` /
            ``added_symbols`` / ``affected_anchors`` stay empty (no git diff);
            ``summary`` describes the blast radius. An unresolved seed yields an
            empty report with an explanatory summary (not an exception).

        Raises:
            EngineCapabilityError: git-based diff (since=None/bytes) — use a
                git-aware engine (e.g. code-native).
        """
        if not isinstance(since, str):
            # Git-based diff (since=None or bytes) not supported for in-process engine
            raise EngineCapabilityError(
                "detect_changes() with git-based diff (since=None/bytes) is not supported "
                "on GraphifyLibEngine. Pass a seed symbol (str) for blast-radius analysis, "
                "or use a git-aware engine (e.g. code-native)."
            )

        if not self.G:
            return ChangeReport(
                deleted_symbols=[],
                modified_symbols=[],
                added_symbols=[],
                affected_anchors=[],
                summary="No graph loaded; run index() first.",
            )

        # Resolve the seed (accepts a node id or a natural-language-ish query).
        seed_node = resolve_seed(self.G, since)
        if not seed_node:
            return ChangeReport(
                deleted_symbols=[],
                modified_symbols=[],
                added_symbols=[],
                affected_anchors=[],
                summary=f"No node resolved for seed {since!r}; empty blast radius.",
            )

        # Reverse-BFS blast radius. AffectedHit carries node_id/depth/via_relation.
        hits = affected_nodes(self.G, seed=seed_node, depth=2)
        affected: list[str] = []
        for h in hits:
            if not h.node_id:
                continue
            try:
                canonical = self.to_canonical(h.node_id)
            except Exception:
                continue
            if canonical:
                affected.append(canonical)

        logger.debug("Blast radius for %s (node %s): %d affected symbols",
                     since, seed_node, len(affected))
        return ChangeReport(
            deleted_symbols=[],
            modified_symbols=affected,  # blast-radius: symbols that may be affected
            added_symbols=[],
            affected_anchors=[],
            summary=f"Blast radius for {since!r}: {len(affected)} affected symbol(s).",
        )

    def teardown(self) -> dict:
        """R-C: drop the in-process graph for a true cold rebuild.

        Clears the NetworkX graph + build timestamp; the next read lazily
        rebuilds from source (``resolve_code_engine`` builds when ``G is None``).
        The caller (``code_dispatch.teardown_code_space``) additionally evicts
        this engine's ``_ctx_cache`` entry so the next bind is a fresh instance.
        GraphifyLibEngine keeps nothing on disk and never holds the memory graph,
        so there is nothing else to reset — the moat is untouched.

        Idempotent: tearing down an already-empty engine drops 0.
        """
        n = self.G.number_of_nodes() if self.G is not None else 0
        self.G = None
        self._indexed_at = None
        logger.info(
            "GraphifyLib teardown code_space=%s: dropped %d-node in-process graph",
            self.code_space, n,
        )
        return {"nodes_dropped": n}

    def get_symbol_inventory(self) -> set[str]:
        """Get current symbol inventory (canonical FQNs) for liveness tracking.

        Phase E: Used to detect deleted/changed symbols after reindex.
        Returns canonical FQNs (via to_canonical) so the liveness diff is
        engine-agnostic.

        Returns:
            Set of canonical FQNs currently indexed.
        """
        if not self.G:
            return set()

        # Extract all node IDs (raw FQNs) and canonicalize them. Skip falsy node
        # ids and falsy canonicals so no "repo::None"/"repo::" anchor key is built.
        canonical_fqns = set()
        for node_id in self.G.nodes():
            if not node_id:
                continue
            try:
                canonical = self.to_canonical(node_id)
            except Exception:
                # Skip malformed node IDs
                continue
            if canonical:
                canonical_fqns.add(canonical)

        logger.debug("GraphifyLib symbol inventory: %d canonical FQNs", len(canonical_fqns))
        return canonical_fqns

    # ── Not supported (honest N/A) ───────────────────────────────────────

    def locate(
        self,
        query: str,
        *,
        k: int = 10,
    ) -> list[LocateHit]:
        """Not supported — no dense-embedding code_index (use query() for structure search)."""
        raise EngineCapabilityError(
            "locate() is not supported on GraphifyLibEngine (no dense-embedding code_index). "
            "Use query() for structure-based search or CBM for semantic locate."
        )

    def semantic_layer(self) -> list[SemanticFact]:
        """Not supported via protocol yet (ingest-time distillation lives outside protocol)."""
        raise EngineCapabilityError(
            "semantic_layer() not yet exposed via the protocol (Phase F). "
            "Use the existing ingest-time distillation (semantic.py:extract_semantic_layer)."
        )

    def export_snapshot(self) -> bytes:
        """Export graph.json as a snapshot (for index-in-CI workflows).

        Returns:
            JSON bytes (NetworkX graph serialized as graph.json).
        """
        if not self.G:
            raise ValueError("No graph loaded. Run index() first.")

        import json
        from networkx.readwrite import json_graph

        # Serialize the graph to JSON (same format as graphify's graph.json artifacts)
        graph_data = json_graph.node_link_data(self.G)
        return json.dumps(graph_data, indent=2).encode("utf-8")
