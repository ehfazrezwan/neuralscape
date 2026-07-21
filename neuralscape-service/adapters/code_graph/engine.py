"""CodeIntelEngine protocol — the engine-selection seam for F2 native code-intel.

E1 (this slice): protocol definition + GraphifyJsonEngine wrapper preserving
today's exact behavior. E2+: NativeEngine implementing the F2 native indexer.

The protocol declares BOTH the methods that work today (query, neighbors, path)
and the F2-future methods (locate, detect_changes, semantic_layer, index,
export_snapshot). E1's GraphifyJsonEngine implements the working three and
raises NotSupported for the rest — this lands the seam with zero behavior
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class EngineCapabilityError(Exception):
    """Raised when a CodeIntelEngine implementation lacks a requested capability."""


# ── F2-future result types (stubs for E1; real impls in E2+) ────────


@dataclass(frozen=True)
class LocateHit:
    """One hybrid-retrieval hit from locate() — symbol name/signature + file:line.

    E4: enriched with attached memories via anchors (decisions/gotchas/bugfixes).
    """

    fqn: str  # fully-qualified symbol name
    kind: str  # function / class / method / module / ...
    file: str
    line: int
    signature: str  # for functions: def foo(x: int) -> str
    docstring: str  # first paragraph
    score: float
    anchor_id: str | None = None  # E4 (anchors): the stable key for memory linkage
    memories: list[dict] | None = None  # E4: [{id, content, category, ...}, ...]


@dataclass(frozen=True)
class ChangeReport:
    """Blast-radius report from detect_changes() — what moved/broke since a ref."""

    deleted_symbols: list[str]  # FQNs that vanished
    modified_symbols: list[str]  # FQNs whose signature/body changed
    added_symbols: list[str]
    affected_anchors: list[str]  # anchor_ids of memories that may need invalidation
    summary: str


@dataclass(frozen=True)
class IndexReport:
    """Result of index() — what got indexed/updated."""

    files_indexed: int
    symbols_indexed: int
    edges_indexed: int
    incremental: bool
    duration_s: float
    # Phase C (plan §3.3): engine version stamp on every IndexReport. Optional
    # with default None so native/graphify keep working; CBM stamps it from the
    # bridge's index_status.
    system_version: str | None = None
    # C3: True when the local/cloud dense leg could NOT be built (e.g. the local
    # ONNX model could not be fetched in an air-gapped deployment). The card-text
    # BM25 leg still succeeded, so locate degrades to ~0.60 h@1 rather than the
    # index job hard-failing. None ⇒ dense leg not applicable (code_embedder=off)
    # or fully succeeded.
    dense_degraded: bool | None = None


# ── The protocol ────────────────────────────────────────────────────


class CodeIntelEngine(Protocol):
    """Protocol for code-intelligence engines (graphify JSON artifact vs. native).

    Implementations:
      - GraphifyJsonEngine (E1): wraps today's query.py / semantic.py over a
        static graph.json artifact; the core three methods (query, neighbors,
        path) delegate to graphify's serve layer verbatim. F2-future methods
        raise NotSupported.
      - NativeEngine (E2+): NS-owned tree-sitter indexer + Neo4j code-label-space
        + Qdrant code_index collection, implementing the full protocol.

    The three existing MCP/REST tools route through this protocol, so swapping
    engines is transparent to clients.
    """

    def query(
        self,
        question: str,
        *,
        mode: str = "bfs",
        depth: int = 3,
        token_budget: int = 2000,
    ) -> str:
        """Search the code graph (BFS/DFS from scored seed nodes).

        Args:
            question: Natural-language question or keyword search.
            mode: "bfs" (broad context) or "dfs" (trace a specific path).
            depth: Traversal depth (1-6).
            token_budget: Max output tokens (100-20000).

        Returns:
            Text rendering of the traversal results (nodes + edges + context).
        """
        ...

    def neighbors(
        self,
        label: str,
        *,
        relation_filter: str = "",
    ) -> str:
        """Direct in/out neighbors of one code-graph node.

        Args:
            label: Node label or id to look up (e.g. "MemoryEngine").
            relation_filter: Only edges whose relation contains this substring.

        Returns:
            Text list of neighbors with relation + confidence tags.
        """
        ...

    def path(
        self,
        source: str,
        target: str,
        *,
        max_hops: int = 8,
    ) -> str:
        """Shortest connection path between two code-graph symbols.

        Args:
            source: Label of the starting symbol.
            target: Label of the target symbol.
            max_hops: Give up beyond this many hops (1-32).

        Returns:
            Text rendering of the shortest path with relations + confidence.
        """
        ...

    # ── F2-future methods (E2+) ──────────────────────────────────────

    def locate(
        self,
        query: str,
        *,
        k: int = 10,
    ) -> list[LocateHit]:
        """Hybrid code retrieval: dense embeddings + BM25 + graph degree.

        E2+ only. Serves the locate MCP tool and the plugin's grep-steering
        PreToolUse hook. Searches symbol cards (name + signature + docstring +
        first lines) in the separate `code_index` Qdrant collection.

        Args:
            query: Natural-language description or symbol name pattern.
            k: Max hits to return.

        Returns:
            Scored hits with file:line and anchor_id (for memory linkage).

        Raises:
            EngineCapabilityError: When the engine lacks locate (E1 engines).
        """
        ...

    def detect_changes(
        self,
        since: str | bytes | None = None,
    ) -> ChangeReport:
        """Real blast-radius BFS: what broke since a git ref / index snapshot.

        E5: Compares persisted Neo4j index vs fresh working-tree parse (since=None).
        E6: Extends to support snapshot-based comparison (since=bytes).

        Args:
            since: None (persisted vs fresh), bytes (snapshot vs current), or
                   str (git ref, deferred).

        Raises:
            EngineCapabilityError: When the engine lacks change detection.
        """
        ...

    def semantic_layer(self) -> list[SemanticFact]:
        """Distill the stable semantic layer: communities, hotspots, boundaries.

        E2+ NativeEngine: runs off persisted Louvain community_id + degree +
        surprise scores (computed at index time, never a per-call NetworkX
        pass). GraphifyJsonEngine already exposes this via semantic.py but
        outside the protocol — E2+ unifies it.

        Returns:
            Memory candidates (categories: module, hotspot, boundary, rationale).

        Raises:
            EngineCapabilityError: When the engine lacks semantic distillation.
        """
        ...

    def index(
        self,
        source: str,
        *,
        incremental: bool = True,
    ) -> IndexReport:
        """Index a codebase into the engine's storage (E2+ native only).

        Args:
            source: Repo path or git URL to index.
            incremental: Incremental by file content-hash (True) or full rebuild.

        Raises:
            EngineCapabilityError: GraphifyJsonEngine can't index (immutable artifact).
        """
        ...

    def export_snapshot(self) -> bytes:
        """Export a content-addressed index snapshot for index-in-CI workflows.

        E2+ / E6. The CI indexes, the deployment imports the artifact. Absorbed
        CBM capability.

        Raises:
            EngineCapabilityError: When the engine lacks snapshot export.
        """
        ...


# E1 only needs SemanticFact imported for the stub signature; the real impl is
# in semantic.py. E2+ will define it here or import from a shared module.
from adapters.code_graph.semantic import SemanticFact  # noqa: E402, F401
