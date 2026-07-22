"""KnowledgeSystem protocol — the registry-level seam for pluggable knowledge backends.

A :class:`KnowledgeSystem` wraps a queryable backend (NS memory, CBM code intel,
graphify, frozen native) with uniform capability declaration, health checking,
and recall/index operations. This is the router's view: engines are selected
per-request, systems declare what they can do, and the registry gates
eligibility by health + declared capabilities.

**Relationship to other seams** (see ICE_V2_KNOWLEDGE_SYSTEMS_PLAN.md §1):

- :class:`KnowledgeAdapter` (``adapters/base.py``) shapes what enters the
  **base** NS store (taxonomy, chunker, extractor, Graphiti ontology). It
  parameterizes the base system's ingestion, not the external-system seam.
- :class:`CodeIntelEngine` (``adapters/code_graph/engine.py``) is the
  per-backend driver protocol for code-domain operations (query/neighbors/path/
  locate/detect_changes/index/export). Engines are wrapped by CodeKnowledgeSystem;
  they never appear in routing directly.
- **KnowledgeSystem** is what the router/fusion layer sees: one registry entry
  per backend, health-gated eligibility, capability-driven routing.

**``remember()`` is deliberately ABSENT** (PLAN §0.1): external code systems are
derived indexes (rebuilt from source), never authoritative. Writes always go to
the base NS store; what routing governs is (a) recall/query-side system selection
and (b) index triggering for code systems. Memories that concern a code symbol
use **anchors**, which live in NS Neo4j, not in the external system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field


# ── Info & Health types ────────────────────────────────────────────


@dataclass(frozen=True)
class KnowledgeSystemInfo:
    """Static metadata about a KnowledgeSystem.

    Registered once at import; introspectable via the registry for routing,
    health aggregation, and bench attribution.
    """

    name: str  # Registry key: "ns-memory", "code-cbm", "code-graphify", "code-native"
    kind: str  # Broad class: "base" (NS memory) | "code" (code intel) | (future: "docs", "traces", ...)
    capabilities: frozenset[str]  # Op-class names it serves, e.g.
    # base: {"recall", "timeline", "cards", "ask", "graph_search"}
    # code: {"symbol_lookup", "neighbors", "path", "locate", "impact",
    #        "semantic_chunks", "index", "snapshot"}
    transport: str  # Implementation note: "in-process" | "mcp-stdio-bridge" | "http"
    # Per DECISIONS.md cross-cutting rule: **nothing above the seam may branch
    # on transport** — this is a declared info field, not a routing signal.


class HealthStatus(BaseModel):
    """Health check result for a knowledge system.

    Allowed ``status`` values:
      - ``ok``            — reachable and ready to serve queries.
      - ``degraded``      — partially available (e.g. vector up, graph down).
      - ``unreachable``   — the system reports it can't be reached.
      - ``not_configured``— the system's backend isn't configured/enabled.
      - ``error``         — the ``health()`` probe itself raised (distinct from
                            a system that *reports* unreachable). Emitted by the
                            /health aggregator when a system's health() throws.
    """

    status: str = Field(
        description="ok | degraded | unreachable | not_configured | error"
    )
    details: dict = Field(default_factory=dict, description="Backend-specific diagnostics")


# ── Request/Response types (minimal; prefer reuse where existing types fit) ──


class RecallRequest(BaseModel):
    """Unified recall request across knowledge systems.

    Wraps SearchMemoryRequest for base; maps to CodeIntelEngine protocol methods
    for code systems. Fields unused by a given system are ignored (N/A honesty).
    """

    query: str = Field(min_length=1, max_length=2000)
    user_id: str | None = None
    project_id: str | None = None
    limit: int = Field(default=10, ge=1, le=100)
    # Op hint for code systems (query/neighbors/path/locate); base ignores it.
    operation: str | None = Field(
        default=None,
        description="Code-system op hint: query | neighbors | path | locate | impact",
    )
    # Code-system-specific params; base ignores these.
    label: str | None = Field(default=None, description="For neighbors: symbol label")
    source: str | None = Field(default=None, description="For path: source symbol")
    target: str | None = Field(default=None, description="For path: target symbol")
    mode: str | None = Field(default="bfs", description="For query: bfs | dfs")
    depth: int | None = Field(default=3, description="For query: traversal depth")
    # Phase G: forward caller caps so routed code ops honor them (not hardcoded).
    max_hops: int | None = Field(default=None, description="For path/impact: max hops")
    relation_filter: str | None = Field(default=None, description="For neighbors: relation filter")
    token_budget: int | None = Field(default=None, description="For query: max output tokens")


class SystemAnswer(BaseModel):
    """A knowledge system's answer to a recall request.

    Section-composed fusion (PLAN §6): each system returns a self-contained
    answer; the fusion layer composes sections (structure + semantics + memory)
    without interleaving scores.
    """

    system_name: str = Field(description="Which system answered (registry key)")
    system_version: str | None = Field(
        default=None, description="Engine version stamp (for bench attribution)"
    )
    content: str = Field(
        description="Text rendering of results (code: nodes+edges; base: memory list)"
    )
    hits: list[dict] | None = Field(
        default=None,
        description="Structured hits (optional; base: MemoryResponse dicts, code: symbol cards)",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="System-specific metadata (scores, provenance, staleness hints)",
    )


class IndexRequest(BaseModel):
    """Index-triggering request for code systems (base is a no-op)."""

    source: str = Field(
        description="Repo path, git URL, or code_space ref to index"
    )
    project_id: str | None = None
    incremental: bool = Field(
        default=True, description="Incremental by content-hash vs full rebuild"
    )
    system_name: str | None = Field(
        default=None,
        description="Explicit system to index with; None = routing decision",
    )


class IndexReport(BaseModel):
    """Synchronous index report (small/fast indexes); async jobs return TaskRef."""

    files_indexed: int
    symbols_indexed: int
    edges_indexed: int = 0
    incremental: bool
    duration_s: float
    system_version: str | None = None


class TaskRef(BaseModel):
    """Reference to an async task (ARQ job) for long-running index operations."""

    task_id: str
    status_url: str | None = None


# ── The Protocol ────────────────────────────────────────────────────


class KnowledgeSystem(Protocol):
    """Protocol for pluggable knowledge backends.

    Implementations:
      - NSMemorySystem: wraps the existing MemoryService search facade (base system).
      - CodeKnowledgeSystem: wraps a CodeIntelEngine implementation (one registry
        entry per backend: code-cbm, code-graphify, code-native).

    The router sees only systems; engines are an implementation detail of
    CodeKnowledgeSystem. Each system declares capabilities honestly (N/A for
    unsupported ops rather than emulating), and only healthy systems are eligible
    for routing.
    """

    info: KnowledgeSystemInfo

    def health(self) -> HealthStatus:
        """Health check: is this system reachable and ready to serve queries?

        Used by /health aggregation and routing eligibility. A system with
        status != "ok" is ineligible for routing (base always answers; code
        systems degrade gracefully).
        """
        ...

    def recall(self, req: RecallRequest) -> SystemAnswer:
        """Read-side query: search/lookup across this system's knowledge.

        Base: semantic search over vector+graph stores (existing search facade).
        Code: map operation hint to the engine's protocol methods (query/neighbors/
        path/locate/impact), normalize FQNs to canonical, batched anchor join.
        """
        ...

    def index(self, req: IndexRequest) -> TaskRef | IndexReport:
        """Trigger indexing (code systems only; base is a no-op).

        Returns TaskRef for async/long-running jobs (CBM large repos); IndexReport
        for fast in-process indexes (graphify lib). Base raises NotImplementedError
        or returns a no-op report.
        """
        ...

    # ── remember() is DELIBERATELY ABSENT (PLAN §0.1) ──
    # External code systems are derived indexes — rebuilt from source, never
    # authoritative. Writing memories into CBM/graphify would create a second
    # source of truth that reindex destroys. `remember` always writes to the
    # base NS store; what routing governs is recall-side system selection and
    # index triggering. Memories concerning code symbols use **anchors**, which
    # live in NS Neo4j keyed by canonical FQN — anchors survive reindex and
    # engine swap because they're not in the external system.
