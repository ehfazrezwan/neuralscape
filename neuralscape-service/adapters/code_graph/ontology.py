"""Graphiti custom entity + edge types for the code_graph adapter.

MINIMAL by design (roadmap Phase F hard rule): the raw code graph is NEVER
mirrored into Graphiti — it is huge, churns with every commit, and Graphify
already serves it live through NS's delegation tools. What lands in Graphiti is
only the *stable semantic layer* distilled at ingest: module purposes
(communities), sparse hub symbols (god nodes), and depends_on-style relations
between them. So two entity types and three thin edges are the whole ontology.

Same Graphiti rules as the trading ontology:
1. the class docstring is load-bearing (fed to the LLM as the type definition);
2. avoid reserved field names; every attribute Optional with a rich
   ``Field(description=...)``. Load-bearing data lives on entity nodes, never
   on edge attributes (#1111 hedge).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Entity types ────────────────────────────────────────────────────


class Module(BaseModel):
    """A module / architectural community of the codebase — a named cluster of code with one purpose.

    Corresponds to a Graphify LLM-labeled community (e.g. "Ingestion pipeline",
    "Auth & tokens"), not to a single file.
    """

    purpose: str | None = Field(default=None, description="What this module is for, in one sentence")
    key_members: str | None = Field(default=None, description="Representative member symbols/files, comma-separated")
    community_id: str | None = Field(default=None, description="Graphify community id in the source graph.json")
    graph_ref: str | None = Field(default=None, description="graph_id (NS artifact id) of the code graph this came from")


class Symbol(BaseModel):
    """A code symbol (class/function) worth naming in memory — SPARSE, hubs only.

    Only god nodes / heavily-connected abstractions from the code graph become
    Symbols; ordinary functions never do (they churn with every commit and are
    served live by the code-graph query tools instead).
    """

    symbol_kind: str | None = Field(default=None, description="e.g. 'class', 'function'")
    role: str | None = Field(default=None, description="Why this symbol is a hub (what depends on it)")
    degree: str | None = Field(default=None, description="Connection count in the code graph when ingested")
    source_file: str | None = Field(default=None, description="File path the symbol is defined in")
    node_ref: str | None = Field(default=None, description="Graphify node id in the source graph.json")


# ── Edge types (thin markers — semantics live in the nodes) ─────────


class DEPENDS_ON(BaseModel):
    """Source module/symbol depends on the target (a stable, summary-level dependency)."""


class PART_OF(BaseModel):
    """Symbol is part of this Module (community membership)."""


class CONNECTS_TO(BaseModel):
    """A surprising cross-module connection Graphify flagged (non-obvious coupling)."""


# ── Registries handed to add_episode ────────────────────────────────

ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Module": Module,
    "Symbol": Symbol,
}

EDGE_TYPES: dict[str, type[BaseModel]] = {
    "DEPENDS_ON": DEPENDS_ON,
    "PART_OF": PART_OF,
    "CONNECTS_TO": CONNECTS_TO,
}

EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("Module", "Module"): ["DEPENDS_ON", "CONNECTS_TO"],
    ("Symbol", "Module"): ["PART_OF", "DEPENDS_ON"],
    ("Symbol", "Symbol"): ["DEPENDS_ON", "CONNECTS_TO"],
}

CUSTOM_EXTRACTION_INSTRUCTIONS = (
    "This text is a distilled summary of a code knowledge graph (modules, hub "
    "symbols, dependencies, rationale comments). Extract ONLY the stable "
    "semantic layer: Modules (architectural communities and their purpose) and "
    "sparse hub Symbols (god nodes / core abstractions), with DEPENDS_ON / "
    "PART_OF / CONNECTS_TO relations between them. Do NOT extract ordinary "
    "functions, individual files, import lists, or line-level details — the "
    "raw code structure churns with every commit and is served live by the "
    "code-graph query tools, so mirroring it here would only rot."
)
