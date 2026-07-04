"""Schemas for neuralscape-service: enums, category taxonomy, request/response models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────


class MemoryScope(str, Enum):
    GLOBAL = "global"
    PROJECT = "project"


class MemoryVisibility(str, Enum):
    """Who can read a memory.

    - PRIVATE: only the writing user (`owner_user_id`) can read.
    - SHARED: any authenticated user in this Neuralscape instance can read.
    - STANDARD: an *authoritative* pool readable by everyone but writable only
      by a "dictator" (see `Settings.dictator_user_ids`). Standard memories are
      organization rules/policies that always load at session start and take
      precedence over personal preferences on conflict. Gated behind
      `STANDARDS_ENABLED`; it is never a per-category default (explicit opt-in
      only).

    Visibility is orthogonal to scope: a private memory can be `global`
    (cross-project for that user) or `project`-scoped; a shared memory can
    be cross-project or project-scoped (visible to anyone who searches that
    project).

    NOTE on ``__str__`` override: in Python 3.11+, ``str(EnumMember)`` of a
    ``(str, Enum)`` subclass returns the repr-style ``"MemoryVisibility.SHARED"``
    instead of the value ``"shared"`` (PEP 663-related regression). Without
    this override, ``str(visibility)`` calls in memory_service.py used to
    write ``"MemoryVisibility.SHARED"`` into Qdrant metadata, breaking the
    GET API response shape and crashing the conversation_compiler extension
    handler with ``ValueError: 'MemoryVisibility.SHARED' is not a valid
    MemoryVisibility``. Overriding ``__str__`` to return the value matches
    the ``StrEnum`` behavior (Python 3.11+) without bumping the minimum
    Python version. See ``normalize_visibility()`` for legacy-data recovery.
    """
    PRIVATE = "private"
    SHARED = "shared"
    STANDARD = "standard"

    def __str__(self) -> str:
        return self.value


def normalize_visibility(v) -> str | None:
    """Coerce any visibility input to its canonical lowercase string value.

    Handles four input shapes:
      - ``None``                              → ``None``
      - ``MemoryVisibility`` enum             → its ``.value``
      - ``"shared"`` / ``"private"`` (str)    → unchanged
      - ``"MemoryVisibility.SHARED"`` (legacy stringified-enum from the
        Python 3.11+ ``str(Enum)`` bug)        → ``"shared"``

    Use this at any boundary where visibility crosses a serialization
    layer (event payloads, Qdrant writes, JSON responses) so old data
    written before the ``__str__`` fix still parses cleanly.

    Raises ``ValueError`` if the input is a string that doesn't match
    a known visibility after prefix-stripping.
    Raises ``TypeError`` for unsupported input types.
    """
    if v is None:
        return None
    if isinstance(v, MemoryVisibility):
        return v.value
    if isinstance(v, str):
        # Strip any "MemoryVisibility." prefix that may have been written
        # by the pre-__str__-override stringification bug, then lowercase.
        candidate = v.rsplit(".", 1)[-1].lower()
        MemoryVisibility(candidate)  # validates membership; raises ValueError if unknown
        return candidate
    raise TypeError(f"Cannot normalize visibility from {type(v).__name__}: {v!r}")


class MemoryCategory(str, Enum):
    # Semantic (facts/knowledge)
    PREFERENCE = "preference"
    PERSONAL_FACT = "personal_fact"
    TECHNICAL_SKILL = "technical_skill"
    DOMAIN_KNOWLEDGE = "domain_knowledge"

    # Project-specific
    TECH_STACK = "tech_stack"
    CONVENTION = "convention"
    ARCHITECTURE = "architecture"
    DEPENDENCY = "dependency"

    # Episodic
    DECISION = "decision"
    INTERACTION = "interaction"

    # Procedural
    WORKFLOW = "workflow"
    PROCEDURE = "procedure"

    # Working
    TASK_CONTEXT = "task_context"


# ──────────────────────────────────────────────
# Category taxonomy (descriptions are domain-neutral as of memory-model v2)
# ──────────────────────────────────────────────

MEMORY_CATEGORIES: dict[str, str] = {
    "preference": "Personal preferences: how the user likes to work, communicate, and consume information",
    "personal_fact": "Personal details about the user: name, timezone, role, team, working hours",
    "technical_skill": "Skills and proficiencies the user has, technical or otherwise",
    "domain_knowledge": "Subject-matter knowledge the user has accumulated (industry, market, scientific, organizational)",
    "tech_stack": "Tools, systems, or platforms used in this project",
    "convention": "Norms and conventions adopted by this project (code style, communication, naming, process)",
    "architecture": "Structural decisions about this project (system design, org structure, information architecture)",
    "dependency": "External dependencies of this project (libraries, vendors, blocking teams, pinned versions)",
    "decision": "Decisions made — with the why, not just the what",
    "interaction": "Notable events: meetings, conversations, calls, demos",
    "workflow": "Recurring multi-step processes (git flow, deployment, review, weekly rituals)",
    "procedure": "Step-by-step how-tos for repeatable tasks",
    "task_context": "Active work-in-progress: current goals, recent state, blockers — short-lived",
}

# Categories that default to global scope
GLOBAL_CATEGORIES = {"preference", "personal_fact", "technical_skill", "domain_knowledge"}

# Categories that require project_id and default to project scope
PROJECT_CATEGORIES = {"tech_stack", "convention", "architecture", "dependency"}

# Categories that can be either scope
FLEXIBLE_CATEGORIES = {"decision", "interaction", "workflow", "procedure", "task_context"}


def default_scope_for_category(category: str) -> MemoryScope:
    """Return the default scope for a given category."""
    if category in GLOBAL_CATEGORIES:
        return MemoryScope.GLOBAL
    if category in PROJECT_CATEGORIES:
        return MemoryScope.PROJECT
    return MemoryScope.GLOBAL


def register_categories(
    categories: dict[str, str],
    *,
    global_categories: set[str] | None = None,
    project_categories: set[str] | None = None,
    vault_paths: dict[str, str] | None = None,
) -> None:
    """Extend the shared taxonomy with a knowledge adapter's categories.

    Adapters (see :mod:`adapters`) call this at import so that their categories
    become first-class citizens of the *fixed envelope*: ``store_raw``'s
    membership check, the ingest request validators, and the fact parser all
    read the same module-level registries mutated here. Purely **additive** —
    it never removes or reassigns the core 13, so nothing regresses.

    Idempotent: re-registering the same category updates its description in
    place. Categories left out of ``global_categories``/``project_categories``
    stay *flexible* (scope follows the caller's ``project_id`` — the same rule
    the ingest pipeline applies to ``domain_knowledge``).
    """
    MEMORY_CATEGORIES.update(categories)
    if global_categories:
        GLOBAL_CATEGORIES.update(global_categories)
    if project_categories:
        PROJECT_CATEGORIES.update(project_categories)
    # Categories that are neither global nor project are flexible.
    flexible = set(categories) - GLOBAL_CATEGORIES - PROJECT_CATEGORIES
    FLEXIBLE_CATEGORIES.update(flexible)
    if vault_paths:
        CATEGORY_VAULT_PATHS.update(vault_paths)


# Vault folder paths for each category (mirrors neuralscape-plugin/src/types.ts)
CATEGORY_VAULT_PATHS: dict[str, str] = {
    "preference": "Semantic/Preferences",
    "personal_fact": "Semantic/Personal-Facts",
    "technical_skill": "Semantic/Technical-Skills",
    "domain_knowledge": "Semantic/Domain-Knowledge",
    "tech_stack": "Project/Tech-Stack",
    "convention": "Project/Conventions",
    "architecture": "Project/Architecture",
    "dependency": "Project/Dependencies",
    "decision": "Episodic/Decisions",
    "interaction": "Episodic/Interactions",
    "workflow": "Procedural/Workflows",
    "procedure": "Procedural/Procedures",
    "task_context": "Working/Task-Context",
}


# ──────────────────────────────────────────────
# Memory model v2: controlled vocabularies
# ──────────────────────────────────────────────

DOMAIN_VOCAB: set[str] = {
    "coding", "research", "meeting", "writing", "ops", "personal", "general",
}

OBSERVATION_TYPE_VOCAB: set[str] = {
    "bugfix", "feature", "refactor", "decision", "discovery",
    "gotcha", "pattern", "trade_off", "research_note",
    "meeting_outcome", "task_plan", "fact",
    # dreaming: higher-order insight synthesized by the REM phase
    "reflection",
}

CONCEPT_VOCAB: set[str] = {
    "how-it-works", "why-it-exists", "what-changed",
    "problem-solution", "gotcha", "pattern", "trade-off",
    "open-question", "next-step", "blocker",
}

SOURCE_TYPE_VOCAB: set[str] = {
    "conversation", "tool_extraction", "explicit", "imported", "compiler",
    # dreaming: authored by the background consolidation sweep. Excluded
    # from the next sweep's LIGHT intake (feedback-loop guard).
    "dream",
}

# Epistemic level: HOW a memory came to be known (provenance epistemics).
# - "explicit":   directly stated by the user/source (extraction default)
# - "deductive":  strictly entailed by the specific premise memories it cites
# - "inductive":  a generalization across >= 2 premise memories
# - "reflection": dream-authored insight without a stricter self-label
# Legacy memories carry no epistemic_level; readers treat null as unknown.
# Pairs with `derived_from` (the premise memory-id list) to make every
# derived memory able to show its premises via get_reasoning_chain.
EPISTEMIC_LEVEL_VOCAB: set[str] = {"explicit", "deductive", "inductive", "reflection"}

# memory_kind distinguishes a distilled atomic fact from a verbatim passage
# (chunk) of an ingested document. Legacy memories have no memory_kind set;
# readers should treat a null value as "fact".
MEMORY_KIND_VOCAB: set[str] = {"fact", "passage"}

# Connector adapter types recognised by the ingest/connector subsystem.
# `manual` = content a user provided directly (pasted context); `file_upload` =
# a file/zip the user uploaded. Both are first-class origins that don't come from
# a configured external connector, so their memories carry a synthetic source_ref.
# `okf_bundle` = a concept document imported from an Open Knowledge Format
# bundle (directory or zip); external_id is the concept ID, parent_id the
# bundle URI/path.
CONNECTOR_TYPE_VOCAB: set[str] = {
    "google_drive",
    "notion",
    "generic_rest",
    "mcp",
    "manual",
    "file_upload",
    "okf_bundle",
}


# ──────────────────────────────────────────────
# Default visibility per category (multi-user model)
# ──────────────────────────────────────────────
#
# When a write doesn't supply `visibility`, the server picks from this table.
# Personal categories default to PRIVATE; project / team-relevant categories
# default to SHARED. Callers can always override.

DEFAULT_VISIBILITY_FOR_CATEGORY: dict[str, MemoryVisibility] = {
    # Personal — about the user themselves
    "preference": MemoryVisibility.PRIVATE,
    "personal_fact": MemoryVisibility.PRIVATE,
    "technical_skill": MemoryVisibility.PRIVATE,
    "domain_knowledge": MemoryVisibility.PRIVATE,
    # Working — short-lived WIP, stays private until shipped
    "task_context": MemoryVisibility.PRIVATE,
    # Project — team-wide knowledge
    "tech_stack": MemoryVisibility.SHARED,
    "convention": MemoryVisibility.SHARED,
    "architecture": MemoryVisibility.SHARED,
    "dependency": MemoryVisibility.SHARED,
    # Episodic — decisions and notable events affect the team
    "decision": MemoryVisibility.SHARED,
    "interaction": MemoryVisibility.SHARED,
    # Procedural — team processes
    "workflow": MemoryVisibility.SHARED,
    "procedure": MemoryVisibility.SHARED,
}


def default_visibility_for_category(category: str) -> MemoryVisibility:
    """Return the default visibility for a given category.

    Falls back to PRIVATE for unknown categories (safe default — no
    accidental cross-user reads).
    """
    return DEFAULT_VISIBILITY_FOR_CATEGORY.get(category, MemoryVisibility.PRIVATE)


# ──────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────


# Reusable field constraints
_ID_PATTERN = r"^[a-zA-Z0-9_.\-]+$"


def validate_adapter_name(v: str) -> str:
    """Reject unknown knowledge-adapter names at the request boundary.

    A typo'd ``adapter`` must fail loudly here (422) — silently degrading to the
    default adapter would ingest a document *without* the taxonomy/ontology the
    caller asked for, which is far worse than an error. (The worker-side
    ``get_adapter`` still degrades gracefully, but only for jobs already queued
    when an adapter was removed.) Imported lazily: ``adapters`` imports this
    module at import time, so a top-level import here would be circular.
    """
    from adapters import list_adapters

    known = list_adapters()
    if v not in known:
        raise ValueError(f"Unknown adapter '{v}'. Available: {known}")
    return v


# ──────────────────────────────────────────────
# Source provenance (data-layer connectors)
# ──────────────────────────────────────────────


class RetrievalHandle(BaseModel):
    """How a consuming agent can deterministically re-fetch the source.

    Points at the MCP server + tool + args the agent should invoke to pull
    the original (or fuller) content — e.g.
    ``{mcp_server: "claude_ai_Notion", tool: "notion-fetch", args: {"id": "<page>"}}``.
    """
    mcp_server: str | None = Field(default=None, max_length=200)
    tool: str | None = Field(default=None, max_length=200)
    args: dict | None = None


class SourceDescriptor(BaseModel):
    """Where an ingested memory came from + how to fetch more.

    Nested into the existing Qdrant ``metadata`` dict under ``source_ref`` —
    no payload restructuring. ``connector_id``/``connector_type`` are required
    so a memory can always be traced back to the connector instance that
    produced it; everything else is optional and connector-dependent.
    """
    connector_id: str = Field(max_length=200, description="Configured connector INSTANCE id")
    connector_type: str = Field(description="Adapter type (see CONNECTOR_TYPE_VOCAB)")
    external_id: str | None = Field(default=None, max_length=500, description="Stable id in the source system")
    parent_id: str | None = Field(default=None, max_length=500, description="The file/page; chunks of one doc share this")
    url: str | None = Field(default=None, max_length=2000, description="Human/clickable link to the source")
    title: str | None = Field(default=None, max_length=1000)
    chunk_index: int | None = Field(default=None, ge=0, description="Position of this passage within the parent doc")
    span: list[int] | None = Field(default=None, max_length=2, description="[start_char, end_char] within the parent")
    content_hash: str | None = Field(default=None, max_length=64)
    stored_path: str | None = Field(default=None, max_length=1000, description="Path to the persisted artifact within the ingest storage volume")
    revision: str | None = Field(default=None, max_length=200, description="Source-side version (etag / last_edited_time)")
    last_synced_at: str | None = Field(default=None, description="ISO 8601 timestamp of last sync from source")
    retrieval: RetrievalHandle | None = None

    @field_validator("connector_type")
    @classmethod
    def _validate_connector_type(cls, v: str) -> str:
        if v not in CONNECTOR_TYPE_VOCAB:
            raise ValueError(
                f"Invalid connector_type '{v}'. Must be one of: {sorted(CONNECTOR_TYPE_VOCAB)}"
            )
        return v


class StoreMemoryRequest(BaseModel):
    """Store memories from conversation via LLM extraction."""
    messages: list[dict] = Field(
        description="Messages to extract memories from (list of {role, content} dicts)",
        max_length=500,
    )
    # user_id is optional when the caller authenticates with a per-user token —
    # the server uses request.state.user_id from the token. Kept for legacy
    # shared-key callers that supply identity in the body. When both are
    # present the server validates they agree (400 on mismatch).
    user_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    project_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    agent_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    run_id: str | None = Field(default=None, max_length=100)
    domain: str | None = Field(default=None, description="High-level life-context domain (memory-model v2)")
    visibility: MemoryVisibility | None = Field(
        default=None,
        description="Multi-user model: who can read this memory. Defaults per-category. 'standard' is a dictator-only authoritative tier (requires STANDARDS_ENABLED).",
    )

    @field_validator("domain")
    @classmethod
    def _validate_domain(cls, v: str | None) -> str | None:
        if v is not None and v not in DOMAIN_VOCAB:
            raise ValueError(f"Invalid domain '{v}'. Must be one of: {sorted(DOMAIN_VOCAB)}")
        return v


class RawMemoryRequest(BaseModel):
    """Store a single pre-categorized fact (no LLM extraction).

    Memory-model v2 fields (domain, observation_type, concepts, source_type,
    related_memory_ids, confidence, expires_at) are all optional and additive —
    omitting them produces the same behavior as memory-model v1.

    Multi-user model: when authenticated via a per-user token, `user_id`
    becomes optional (the token's user_id is used). The new `visibility`
    field controls whether the memory is private to its writer or shared
    with the team.
    """
    content: str = Field(
        description="The memory content to store",
        min_length=1,
        max_length=10000,
    )
    user_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    category: str = Field(description="Memory category (must be one of MEMORY_CATEGORIES)")
    scope: str = Field(default="global", description="'global' or 'project'")
    project_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    tags: list[str] | None = Field(default=None, max_length=20)
    agent_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    run_id: str | None = Field(default=None, max_length=100)
    visibility: MemoryVisibility | None = Field(
        default=None,
        description="Multi-user model: 'private' (owner only), 'shared' (team-wide), or 'standard' (dictator-only authoritative tier; requires STANDARDS_ENABLED). Defaults per-category.",
    )

    # Memory-model v2 (all optional)
    domain: str | None = Field(default=None, description="High-level life-context domain")
    observation_type: str | None = Field(default=None, description="Shape of observation, orthogonal to category")
    concepts: list[str] | None = Field(default=None, max_length=5, description="Cross-cutting controlled-vocab tags")
    source_type: str | None = Field(default=None, description="Provenance of this memory")
    related_memory_ids: list[str] | None = Field(default=None, max_length=10, description="UUIDs of related memories")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Extractor's self-rated confidence")
    expires_at: datetime | None = Field(default=None, description="Optional expiry for short-lived memories")

    # Provenance epistemics (A1, optional, additive)
    derived_from: list[str] | None = Field(
        default=None, max_length=10,
        description="Premise memory IDs this memory was derived from (reasoning-chain provenance)",
    )
    epistemic_level: str | None = Field(
        default=None,
        description="How this memory is known: explicit | deductive | inductive | reflection",
    )

    # Data-layer connectors (optional, additive)
    memory_kind: str | None = Field(default=None, description="'fact' (distilled) or 'passage' (verbatim chunk). Null → fact.")
    source_ref: SourceDescriptor | None = Field(default=None, description="Provenance + retrieval handle for ingested content")

    @field_validator("domain")
    @classmethod
    def _validate_domain(cls, v: str | None) -> str | None:
        if v is not None and v not in DOMAIN_VOCAB:
            raise ValueError(f"Invalid domain '{v}'. Must be one of: {sorted(DOMAIN_VOCAB)}")
        return v

    @field_validator("observation_type")
    @classmethod
    def _validate_observation_type(cls, v: str | None) -> str | None:
        if v is not None and v not in OBSERVATION_TYPE_VOCAB:
            raise ValueError(
                f"Invalid observation_type '{v}'. Must be one of: {sorted(OBSERVATION_TYPE_VOCAB)}"
            )
        return v

    @field_validator("concepts")
    @classmethod
    def _validate_concepts(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        unknown = [c for c in v if c not in CONCEPT_VOCAB]
        if unknown:
            raise ValueError(
                f"Unknown concepts: {unknown}. Must be from: {sorted(CONCEPT_VOCAB)}"
            )
        return v

    @field_validator("source_type")
    @classmethod
    def _validate_source_type(cls, v: str | None) -> str | None:
        if v is not None and v not in SOURCE_TYPE_VOCAB:
            raise ValueError(
                f"Invalid source_type '{v}'. Must be one of: {sorted(SOURCE_TYPE_VOCAB)}"
            )
        return v

    @field_validator("memory_kind")
    @classmethod
    def _validate_memory_kind(cls, v: str | None) -> str | None:
        if v is not None and v not in MEMORY_KIND_VOCAB:
            raise ValueError(
                f"Invalid memory_kind '{v}'. Must be one of: {sorted(MEMORY_KIND_VOCAB)}"
            )
        return v

    @field_validator("epistemic_level")
    @classmethod
    def _validate_epistemic_level(cls, v: str | None) -> str | None:
        if v is not None and v not in EPISTEMIC_LEVEL_VOCAB:
            raise ValueError(
                f"Invalid epistemic_level '{v}'. Must be one of: {sorted(EPISTEMIC_LEVEL_VOCAB)}"
            )
        return v


class RawMemoryBatchRequest(BaseModel):
    """Store multiple pre-categorized facts in one request (memory-model v2)."""
    memories: list[RawMemoryRequest] = Field(min_length=1, max_length=50)


class IngestDocumentRequest(BaseModel):
    """Ingest a document from a data layer: chunk → passages + distilled facts.

    The ``source`` descriptor is attached to every memory produced (passages
    carry per-chunk ``chunk_index``/``span``; facts carry the parent-level
    descriptor). Re-ingesting the same content is idempotent via content-hash
    dedup in ``store_raw``.
    """
    content: str = Field(min_length=1, max_length=2_000_000, description="Full document text to ingest")
    source: SourceDescriptor = Field(description="Provenance + retrieval handle for this document")
    category: str = Field(default="domain_knowledge", description="Category for produced memories")
    scope: str = Field(default="global", description="'global' or 'project'")
    project_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    user_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    visibility: MemoryVisibility | None = Field(default=None)
    extract_facts: bool = Field(default=True, description="Run LLM extraction to also store distilled facts")
    index_passages: bool = Field(default=True, description="Chunk + store verbatim passages")
    tags: list[str] | None = Field(default=None, max_length=20)
    agent_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    run_id: str | None = Field(default=None, max_length=100)
    adapter: str = Field(
        default="default",
        max_length=100,
        description="Knowledge adapter selecting taxonomy/chunker/extractor/graph-ontology (e.g. 'default', 'trading_strategy').",
    )

    @field_validator("category")
    @classmethod
    def _validate_category(cls, v: str) -> str:
        if v not in MEMORY_CATEGORIES:
            raise ValueError(f"Invalid category '{v}'. Must be one of: {list(MEMORY_CATEGORIES.keys())}")
        return v

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, v: str) -> str:
        if v not in ("global", "project"):
            raise ValueError("scope must be 'global' or 'project'")
        return v

    @field_validator("adapter")
    @classmethod
    def _validate_adapter(cls, v: str) -> str:
        return validate_adapter_name(v)


class IngestTextRequest(BaseModel):
    """Manually provide a block of context — a first-class ingestion path.

    Unlike :class:`IngestDocumentRequest`, no ``source`` descriptor is required:
    the server fabricates a synthetic ``manual`` source_ref so the produced
    memories still carry provenance ("manually provided") and backlink to a
    stable parent id (the content hash). Same chunk→passages + distilled-facts
    pipeline as document ingest; runs async on the ingest queue.
    """
    content: str = Field(min_length=1, max_length=2_000_000, description="The context text to ingest")
    title: str | None = Field(default=None, max_length=1000, description="Human label for this context")
    category: str = Field(default="domain_knowledge", description="Category for produced memories")
    scope: str = Field(default="global", description="'global' or 'project'")
    project_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    user_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    visibility: MemoryVisibility | None = Field(default=None)
    extract_facts: bool = Field(default=True, description="Run LLM extraction to also store distilled facts")
    index_passages: bool = Field(default=True, description="Chunk + store verbatim passages")
    tags: list[str] | None = Field(default=None, max_length=20)
    agent_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    run_id: str | None = Field(default=None, max_length=100)
    adapter: str = Field(
        default="default",
        max_length=100,
        description="Knowledge adapter selecting taxonomy/chunker/extractor/graph-ontology (e.g. 'default', 'trading_strategy').",
    )

    @field_validator("category")
    @classmethod
    def _validate_category(cls, v: str) -> str:
        if v not in MEMORY_CATEGORIES:
            raise ValueError(f"Invalid category '{v}'. Must be one of: {list(MEMORY_CATEGORIES.keys())}")
        return v

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, v: str) -> str:
        if v not in ("global", "project"):
            raise ValueError("scope must be 'global' or 'project'")
        return v

    @field_validator("adapter")
    @classmethod
    def _validate_adapter(cls, v: str) -> str:
        return validate_adapter_name(v)


class ConnectorConfigRequest(BaseModel):
    """Register a data-layer connector instance.

    ``credentials`` is connector-type specific (e.g. ``{"token": "..."}`` for
    Notion, an OAuth refresh-token blob for Google Drive, an MCP server spec
    for the generic adapter). It is encrypted at rest in the vault and never
    returned by the API.
    """
    connector_id: str = Field(max_length=200, pattern=_ID_PATTERN, description="Unique instance id, e.g. 'notion-personal'")
    connector_type: str = Field(description="Adapter type (see CONNECTOR_TYPE_VOCAB)")
    name: str | None = Field(default=None, max_length=200, description="Human label")
    credentials: dict = Field(description="Connector-type-specific secrets (encrypted at rest)")
    config: dict | None = Field(default=None, description="Non-secret connector options (e.g. folder/db filters)")
    enabled: bool = Field(default=True, description="Whether the sync cron includes this connector")
    # Defaults for memories produced by this connector's syncs.
    default_category: str = Field(default="domain_knowledge")
    default_scope: str = Field(default="global")
    default_project_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    default_visibility: MemoryVisibility | None = Field(default=None)

    @field_validator("connector_type")
    @classmethod
    def _validate_connector_type(cls, v: str) -> str:
        if v not in CONNECTOR_TYPE_VOCAB:
            raise ValueError(
                f"Invalid connector_type '{v}'. Must be one of: {sorted(CONNECTOR_TYPE_VOCAB)}"
            )
        return v

    @field_validator("default_category")
    @classmethod
    def _validate_default_category(cls, v: str) -> str:
        if v not in MEMORY_CATEGORIES:
            raise ValueError(f"Invalid default_category '{v}'.")
        return v


class SearchMemoryRequest(BaseModel):
    """Semantic search across memories.

    Multi-user model: results combine the caller's personal memories
    (visibility=private, owned by user_id) and the shared pool
    (visibility=shared, any owner). Use `visibility` to filter to one
    pool, or `include_shared=False` to scope to personal-only.
    """
    query: str = Field(min_length=1, max_length=2000)
    user_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    project_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    # Item cap sized for the core 13 + adapter-registered categories (e.g. the
    # 12 trading categories) so a filter can name every known category.
    categories: list[str] | None = Field(default=None, max_length=40)
    scope: str | None = None
    limit: int = Field(default=10, ge=1, le=100)

    # Memory-model v2 filters (all optional)
    domain: str | None = Field(default=None, description="Filter by domain")
    observation_type: str | None = Field(default=None, description="Filter by observation_type")
    concepts: list[str] | None = Field(default=None, max_length=5, description="Filter by concept tags (any-match)")
    memory_kind: str | None = Field(default=None, description="Filter by 'fact' or 'passage'")

    # Multi-user pool selection
    visibility: MemoryVisibility | None = Field(
        default=None,
        description="Restrict results to one pool: 'private' (yours only), 'shared', or 'standard' (authoritative tier). Default: your private + shared (+ standard when enabled).",
    )
    include_shared: bool = Field(
        default=True,
        description="When false, exclude shared-pool memories entirely (search personal only).",
    )

    @field_validator("domain")
    @classmethod
    def _validate_domain(cls, v: str | None) -> str | None:
        if v is not None and v not in DOMAIN_VOCAB:
            raise ValueError(f"Invalid domain '{v}'. Must be one of: {sorted(DOMAIN_VOCAB)}")
        return v

    @field_validator("observation_type")
    @classmethod
    def _validate_observation_type(cls, v: str | None) -> str | None:
        if v is not None and v not in OBSERVATION_TYPE_VOCAB:
            raise ValueError(
                f"Invalid observation_type '{v}'. Must be one of: {sorted(OBSERVATION_TYPE_VOCAB)}"
            )
        return v

    @field_validator("concepts")
    @classmethod
    def _validate_concepts(cls, v: list[str] | None) -> list[str] | None:
        # Mirror RawMemoryRequest's validation so the search filter contract
        # is symmetric with the write contract — typos surface as 422, not
        # silent misses.
        if v is None:
            return v
        unknown = [c for c in v if c not in CONCEPT_VOCAB]
        if unknown:
            raise ValueError(
                f"Unknown concepts: {unknown}. Must be from: {sorted(CONCEPT_VOCAB)}"
            )
        return v

    @field_validator("memory_kind")
    @classmethod
    def _validate_memory_kind(cls, v: str | None) -> str | None:
        if v is not None and v not in MEMORY_KIND_VOCAB:
            raise ValueError(
                f"Invalid memory_kind '{v}'. Must be one of: {sorted(MEMORY_KIND_VOCAB)}"
            )
        return v


class GraphSearchRequest(BaseModel):
    """Knowledge graph search."""
    query: str = Field(min_length=1, max_length=2000)
    user_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    project_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    limit: int = Field(default=10, ge=1, le=100)
    search_config: dict | None = Field(
        default=None,
        description="Optional SearchConfig dict to override default hybrid search",
    )


class PatchMemoryRequest(BaseModel):
    """Partially update a memory (true PATCH semantics).

    Only fields the caller actually sent are applied — handlers must read
    ``model_fields_set`` so an explicit ``null`` (clear the field, where legal)
    is distinguishable from an omitted field. ``scope`` is never accepted: it
    is always re-derived server-side from the effective category + project_id,
    exactly as on write. ``owner_user_id`` is never editable.

    Permission model (enforced in the service):
    - shared memories: any authenticated user may edit organizational metadata
      (tags/category/project_id/v2 fields); content and visibility edits are
      owner-or-dictator.
    - private memories: owner only.
    - standard tier: dictator only.
    """
    user_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    content: str | None = Field(default=None, min_length=1, max_length=10000)
    category: str | None = None
    project_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    tags: list[str] | None = Field(default=None, max_length=20, description="Full replacement of the tags list")
    visibility: MemoryVisibility | None = None

    # Memory-model v2 (all optional)
    domain: str | None = None
    observation_type: str | None = None
    concepts: list[str] | None = Field(default=None, max_length=5)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    expires_at: datetime | None = None

    @field_validator("domain")
    @classmethod
    def _validate_domain(cls, v: str | None) -> str | None:
        if v is not None and v not in DOMAIN_VOCAB:
            raise ValueError(f"Invalid domain '{v}'. Must be one of: {sorted(DOMAIN_VOCAB)}")
        return v

    @field_validator("observation_type")
    @classmethod
    def _validate_observation_type(cls, v: str | None) -> str | None:
        if v is not None and v not in OBSERVATION_TYPE_VOCAB:
            raise ValueError(
                f"Invalid observation_type '{v}'. Must be one of: {sorted(OBSERVATION_TYPE_VOCAB)}"
            )
        return v

    @field_validator("concepts")
    @classmethod
    def _validate_concepts(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        unknown = [c for c in v if c not in CONCEPT_VOCAB]
        if unknown:
            raise ValueError(
                f"Unknown concepts: {unknown}. Must be from: {sorted(CONCEPT_VOCAB)}"
            )
        return v


class RetagFilters(BaseModel):
    """Filter set selecting the memories a bulk retag applies to (AND semantics)."""
    scope: str | None = None
    category: str | None = Field(default=None, min_length=1)
    project_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    visibility: MemoryVisibility | None = None
    tags_contains: list[str] | None = Field(
        default=None, min_length=1, max_length=10,
        description="Only memories carrying ALL of these tags match",
    )

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, v: str | None) -> str | None:
        if v is not None and v not in {"global", "project"}:
            raise ValueError("scope must be 'global' or 'project'")
        return v

    def any_set(self) -> bool:
        # Truthiness, not `is not None`: an empty string / empty list produces
        # NO Qdrant filter condition downstream, so counting it as "a filter is
        # present" would let `{"tags_contains": []}` bypass the sweep guard.
        return any(
            bool(getattr(self, f))
            for f in ("scope", "category", "project_id", "visibility", "tags_contains")
        )


class RetagOps(BaseModel):
    """Operations a bulk retag applies to each matched memory.

    ``set_project_id`` supports explicit-null clearing: send the key with
    ``null`` to remove the project assignment (handlers read
    ``model_fields_set``). Visibility and content are deliberately NOT bulk
    operations — those are sensitive, single-memory edits.
    """
    add_tags: list[str] | None = Field(default=None, max_length=20)
    remove_tags: list[str] | None = Field(default=None, max_length=20)
    set_project_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    set_category: str | None = None

    def any_set(self) -> bool:
        return bool(
            self.add_tags or self.remove_tags or self.set_category
            or "set_project_id" in self.model_fields_set
        )


class RetagRequest(BaseModel):
    """Bulk retag memories matching a filter set (async, 202 + poll).

    Requires at least one filter — an unfiltered whole-store sweep is refused
    at the request boundary, mirroring the bulk-delete safety posture.
    """
    user_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    filters: RetagFilters
    ops: RetagOps
    dry_run: bool = Field(
        default=False,
        description="When True, report matched/would_update counts without writing",
    )

    @model_validator(mode="after")
    def _validate_shape(self) -> "RetagRequest":
        if not self.filters.any_set():
            raise ValueError("At least one filter is required — refusing an unfiltered retag sweep")
        if not self.ops.any_set():
            raise ValueError("At least one operation is required")
        overlap = set(self.ops.add_tags or []) & set(self.ops.remove_tags or [])
        if overlap:
            raise ValueError(f"Tags cannot be both added and removed: {sorted(overlap)}")
        return self


class BulkDeleteRequest(BaseModel):
    """Bulk delete memories with filters."""
    user_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    scope: str | None = None
    category: str | None = None
    project_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    filter_null_category: bool = Field(
        default=False,
        description="When True, delete only memories with null/missing category instead of all",
    )
    include_shared: bool = Field(
        default=False,
        description=(
            "When False (default), shared memories the caller authored are "
            "preserved even on bulk delete — they're team artifacts. Set True "
            "to also remove the caller's shared writes (admin-style nuke)."
        ),
    )


# ──────────────────────────────────────────────
# Response Models
# ──────────────────────────────────────────────


class MemoryResponse(BaseModel):
    """Single memory in response.

    Memory-model v2 fields render as nulls for legacy memories that didn't store them.
    Multi-user fields (`visibility`, `owner_user_id`) likewise render as null on
    pre-multi-user memories — clients should treat null `visibility` as 'private'
    and null `owner_user_id` as 'unknown / pre-multi-user write'.
    """
    id: str
    memory: str
    category: str | None = None
    scope: str | None = None
    project_id: str | None = None
    tags: list[str] | None = None
    score: float | None = None
    created_at: str | None = None
    updated_at: str | None = None
    source: str | None = None

    # Memory-model v2 (all optional, render as null for legacy memories)
    domain: str | None = None
    observation_type: str | None = None
    concepts: list[str] | None = None
    source_type: str | None = None
    related_memory_ids: list[str] | None = None
    confidence: float | None = None
    expires_at: str | None = None

    # Provenance epistemics (A1; null for memories that predate it)
    derived_from: list[str] | None = None
    epistemic_level: str | None = None

    # Data-layer connectors (null for non-ingested memories)
    memory_kind: str | None = None
    source_ref: dict | None = None

    # Multi-user model
    visibility: str | None = None
    owner_user_id: str | None = None


class StoreMemoryResponse(BaseModel):
    status: str = "ok"
    memories: list[MemoryResponse] = Field(default_factory=list)
    task_id: str | None = None


class SearchMemoryResponse(BaseModel):
    status: str = "ok"
    results: list[MemoryResponse] = Field(default_factory=list)
    graph_results: list[dict] | None = None


class ContextResponse(BaseModel):
    """Organized context by category."""
    status: str = "ok"
    user_id: str
    project_id: str | None = None
    categories: dict[str, list[MemoryResponse]] = Field(default_factory=dict)
    # Authoritative org standards (visibility=standard). Always returned in
    # full and never paginated/truncated — clients render these as binding
    # directives that override personal preferences. Empty unless
    # STANDARDS_ENABLED. Kept out of `categories` so a large standard set can't
    # evict a caller's own recalled context.
    standards: list[MemoryResponse] = Field(default_factory=list)
    # Pagination over the combined (global + project) memory set. The page is
    # sorted newest-first, so `offset`/`limit` page deterministically.
    total: int = 0
    returned: int = 0
    offset: int = 0
    limit: int | None = None
    has_more: bool = False


class TaskAcceptedResponse(BaseModel):
    """Response for async write endpoints (202 Accepted)."""
    status: str = "accepted"
    task_id: str
    poll_url: str


class TaskStatusResponse(BaseModel):
    """Response for task status polling."""
    task_id: str
    status: str  # "queued", "processing", "completed", "failed", "not_found"
    result: dict | None = None
    error: str | None = None


class CategoryListResponse(BaseModel):
    status: str = "ok"
    categories: dict[str, str] = Field(default_factory=dict)
