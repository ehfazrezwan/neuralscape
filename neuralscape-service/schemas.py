"""Schemas for neuralscape-service: enums, category taxonomy, request/response models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


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

    Visibility is orthogonal to scope: a private memory can be `global`
    (cross-project for that user) or `project`-scoped; a shared memory can
    be cross-project or project-scoped (visible to anyone who searches that
    project).
    """
    PRIVATE = "private"
    SHARED = "shared"


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
}

CONCEPT_VOCAB: set[str] = {
    "how-it-works", "why-it-exists", "what-changed",
    "problem-solution", "gotcha", "pattern", "trade-off",
    "open-question", "next-step", "blocker",
}

SOURCE_TYPE_VOCAB: set[str] = {
    "conversation", "tool_extraction", "explicit", "imported", "compiler",
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
        description="Multi-user model: who can read this memory. Defaults per-category.",
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
        description="Multi-user model: 'private' (owner only) or 'shared' (team-wide). Defaults per-category.",
    )

    # Memory-model v2 (all optional)
    domain: str | None = Field(default=None, description="High-level life-context domain")
    observation_type: str | None = Field(default=None, description="Shape of observation, orthogonal to category")
    concepts: list[str] | None = Field(default=None, max_length=5, description="Cross-cutting controlled-vocab tags")
    source_type: str | None = Field(default=None, description="Provenance of this memory")
    related_memory_ids: list[str] | None = Field(default=None, max_length=10, description="UUIDs of related memories")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Extractor's self-rated confidence")
    expires_at: datetime | None = Field(default=None, description="Optional expiry for short-lived memories")

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


class RawMemoryBatchRequest(BaseModel):
    """Store multiple pre-categorized facts in one request (memory-model v2)."""
    memories: list[RawMemoryRequest] = Field(min_length=1, max_length=50)


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
    categories: list[str] | None = Field(default=None, max_length=13)
    scope: str | None = None
    limit: int = Field(default=10, ge=1, le=100)

    # Memory-model v2 filters (all optional)
    domain: str | None = Field(default=None, description="Filter by domain")
    observation_type: str | None = Field(default=None, description="Filter by observation_type")
    concepts: list[str] | None = Field(default=None, max_length=5, description="Filter by concept tags (any-match)")

    # Multi-user pool selection
    visibility: MemoryVisibility | None = Field(
        default=None,
        description="Restrict results to one pool: 'private' (yours only) or 'shared'. Default: both.",
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


class UpdateMemoryRequest(BaseModel):
    """Update a memory's content or category."""
    content: str | None = Field(default=None, max_length=10000)
    category: str | None = None
    tags: list[str] | None = Field(default=None, max_length=20)


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
