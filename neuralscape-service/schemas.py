"""Schemas for neuralscape-service: enums, category taxonomy, request/response models."""

from enum import Enum

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────


class MemoryScope(str, Enum):
    GLOBAL = "global"
    PROJECT = "project"


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
# Category taxonomy
# ──────────────────────────────────────────────

MEMORY_CATEGORIES: dict[str, str] = {
    "preference": "User preferences: language, editor, code style, communication style",
    "personal_fact": "Personal details: name, timezone, role, team",
    "technical_skill": "Known technologies, proficiency levels",
    "domain_knowledge": "Industry/domain-specific knowledge",
    "tech_stack": "Project technology choices",
    "convention": "Coding conventions, naming, file structure",
    "architecture": "Design decisions, module boundaries, API patterns",
    "dependency": "Packages, versions, compatibility notes",
    "decision": "Decisions made with rationale",
    "interaction": "Notable past interactions/events",
    "workflow": "Git flow, CI/CD, deployment, review process",
    "procedure": "Step-by-step how-to patterns",
    "task_context": "Current task, recent changes, blockers",
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
    user_id: str = Field(min_length=1, max_length=100, pattern=_ID_PATTERN)
    project_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    agent_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    run_id: str | None = Field(default=None, max_length=100)


class RawMemoryRequest(BaseModel):
    """Store a single pre-categorized fact (no LLM extraction)."""
    content: str = Field(
        description="The memory content to store",
        min_length=1,
        max_length=10000,
    )
    user_id: str = Field(min_length=1, max_length=100, pattern=_ID_PATTERN)
    category: str = Field(description="Memory category (must be one of MEMORY_CATEGORIES)")
    scope: str = Field(default="global", description="'global' or 'project'")
    project_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    tags: list[str] | None = Field(default=None, max_length=20)
    agent_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    run_id: str | None = Field(default=None, max_length=100)


class SearchMemoryRequest(BaseModel):
    """Semantic search across memories."""
    query: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(min_length=1, max_length=100, pattern=_ID_PATTERN)
    project_id: str | None = Field(default=None, max_length=100, pattern=_ID_PATTERN)
    categories: list[str] | None = Field(default=None, max_length=13)
    scope: str | None = None
    limit: int = Field(default=10, ge=1, le=100)


class GraphSearchRequest(BaseModel):
    """Knowledge graph search."""
    query: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(min_length=1, max_length=100, pattern=_ID_PATTERN)
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
    user_id: str = Field(min_length=1, max_length=100, pattern=_ID_PATTERN)
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
    """Single memory in response."""
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
