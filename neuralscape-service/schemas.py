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


# ──────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────


class StoreMemoryRequest(BaseModel):
    """Store memories from conversation via LLM extraction."""
    messages: list[dict] = Field(description="Messages to extract memories from (list of {role, content} dicts)")
    user_id: str
    project_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None


class RawMemoryRequest(BaseModel):
    """Store a single pre-categorized fact (no LLM extraction)."""
    content: str = Field(description="The memory content to store")
    user_id: str
    category: str = Field(description="Memory category (must be one of MEMORY_CATEGORIES)")
    scope: str = Field(default="global", description="'global' or 'project'")
    project_id: str | None = None
    tags: list[str] | None = None
    agent_id: str | None = None
    run_id: str | None = None


class SearchMemoryRequest(BaseModel):
    """Semantic search across memories."""
    query: str
    user_id: str
    project_id: str | None = None
    categories: list[str] | None = None
    scope: str | None = None
    limit: int = Field(default=10, ge=1, le=100)


class GraphSearchRequest(BaseModel):
    """Knowledge graph search."""
    query: str
    user_id: str
    project_id: str | None = None
    limit: int = Field(default=10, ge=1, le=100)
    search_config: dict | None = Field(
        default=None,
        description="Optional SearchConfig dict to override default hybrid search",
    )


class UpdateMemoryRequest(BaseModel):
    """Update a memory's content or category."""
    content: str | None = None
    category: str | None = None
    tags: list[str] | None = None


class BulkDeleteRequest(BaseModel):
    """Bulk delete memories with filters."""
    user_id: str
    scope: str | None = None
    category: str | None = None
    project_id: str | None = None


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


class CategoryListResponse(BaseModel):
    status: str = "ok"
    categories: dict[str, str] = Field(default_factory=dict)
