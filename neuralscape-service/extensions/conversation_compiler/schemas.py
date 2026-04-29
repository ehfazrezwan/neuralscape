"""Pydantic request/response models for the conversation-compiler extension."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Extraction models
# ──────────────────────────────────────────────


class ExtractedFact(BaseModel):
    """A single fact extracted from a conversation turn."""

    category: str = Field(description="Fact category (decision, preference, etc.)")
    content: str = Field(description="The extracted fact text")
    project_id: Optional[str] = Field(default=None, description="Inferred project if any")
    tags: list[str] = Field(default_factory=list)


class FlushResult(BaseModel):
    """Result of flushing (extracting from) a conversation turn."""

    session_id: str
    timestamp: str
    facts_extracted: int = 0
    facts: list[ExtractedFact] = Field(default_factory=list)
    daily_log_path: Optional[str] = None
    category_paths: list[str] = Field(default_factory=list)
    memories_stored: int = 0


# ──────────────────────────────────────────────
# Compilation models
# ──────────────────────────────────────────────


class CompiledArticle(BaseModel):
    """A single article produced by compilation."""

    path: str = Field(description="Relative path within the vault")
    title: str
    article_type: str = Field(description="session | project | decision | research")
    created: bool = Field(default=False, description="True if newly created, False if updated")


class CompileResult(BaseModel):
    """Result of a compilation run."""

    date: str
    articles: list[CompiledArticle] = Field(default_factory=list)
    entries_compiled: int = 0
    dedup_triggered: bool = False


# ──────────────────────────────────────────────
# Lint models
# ──────────────────────────────────────────────


class LintFinding(BaseModel):
    """A single lint finding."""

    check: str = Field(description="Name of the check that found this issue")
    severity: str = Field(default="warning", description="info | warning | error")
    message: str
    file: Optional[str] = None
    suggestion: Optional[str] = None


class LintResult(BaseModel):
    """Result of a lint run."""

    findings: list[LintFinding] = Field(default_factory=list)
    checks_run: int = 0
    files_scanned: int = 0


# ──────────────────────────────────────────────
# Query models
# ──────────────────────────────────────────────


class QueryResult(BaseModel):
    """Result of a knowledge base query."""

    answer: str
    sources: list[str] = Field(default_factory=list, description="Vault files used")
    filed_back: Optional[str] = Field(
        default=None, description="Path if the answer was filed back to the vault"
    )


# ──────────────────────────────────────────────
# API request models
# ──────────────────────────────────────────────


class FlushRequest(BaseModel):
    """Request body for POST /flush."""

    user_message: str = Field(min_length=1, max_length=50000)
    assistant_response: str = Field(min_length=1, max_length=50000)
    session_id: str = Field(min_length=1, max_length=200)
    channel: str = Field(default="api", max_length=50)
    timestamp: Optional[str] = None
    project_id: Optional[str] = None
    user_id: str = Field(min_length=1, max_length=100)


class CompileRequest(BaseModel):
    """Request body for POST /compile."""

    date: Optional[str] = Field(
        default=None,
        description="ISO date (YYYY-MM-DD) to compile. None = all pending.",
    )
    user_id: str = Field(min_length=1, max_length=100)


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(min_length=1, max_length=5000)
    file_back: bool = Field(default=False, description="Whether to save the answer to the vault")
    user_id: str = Field(min_length=1, max_length=100)


class LintRequest(BaseModel):
    """Request body for POST /lint."""

    structural_only: bool = Field(
        default=False,
        description="If true, skip LLM-powered checks (contradictions, data gaps)",
    )


class StatusResponse(BaseModel):
    """Response for GET /status."""

    extension: str = "conversation-compiler"
    status: str = "ok"
    last_flush: Optional[str] = None
    last_compile: Optional[str] = None
    article_count: int = 0
    daily_log_count: int = 0
    vault_path: str = ""
