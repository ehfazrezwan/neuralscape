---
title: Memory Model
date: 2026-05-06
tags: [reference, neuralscape, schemas, categories, scopes]
source: handwritten
---

# Memory Model

## Overview

The neuralscape memory model is structured along three orthogonal axes: **13 categories** organized into **5 type groups**, with **2 scopes** (`global` and `project`). Every memory carries a category that determines its conceptual role and a scope that determines its visibility boundary. Categories also map to a fixed Obsidian vault folder so the on-disk mirror stays predictable. This page is the canonical schema reference; for the write/read pipelines that consume these values, see [memory-service-core](./04-memory-service-core.md).

## Categories

The full taxonomy is defined in `neuralscape-service/schemas.py:18-40` (enum) and `neuralscape-service/schemas.py:47-61` (descriptions). Default scope is computed by `default_scope_for_category` (`schemas.py:73-79`).

| # | Category | Type Group | Default Scope | Description |
|---|----------|-----------|---------------|-------------|
| 1 | `preference` | Semantic | `global` | Language, editor, code style, communication preferences |
| 2 | `personal_fact` | Semantic | `global` | Name, timezone, role, team |
| 3 | `technical_skill` | Semantic | `global` | Known technologies and proficiency levels |
| 4 | `domain_knowledge` | Semantic | `global` | Industry/domain-specific knowledge |
| 5 | `tech_stack` | Project | `project` | Project technology choices |
| 6 | `convention` | Project | `project` | Coding conventions, naming, file structure |
| 7 | `architecture` | Project | `project` | Design decisions, module boundaries, API patterns |
| 8 | `dependency` | Project | `project` | Packages, versions, compatibility notes |
| 9 | `decision` | Episodic | flexible | Decisions made with rationale |
| 10 | `interaction` | Episodic | flexible | Notable past interactions/events |
| 11 | `workflow` | Procedural | flexible | Git flow, CI/CD, deployment, review process |
| 12 | `procedure` | Procedural | flexible | Step-by-step how-to patterns |
| 13 | `task_context` | Working | flexible | Current task, recent changes, blockers |

Flexible-default categories resolve to `global` unless the caller supplies `project_id` or an explicit scope. Category descriptions surface to the LLM extractor in `neuralscape-service/prompts.py:31-43`.

## Type taxonomy

- **Semantic** — user-wide facts and preferences. Always `global`. Persist across all projects and sessions.
- **Project** — codebase-specific tech, conventions, design. Require `project_id`; isolated per project.
- **Episodic** — events with a temporal aspect (decisions made, interactions logged). Flexible scope.
- **Procedural** — workflows and step-by-step how-tos. Flexible scope.
- **Working** — short-lived task context, blockers. Implicitly ephemeral; flexible scope.

## Scope rules

Three sets in `schemas.py:64-70` partition the 13 categories:

```python
GLOBAL_CATEGORIES   = {"preference", "personal_fact", "technical_skill", "domain_knowledge"}
PROJECT_CATEGORIES  = {"tech_stack", "convention", "architecture", "dependency"}
FLEXIBLE_CATEGORIES = {"decision", "interaction", "workflow", "procedure", "task_context"}
```

The service layer overrides the schema default in `neuralscape-service/memory_service.py:500-520`:

- `PROJECT_CATEGORIES` require `project_id`. If missing, `_infer_project_id()` attempts to extract one from content; on failure the memory falls back to `global` with a warning.
- If `project_id` is provided **and** the category is not in `GLOBAL_CATEGORIES`, scope is forced to `project`.
- `FLEXIBLE_CATEGORIES` honor the caller-supplied scope, defaulting to `global`.
- `GLOBAL_CATEGORIES` always remain `global`, even when `project_id` is present.

## group_id format

Both Graphiti's `group_id` and Qdrant's metadata filter use the same scoping token, built by `_build_group_id` and `_get_group_ids` in `memory_service.py:175-187`:

- Global: `"global"`
- Project: `"project--{project_id}"` — note the **double hyphen**, not a colon

This convention was standardized in commit `2edd9b5` (`fix(graphiti): align group_id format to use double-hyphen for project scoping`). Search with `project_id` queries both `global` and `project--{project_id}` group_ids and merges results.

## Vault folder mapping

`CATEGORY_VAULT_PATHS` (`schemas.py:83-97`) maps each category to its Obsidian vault folder:

```python
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
```

This dictionary is mirrored in `neuralscape-plugin/src/types.ts:25-147`. The two sources must stay synchronized — see [plugin-system](./09-plugin-system.md) for plugin-side handling.

## Memory IDs and metadata

- **ID**: UUID v4 generated at insert time via `str(uuid.uuid4())` (`memory_service.py:404, 522`).
- **Content hash**: `hashlib.md5(content.encode()).hexdigest()` (32-char hex), stored in payload for dedup.
- **Qdrant payload metadata** (`memory_service.py:414-424, 530-537`):

```python
"metadata": {
    "scope": "global" | "project",
    "category": str,
    "project_id": str | None,
    "agent_id": str | None,
    "run_id": str | None,
    "source": "conversation" | "explicit" | "vector" | "graph",
    "tags": list[str] | None,
}
```

> [!important] Filter prefix gotcha
> Qdrant filters MUST use the `metadata.` prefix (e.g., `metadata.scope`, `metadata.category`) because these fields are stored nested under the payload's `metadata` key. This was fixed in commit `6c94f20`. See [storage-backends](./06-storage-backends.md) for filter construction.

## Pydantic schemas

Request models (`schemas.py:109-175`):

- **`StoreMemoryRequest`** (109-118) — `messages` (list of `{role, content}`, ≤500), `user_id`, optional `project_id`/`agent_id`/`run_id`. LLM extraction path.
- **`RawMemoryRequest`** (121-134) — pre-categorized single fact: `content` (1-10000 chars), `user_id`, `category`, `scope` (default `"global"`), optional `project_id`/`tags` (≤20)/`agent_id`/`run_id`.
- **`SearchMemoryRequest`** (137-144) — `query` (1-2000), `user_id`, optional `project_id`, `categories` (≤13), `scope`, `limit` (1-100, default 10).
- **`GraphSearchRequest`** (147-156) — Graphiti-only search with optional `search_config` override.
- **`UpdateMemoryRequest`** (159-163) — partial update of `content`/`category`/`tags`.
- **`BulkDeleteRequest`** (166-175) — filtered bulk delete; `filter_null_category` restricts to memories missing a category.

Response models (`schemas.py:183-234`):

- **`MemoryResponse`** (183-194) — `id`, `memory`, `category`, `scope`, `project_id`, `tags`, `score`, `created_at`, `updated_at`, `source`.
- **`StoreMemoryResponse`**, **`SearchMemoryResponse`**, **`ContextResponse`**, **`TaskAcceptedResponse`** (async 202 acks), **`TaskStatusResponse`** (poll), **`CategoryListResponse`**.

Write-path internals (extraction, dedup, persistence) live in [memory-service-core](./04-memory-service-core.md) and [llm-extraction](./05-llm-extraction.md).

## Validation

- **ID pattern** (`schemas.py:106`): `_ID_PATTERN = r"^[a-zA-Z0-9_.\-]+$"` applied to `user_id`, `project_id`, and `agent_id` (not `run_id`).
- **Length limits**: `user_id` 1-100, `project_id` ≤100, `agent_id` ≤100, `content` 1-10000, `query` 1-2000, `messages` list ≤500, `tags` ≤20, `limit` 1-100, context `max_chars` 500-32000.
- **No `@field_validator` decorators** in `schemas.py` — category validation is enforced at the service layer (`prompts.py:69-89`), which falls back to `personal_fact` on unknown input rather than raising.

## Legacy vs v1 surfaces

The FastAPI app exposes two parallel surfaces. Legacy root paths (`/memories`, `/search`) are thin mem0 passthroughs: no scoping, no categories, arbitrary metadata dicts. The `/v1/*` endpoints are the structured surface — explicit `category`, `scope`, `project_id`, typed metadata, and async-by-default writes returning `202` plus a `task_id` to poll. New integrations should target v1; see [service-architecture](./02-service-architecture.md) for routing details.

## Related

- [memory-service-core](./04-memory-service-core.md)
- [storage-backends](./06-storage-backends.md)
- [llm-extraction](./05-llm-extraction.md)
- [plugin-system](./09-plugin-system.md)
- [00-overview](./00-overview.md)
