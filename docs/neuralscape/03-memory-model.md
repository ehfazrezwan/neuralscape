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
- **`BulkDeleteRequest`** (166-175) — filtered bulk delete; `filter_null_category` restricts to memories missing a category. `include_shared` (default `False`) gates whether the caller's *shared* writes are removed: by default shared memories are preserved on every bulk path (team artifacts shouldn't be wipeable by one user's sweep, including via the MCP `delete_memories` tool an LLM agent can trigger).

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

## Memory model v2 (additive)

Released 2026-05-09 alongside the plugin's PostToolUse capture pipeline. v2 adds **seven optional fields** to `RawMemoryRequest`, `StoreMemoryRequest`, `MemoryResponse`, and the `mcp__plugin_neuralscape_neuralscape__remember` MCP tool. Existing v1 memories continue to render unchanged with these fields as `null` — **no migration required**, no data loss.

### v2 fields

| Field | Type | Constraint | Purpose |
|---|---|---|---|
| `domain` | `str \| None` | enum `DOMAIN_VOCAB` (default `general`) | High-level life-context grouping. Lets a teammate filter to research / meeting / writing rather than just coding. |
| `observation_type` | `str \| None` | enum `OBSERVATION_TYPE_VOCAB` | The *shape* of the memory, orthogonal to category. Most useful when the compile-observations skill writes it. |
| `concepts` | `list[str] \| None` | controlled vocab `CONCEPT_VOCAB`, ≤5 items | Cross-cutting tags. Better than free-form `tags` for filtered queries. |
| `source_type` | `str \| None` | enum `SOURCE_TYPE_VOCAB` | Provenance — `tool_extraction` for the new path, `conversation` for legacy compiler, `explicit` for direct API calls. |
| `related_memory_ids` | `list[str] \| None` | UUIDs, ≤10 | Lightweight graph linkage. The compile skill chains related observations from one work session. |
| `confidence` | `float \| None` | 0.0–1.0 | Extractor's self-rated confidence. Low-confidence memories deprioritized in search. |
| `expires_at` | `datetime \| None` | ISO 8601 | Optional expiry for short-lived `task_context` entries. Cleanup cron purges expired memories nightly. |

All seven are **optional** and stored under `metadata.<field>` in the Qdrant payload, matching the prefix convention from commit `6c94f20`. Validators in `schemas.py` reject unknown vocab values on write but accept null/missing on read.

### Controlled vocabularies

Defined in `neuralscape-service/schemas.py:104-126`:

```python
DOMAIN_VOCAB = {
    "coding", "research", "meeting", "writing", "ops", "personal", "general",
}

OBSERVATION_TYPE_VOCAB = {
    "bugfix", "feature", "refactor", "decision", "discovery",
    "gotcha", "pattern", "trade_off", "research_note",
    "meeting_outcome", "task_plan", "fact",
}

CONCEPT_VOCAB = {
    "how-it-works", "why-it-exists", "what-changed",
    "problem-solution", "gotcha", "pattern", "trade-off",
    "open-question", "next-step", "blocker",
}

SOURCE_TYPE_VOCAB = {
    "conversation", "tool_extraction", "explicit", "imported", "compiler",
}
```

### Search filters (v2)

`SearchMemoryRequest` gained three optional filters: `domain`, `observation_type`, `concepts`. The first two are exact-match; `concepts` does any-match (`metadata.concepts` ∩ supplied list). All three compose with the v1 `categories` and `scope` filters.

### Batch endpoint

`POST /v1/memories/raw/batch` accepts up to 50 `RawMemoryRequest` items in one call, dispatched as a single ARQ task. Reuses `service.store_raw()` per item so dedup and validation stay consistent. Useful for the compile-observations skill when a session yields multiple memories.

### Content-hash dedup on insert

`service.store_raw()` now performs a Qdrant scroll on `(user_id, metadata.scope, hash)` before insert. On hit, returns the existing memory unchanged — re-flushes from PostToolUse retries become idempotent. Lookup failures are non-fatal (the insert proceeds); a duplicate is preferred over a dropped write.

### Expiry purge cron

`expire_old_memories_cron` runs nightly at 03:15 (server time) and deletes memories whose `metadata.expires_at` is in the past. Matched memories are also cleaned from Graphiti via `_expire_graph_edges_for_memory`.

## Multi-user model (v2.2)

Released 2026-05-11. Adds a `visibility` axis orthogonal to scope:

- **`private`** (default): only the writing user (`owner_user_id`) reads it.
- **`shared`**: every authenticated user in this Neuralscape instance reads it.

This solves the personal-vs-team-knowledge split: each user has their own
private memory pool that persists across their sessions, plus access to a
shared knowledge pool that everyone on the team can contribute to.

### Per-category default visibility

When a write doesn't supply `visibility`, the server picks from this table:

| Category | Default visibility | Rationale |
|---|---|---|
| `preference` | private | Personal taste |
| `personal_fact` | private | Personal info |
| `technical_skill` | private | About the user, not the project |
| `domain_knowledge` | private | The user's accumulated learning |
| `task_context` | private | WIP, kept private until shipped |
| `tech_stack` | shared | Team should know what we use |
| `convention` | shared | Team norms |
| `architecture` | shared | Team structural decisions |
| `dependency` | shared | Team-wide visibility |
| `decision` | shared | Decisions affect the team |
| `interaction` | shared | Meeting outcomes etc. |
| `workflow` | shared | Team processes |
| `procedure` | shared | Team how-tos |

The map lives in `neuralscape-service/schemas.py:DEFAULT_VISIBILITY_FOR_CATEGORY`. Callers can always override via the explicit `visibility` field on write.

### Graphiti `group_id` format (multi-user)

The legacy `"global"` and `"project--{id}"` group_ids are replaced with:

| Memory shape | `group_id` |
|---|---|
| Private, no project | `user--{user_id}` |
| Private, project-scoped | `user--{user_id}--project--{project_id}` |
| Shared, no project | `shared` |
| Shared, project-scoped | `shared--project--{project_id}` |

The user namespace prevents the cross-user graph leakage that v2.1 still
had (Graphiti's group_id had no user component, so a search across
`"global"` returned every user's facts). The new format scopes private
facts to their writer while keeping shared facts in a single team-wide
namespace.

A search by user `alice` walks group_ids `["user--alice", "shared"]`
(plus the project-scoped variants when the request supplies
`project_id`).

### Dual-pool search

`MemoryService.search()` runs two queries and merges:

1. **Personal pool**: `m.search(user_id=caller, filters=...)` — mem0
   enforces user_id at the Qdrant layer. Returns the caller's own
   memories (any visibility).
2. **Shared pool**: direct `qdrant_client.query_points()` with
   `metadata.visibility=shared` — bypasses mem0's user_id namespacing
   because shared memories span multiple writers. (We use
   `query_points()` rather than the deprecated `.search()` because
   qdrant-client v1.13+ removed the latter; the response wraps hits
   in a `.points` attribute.)

Results are dedup'd by id (a caller's own shared write matches both
pools) and sorted by score. Filters available:

- `visibility="private"` → personal pool only
- `visibility="shared"` → shared pool only
- `include_shared=False` → personal pool only, even on broad queries

### Auth: HMAC-signed per-user tokens

Authentication moves from a single shared API key to per-user HMAC tokens:

```text
base64url({user_id, exp}).hmac_sha256(secret, payload_b64)
```

The signing secret is `NEURALSCAPE_USER_TOKEN_SECRET` (separate from the
legacy `NEURALSCAPE_API_KEY`). Issue tokens via
`scripts/issue_user_token.py --user <name> --days 30`. The
middleware (`auth.py:BearerAuthMiddleware`) verifies the HMAC, extracts
`user_id`, and attaches it to `request.state.user_id` — every v1 route
prefers that over any `user_id` value in the request body.

If a request supplies `user_id` in the body that disagrees with the
token's `user_id`, the server returns 400 to prevent confusion.

When `NEURALSCAPE_USER_TOKEN_SECRET` is unset but `NEURALSCAPE_API_KEY`
is set, the legacy shared-key path still works (body `user_id` is
trusted, a `X-Neuralscape-Deprecation` response header is added).

### Migration from v2.1

Existing memories have no `metadata.visibility` field, so the server
treats them as **private to their owner_user_id**. No accidental cross-
user reads. To make some categories visible to teammates after the fact:

```bash
python scripts/bulk_promote_visibility.py \
  --owner ehfaz --category tech_stack --to shared --apply
```

Graph entries from pre-v2.2 sit under `"global"` / `"project--..."`
and are invisible to the new search until you run:

```bash
python scripts/migrate_graph_groups.py --owner ehfaz --apply
```

Both scripts default to `--dry-run`; pass `--apply` to actually write.

## Related

- [memory-service-core](./04-memory-service-core.md)
- [storage-backends](./06-storage-backends.md)
- [llm-extraction](./05-llm-extraction.md)
- [plugin-system](./09-plugin-system.md)
- [00-overview](./00-overview.md)
