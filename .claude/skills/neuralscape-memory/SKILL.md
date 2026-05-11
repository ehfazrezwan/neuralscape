---
name: neuralscape-memory
description: Use the Neuralscape memory layer to recall and store memories about the user and their projects. Use this at session start to load context, when you learn new facts, and when the user asks about preferences, conventions, or past decisions. Works via MCP tools or REST API at localhost:8199. Memory writes are async (processed by background workers via ARQ + Redis). Multi-user: each authenticated user has a private memory pool plus a shared team pool — pick the right visibility per write.
---

# Neuralscape Memory Layer

Neuralscape provides persistent, categorized memory for AI agents. It remembers user preferences, project conventions, technical decisions, and learned facts across sessions. Since v2.2 it's **multi-user**: each authenticated user has their own private memory pool plus access to a team-wide shared pool. The pool a memory lives in is determined by its `visibility` (private | shared), with sensible per-category defaults.

## Architecture

Memory writes are **asynchronous**. When you store a memory (via MCP or REST), the request is enqueued to Redis and processed by a separate ARQ worker. This means:

- `remember` and `remember_conversation` return immediately with a `task_id` (fire-and-forget by default)
- Set `wait: true` to block until the memory is fully stored (useful for end-of-session memory dumps)
- Read operations (`recall_memories`, `get_project_context`, `search_knowledge_graph`, `list_memories`) are **synchronous** and return results immediately

## Multi-user model (v2.2+)

Every memory carries a `visibility` and an `owner_user_id`:

- **`private`** — only the writing user can read it. Personal preferences, identity facts, in-progress task context.
- **`shared`** — every authenticated user in this Neuralscape instance can read it. Team knowledge: tech stack, conventions, architecture decisions.

**Per-category defaults** (you almost never need to override):

| Default | Categories |
|---|---|
| `private` | `preference`, `personal_fact`, `technical_skill`, `domain_knowledge`, `task_context` |
| `shared` | `tech_stack`, `convention`, `architecture`, `dependency`, `decision`, `interaction`, `workflow`, `procedure` |

The default does the right thing ~80% of the time. Override `visibility` only when:
- A `preference` is actually a team norm worth sharing (rare) → `shared`
- A team `decision` is sensitive WIP nobody else should see yet → `private`
- A `task_context` you want a teammate to pick up → `shared`

**Search by default** returns BOTH pools (the caller's private + the shared team pool), merged by relevance. Use `visibility: "private"` or `visibility: "shared"` to narrow, or `include_shared: false` to skip the shared pool entirely.

**Identity in writes**: the `user_id` you pass becomes `owner_user_id` on the stored memory. Search results surface `owner_user_id` so clients can tell who authored a shared hit.

## Auth (when running against a deployed service)

The server reads `user_id` from a per-user HMAC token in the `Authorization: Bearer …` header, NOT from the request body. If you pass a `user_id` in the body that disagrees with the token, the server returns 400. The simplest rule: always pass the same `user_id` you've been told to use; if it disagrees with the token's encoded user_id, fail fast.

The plugin (when installed in Claude Code) handles auth transparently — your MCP tool calls just include `user_id` and the plugin's `.mcp.json` forwards the bearer token from the user's configured `API_KEY`.

## When to Use Memory

- **Session start**: Call `get_project_context` or `recall_memories` to load what you know about this user and project before doing any work.
- **During work**: Call `recall_memories` when you need context about a specific topic (e.g., "what indentation style does this user prefer?").
- **After learning something**: Call `remember` when the user tells you a preference, makes a decision, or reveals something about their project that future sessions should know. Default fire-and-forget is fine here.
- **End of conversation**: Call `remember_conversation` with `wait: true` to bulk-extract all notable facts and confirm they were stored.

## Identifiers in every call

Always pass:
- `user_id` — the user's identifier (must match the bearer token's encoded user_id when auth is on).
- `project_id` — the current project slug when working in a project context. Project-scoped categories (`tech_stack`, `convention`, `architecture`, `dependency`) **require** this.

## MCP Tools

### 1. `recall_memories` - Search memories (sync)

Search across the caller's private pool AND the shared team pool, dedup'd and merged by relevance. When `project_id` is provided, also searches project-scoped variants of both pools.

```json
{
  "query": "indentation style preferences",
  "user_id": "ehfaz",
  "project_id": "neuralscape-graphiti",
  "categories": ["preference", "convention"],
  "limit": 10
}
```

**Multi-user filters (optional):**

- `visibility: "private"` — only the caller's own memories.
- `visibility: "shared"` — only the team pool.
- `include_shared: false` — skip the shared pool entirely (caller's private only). Default: `true`.

Results include `owner_user_id` so you can tell who authored a shared hit, and `source: "graph" | "vector"` so you know which subsystem returned it (graph wins on conflict — it carries Graphiti's contradiction-resolved state).

### 2. `remember` - Store a single fact (async)

Store one categorized fact. Pick the most specific category from the taxonomy below. The system auto-assigns scope based on category AND visibility based on the per-category default (see Multi-user model above). Returns immediately with `task_id` by default.

```json
{
  "content": "Prefers 4-space indentation in Python files",
  "user_id": "ehfaz",
  "category": "preference"
}
```

Response (fire-and-forget):
```json
{"status": "accepted", "task_id": "abc123"}
```

For project-specific facts, include `project_id`:

```json
{
  "content": "Uses FastAPI 0.115 with Pydantic v2 for the backend",
  "user_id": "ehfaz",
  "category": "tech_stack",
  "project_id": "neuralscape-graphiti"
}
```

**Override visibility** when the default isn't what you want (rare):

```json
{
  "content": "Team decision: defer the multi-team rollout until 2027",
  "user_id": "ehfaz",
  "category": "decision",
  "project_id": "neuralscape",
  "visibility": "private"
}
```

**Memory-model v2 fields** (optional but recommended for richer recall):

| Field | Purpose | Values |
|---|---|---|
| `domain` | High-level life context | `coding` \| `research` \| `meeting` \| `writing` \| `ops` \| `personal` \| `general` |
| `observation_type` | Shape of the observation (orthogonal to category) | `bugfix` \| `feature` \| `refactor` \| `decision` \| `discovery` \| `gotcha` \| `pattern` \| `trade_off` \| `research_note` \| `meeting_outcome` \| `task_plan` \| `fact` |
| `concepts` | Cross-cutting tags (1–5 items) | `how-it-works`, `why-it-exists`, `what-changed`, `problem-solution`, `gotcha`, `pattern`, `trade-off`, `open-question`, `next-step`, `blocker` |
| `source_type` | Provenance | `conversation` \| `tool_extraction` \| `explicit` \| `imported` \| `compiler` |
| `confidence` | Your 0.0–1.0 self-rating | float |
| `expires_at` | ISO 8601; memory is purged after this | `"2026-12-31T00:00:00Z"` |
| `related_memory_ids` | UUIDs of related memories (graph linkage) | array, max 10 |

To wait for confirmation, set `wait: true`:

```json
{
  "content": "Prefers 4-space indentation",
  "user_id": "ehfaz",
  "category": "preference",
  "wait": true
}
```

Response (waited):
```json
{"status": "completed", "task_id": "abc123", "result": {"memories": [...]}}
```

### 3. `remember_conversation` - Bulk extract from conversation (async)

Pass conversation messages and the system uses Gemini to identify facts, categorize them, and store each one automatically. Returns immediately with `task_id` by default.

```json
{
  "messages": [
    {"role": "user", "content": "I switched our database from Postgres to MongoDB"},
    {"role": "assistant", "content": "I'll update the ORM queries to use PyMongo"}
  ],
  "user_id": "ehfaz",
  "project_id": "my-project",
  "wait": true
}
```

### 4. `get_project_context` - Bootstrap session context (sync)

Load all global user preferences plus project-specific memories, organized by category. Call this at session start.

```json
{
  "user_id": "ehfaz",
  "project_id": "neuralscape-graphiti"
}
```

Returns:
```json
{
  "categories": {
    "preference": [{"memory": "Prefers tabs", ...}],
    "tech_stack": [{"memory": "Uses FastAPI with Graphiti backend", ...}],
    "convention": [{"memory": "Uses snake_case for Python, camelCase for JS", ...}]
  }
}
```

### 5. `search_knowledge_graph` - Entity/relationship search (sync)

Search the Graphiti knowledge graph for structured relationships between entities.

```json
{
  "query": "FastAPI",
  "user_id": "ehfaz",
  "project_id": "neuralscape-graphiti",
  "limit": 10
}
```

### 6. `list_memories` - Inspect stored memories (sync)

List memories with optional filters. Use to verify what's been stored or audit memory contents.

```json
{
  "user_id": "ehfaz",
  "scope": "global",
  "category": "preference",
  "limit": 50
}
```

### 7. `delete_memories` - Remove memories (sync)

Delete a specific memory by ID, or bulk delete by filter. Use with caution.

```json
{
  "user_id": "ehfaz",
  "memory_id": "abc-123"
}
```

Or bulk delete:
```json
{
  "user_id": "ehfaz",
  "scope": "project",
  "project_id": "old-project"
}
```

**Shared-pool safety default (v2.2+):** Bulk delete only removes the caller's **private** writes by default. Shared (team) memories the caller authored are preserved — they're team artifacts and one user's sweep shouldn't wipe team knowledge. The result message reports what was preserved: `"Deleted 12 memories (preserved 4 shared)"`.

To also remove the caller's shared writes (admin-style nuke), opt in explicitly:

```json
{
  "user_id": "ehfaz",
  "include_shared": true
}
```

**Always confirm with the user before passing `include_shared: true`** — it's destructive and irreversible. Single-memory delete by ID is unaffected by this default.

## REST API (alternative to MCP)

The service runs at `http://localhost:8199`. All v1 endpoints accept `user_id` (token-derived when an HMAC bearer token is present; body-derived otherwise). Token and body `user_id` must agree or the server returns 400. v1 endpoints also accept `visibility` and `include_shared` filters where they apply (search, list, bulk-delete) — same semantics as the MCP tools above.

**Write endpoints return 202 Accepted** with a `task_id` for polling:

| Action | Method | Endpoint | Response |
|---|---|---|---|
| Store via LLM extraction | `POST` | `/v1/memories` | 202 `{task_id, poll_url}` |
| Store single fact | `POST` | `/v1/memories/raw` | 202 `{task_id, poll_url}` |
| Poll task status | `GET` | `/v1/memories/status/{task_id}` | `{status, result, error}` |

**Read endpoints return 200 with results immediately:**

| Action | Method | Endpoint |
|---|---|---|
| Semantic search | `POST` | `/v1/search` |
| Graph search | `POST` | `/v1/graph/search` |
| Project + global context | `GET` | `/v1/context/{project_id}?user_id=...` |
| Global context only | `GET` | `/v1/context/global?user_id=...` |
| List memories | `GET` | `/v1/memories?user_id=...&scope=...&category=...` |
| Get single memory | `GET` | `/v1/memories/{id}` |
| Update memory | `PUT` | `/v1/memories/{id}` |
| Delete memory | `DELETE` | `/v1/memories/{id}` |
| Bulk delete | `DELETE` | `/v1/memories` (body: `{user_id, scope?, category?, project_id?}`) |
| List categories | `GET` | `/v1/categories` |
| Graph nodes | `GET` | `/v1/graph/nodes?user_id=...&project_id=...` |
| Graph edges | `GET` | `/v1/graph/edges?user_id=...&project_id=...` |
| Graph episodes | `GET` | `/v1/graph/episodes?user_id=...` |
| Graph communities | `GET` | `/v1/graph/communities?user_id=...` |

### Task status values

When polling `/v1/memories/status/{task_id}`:

| Status | Meaning |
|---|---|
| `queued` | Task is in the Redis queue, waiting for a worker |
| `processing` | Worker is actively processing (LLM extraction, storage) |
| `completed` | Done. `result` field contains stored memories |
| `failed` | Error occurred. `error` field has details |

## Category Taxonomy

When storing a memory, always assign the most specific category. The system uses categories to control scope defaults and enable filtered retrieval.

### Global categories (default scope: `global`)

These are about the **user**, not any specific project.

| Category | Use for | Examples |
|---|---|---|
| `preference` | User preferences for tools, style, communication | "Prefers dark mode", "Wants concise responses", "Uses vim keybindings", "Prefers tabs over spaces" |
| `personal_fact` | Personal details and identity | "Name is Ehfaz", "Located in Dhaka, timezone UTC+6", "Works as a backend engineer", "On the platform team" |
| `technical_skill` | Technologies the user knows and proficiency | "Expert in Python 3.12", "Learning Rust", "Familiar with Neo4j and graph databases", "Proficient in TypeScript" |
| `domain_knowledge` | Industry/domain expertise | "Specializes in NLP pipelines", "Understands HIPAA compliance requirements", "Background in fintech" |

### Project categories (default scope: `project`, requires `project_id`)

These are about a **specific project**. Always include `project_id`.

| Category | Use for | Examples |
|---|---|---|
| `tech_stack` | Frameworks, languages, databases, tools used | "Uses FastAPI 0.115 with Python 3.12", "PostgreSQL 16 for primary database", "Deployed on AWS ECS" |
| `convention` | Coding conventions, naming, file organization | "Uses snake_case for Python, camelCase for TypeScript", "Components in src/components/", "Tests mirror source structure in tests/" |
| `architecture` | Design decisions, module boundaries, patterns | "Uses hexagonal architecture", "API gateway pattern with separate auth service", "Event-driven via Redis pub/sub" |
| `dependency` | Package versions, compatibility, constraints | "Pinned pydantic to v2.x, incompatible with v1", "Uses mem0ai[graphiti] with local editable installs", "graphiti-core requires Neo4j 5.x+" |

### Flexible categories (can be either scope)

These can be global or project-specific. Include `project_id` when the fact relates to a specific project.

| Category | Use for | Examples |
|---|---|---|
| `decision` | Decisions made with rationale | "Chose Qdrant over Pinecone for on-disk persistence and no cloud dependency", "Decided to use LLM extraction in service layer instead of mem0 built-in" |
| `interaction` | Notable past interactions or events | "Debugged a Neo4j connection pooling issue on 2026-02-15", "User reported search returning stale results — fixed by adding temporal invalidation" |
| `workflow` | Git flow, CI/CD, deployment, review | "Uses trunk-based development with short-lived feature branches", "CI runs pytest then ruff on every push", "Deploys via GitHub Actions to staging first" |
| `procedure` | Step-by-step how-to patterns | "To add a new MCP tool: define in mcp_server.py list_tools, add handler in call_tool, add test in test_mcp_tools.py", "Database migration: create script, test locally, run in staging, then prod" |
| `task_context` | Current task, recent changes, blockers | "Currently refactoring the auth module", "Blocked on Neo4j license upgrade", "Last session: implemented v1 search endpoint" |

## Categorization Rules

1. Pick the **most specific** category. "Uses Python 3.12" about a person is `technical_skill`. "Uses Python 3.12" about a project is `tech_stack`.
2. If a fact could be multiple categories, prefer: project-specific > episodic > semantic.
3. `preference` is for **how the user wants things done**. `convention` is for **how a project actually does things**.
4. `decision` should include **rationale** ("chose X because Y"), not just the outcome.
5. `task_context` is ephemeral — use it for what's happening now, not permanent facts.

## Scoping Rules

- **Global** categories (`preference`, `personal_fact`, `technical_skill`, `domain_knowledge`) are always stored with `scope=global` regardless of whether `project_id` is passed.
- **Project** categories (`tech_stack`, `convention`, `architecture`, `dependency`) require `project_id` and default to `scope=project`.
- **Flexible** categories default to `global` but switch to `project` when `project_id` is provided.
- When searching with `project_id`, the system **always searches both global and project scope** and merges results by relevance.

## Writing Good Memory Content

Each memory should be a **standalone, specific, factual sentence** that makes sense without conversation context.

**Good:**
- "Prefers 4-space indentation in Python, 2-space in YAML"
- "neuralscape-graphiti uses FastAPI 0.115 with Pydantic v2 and Graphiti temporal knowledge graph"
- "Chose Qdrant over Pinecone because on-disk mode requires no external cloud service"

**Bad:**
- "Uses Python" (too vague)
- "The thing we discussed" (no context)
- "Yes" (not standalone)
- "Prefers good code" (not actionable)
