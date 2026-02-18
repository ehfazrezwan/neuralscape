---
name: neuralscape-memory
description: Use the Neuralscape memory layer to recall and store memories about the user and their projects. Use this at session start to load context, when you learn new facts, and when the user asks about preferences, conventions, or past decisions. Works via MCP tools or REST API at localhost:8199.
---

# Neuralscape Memory Layer

Neuralscape provides persistent, categorized memory for AI agents. It remembers user preferences, project conventions, technical decisions, and learned facts across sessions. Every agent sharing the same `user_id` can read and write to the same memory.

## When to Use Memory

- **Session start**: Call `get_project_context` or `recall_memories` to load what you know about this user and project before doing any work.
- **During work**: Call `recall_memories` when you need context about a specific topic (e.g., "what indentation style does this user prefer?").
- **After learning something**: Call `remember` when the user tells you a preference, makes a decision, or reveals something about their project that future sessions should know.
- **End of conversation**: Call `remember_conversation` to bulk-extract all notable facts from a productive conversation.

## Identity

When calling any memory tool or endpoint, always pass:
- `user_id`: The user's identifier (e.g., `"ehfaz"`)
- `project_id`: The current project slug when working in a project context (e.g., `"neuralscape-graphiti"`)

## MCP Tools

### 1. `recall_memories` - Search memories

Search across global and project-specific memories. When `project_id` is provided, searches both scopes and merges results by relevance.

```json
{
  "query": "indentation style preferences",
  "user_id": "ehfaz",
  "project_id": "neuralscape-graphiti",
  "categories": ["preference", "convention"],
  "limit": 10
}
```

### 2. `remember` - Store a single fact

Store one categorized fact. Pick the most specific category from the taxonomy below. The system auto-assigns scope based on category.

```json
{
  "content": "Prefers 4-space indentation in Python files",
  "user_id": "ehfaz",
  "category": "preference"
}
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

### 3. `remember_conversation` - Bulk extract from conversation

Pass conversation messages and the system uses Gemini to identify facts, categorize them, and store each one automatically.

```json
{
  "messages": [
    {"role": "user", "content": "I switched our database from Postgres to MongoDB"},
    {"role": "assistant", "content": "I'll update the ORM queries to use PyMongo"}
  ],
  "user_id": "ehfaz",
  "project_id": "my-project"
}
```

### 4. `get_project_context` - Bootstrap session context

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

### 5. `search_knowledge_graph` - Entity/relationship search

Search the Graphiti knowledge graph for structured relationships between entities.

```json
{
  "query": "FastAPI",
  "user_id": "ehfaz",
  "project_id": "neuralscape-graphiti",
  "limit": 10
}
```

### 6. `list_memories` - Inspect stored memories

List memories with optional filters. Use to verify what's been stored or audit memory contents.

```json
{
  "user_id": "ehfaz",
  "scope": "global",
  "category": "preference",
  "limit": 50
}
```

### 7. `delete_memories` - Remove memories

Delete by specific ID or by filter. Use with caution.

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

## REST API (alternative to MCP)

The service runs at `http://localhost:8199`. All v1 endpoints require `user_id`.

| Action | Method | Endpoint |
|---|---|---|
| Store via LLM extraction | `POST` | `/v1/memories` |
| Store single fact | `POST` | `/v1/memories/raw` |
| Store async | `POST` | `/v1/memories/async` |
| Poll async status | `GET` | `/v1/memories/status/{task_id}` |
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
