# Neuralscape - Agentic Memory Layer

A production-grade memory system for AI coding assistants and personal agents. Neuralscape gives any LLM-powered agent persistent, structured memory across sessions and projects — remembering user preferences, project conventions, technical decisions, and learned facts.

Built on [mem0](https://github.com/mem0ai/mem0) (vector storage + LLM deduplication) and [Graphiti](https://github.com/getzep/graphiti) (temporal knowledge graph), exposed via REST API and MCP server. Memory writes are processed asynchronously by background workers via [ARQ](https://github.com/python-arq/arq) + Redis.

## Prerequisites

- **Python 3.10+** and **uv** (for local development)
- **Docker** + **Docker Compose** (for containerized deployment)
- **Google API key** with Gemini access (for LLM extraction + embeddings)
- **Neo4j** — included in Docker Compose, or use Neo4j Desktop for local dev
- **Redis** — included in Docker Compose, or run locally
- **Qdrant** — included in Docker Compose as a server, or run locally

## Quick Start (Docker)

```bash
# 1. Copy env template and add your Gemini API key
cp .env.example .env
# Edit .env: set GOOGLE_API_KEY=your-key

# 2. Start the full stack
docker compose up --build -d

# 3. Verify
docker compose ps
# Should show: neo4j, redis, qdrant, neuralscape, neuralscape-worker

curl http://localhost:8199/health
# → {"status":"ok","service":"neuralscape-memory","checks":{"redis":"ok","vector_store":"ok","graph_store":"ok"}}

# 4. Test async memory storage
curl -X POST http://localhost:8199/v1/memories/raw \
  -H "Content-Type: application/json" \
  -d '{"content": "Prefers dark mode", "user_id": "test", "category": "preference"}'
# → {"status":"accepted","task_id":"...","poll_url":"/v1/memories/status/..."}

# 5. Install the Claude Code plugin (optional but recommended)
cd neuralscape-plugin && npm install && npm run build && cd ..
# Then inside Claude Code:
#   /plugin marketplace add /path/to/neuralscape
#   /plugin install neuralscape@neuralscape-plugins --scope user

# Stop with: docker compose down
```

## Local Setup (without Docker)

```bash
# Start Redis and Qdrant (via Docker or locally)
docker run -d --name redis -p 6379:6379 redis:7-alpine
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.13.2

cd neuralscape-service

# Create .env file
cat > .env << 'EOF'
GOOGLE_API_KEY=your-gemini-api-key
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password
NEO4J_DATABASE=memory
QDRANT_URL=http://localhost:6333
EOF

# Install dependencies
uv sync

# Start the ARQ worker (terminal 1)
uv run arq worker.WorkerSettings

# Start the API server (terminal 2)
uv run python main.py
# → Listening on http://0.0.0.0:8199

# Run unit tests (no external services needed)
uv run pytest tests/test_service.py -v

# Run integration tests (requires all services running)
uv run pytest tests/test_async_pipeline.py -v -s
```

## Claude Code Plugin

The **neuralscape plugin** gives Claude Code automatic, persistent memory without any LLM involvement at the capture layer. It uses Claude Code's lifecycle hooks to inject stored context at session start and capture tool observations in the background as you work.

### How It Works

```
[Claude Code]
    ↓ SessionStart hook (sync — injects stored context)
    ↓ PostToolUse hook (async — captures tool observations)
    ↓ Stop hook (async — stores session marker)
[neuralscape-plugin/]  ← TypeScript plugin (thin capture layer)
    ↓ HTTP calls
[neuralscape-service/]  ← Python service (storage, extraction, graph)
```

| Hook | When | What it does | Blocking? |
|------|------|-------------|-----------|
| **SessionStart** | Session start/resume | Fetches context from `/v1/context/{projectId}`, injects as `additionalContext` | Yes (sync, ~1s) |
| **PostToolUse** | After Write, Edit, Bash, WebFetch, WebSearch, Task, NotebookEdit | Summarizes the tool action, fire-and-forget POST to `/v1/memories/raw` | No (async) |
| **Stop** | Session end | Stores a session-end marker as an `interaction` memory | No (async) |

Noisy tools (Glob, Grep, Read, AskUserQuestion) are excluded via the hook matcher.

### Installation

```bash
# 1. Build the plugin
cd neuralscape-plugin
npm install && npm run build
cd ..

# 2. Add the local marketplace (inside Claude Code)
/plugin marketplace add /path/to/neuralscape

# 3. Install the plugin
/plugin install neuralscape@neuralscape-plugins --scope user

# 4. Restart Claude Code — context will be injected automatically
```

The plugin is cached at `~/.claude/plugins/cache/` and loads on every session. After updating the plugin source, re-run steps 1 and 3.

### Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `NEURALSCAPE_URL` | `http://localhost:8199` | Neuralscape API URL |
| `NEURALSCAPE_USER_ID` | `ehfaz` | User ID for memory operations |

### Plugin + MCP Coexistence

The plugin and MCP server complement each other:

- **Plugin hooks** handle automatic capture (PostToolUse observations) and automatic injection (SessionStart context) — no LLM involvement
- **MCP tools** remain available for explicit operations: targeted search (`recall_memories`), manual storage (`remember`), knowledge graph queries (`search_knowledge_graph`)

Both can run simultaneously. The plugin captures breadcrumbs in the background; MCP tools let you or the agent interact with memory directly.

## MCP Server

7 tools exposed via MCP for direct use by AI agents:

| Tool | Mode | Purpose |
|---|---|---|
| `recall_memories` | sync | Semantic search across global + project memories. Agents should call this before starting work. |
| `remember` | async | Store a single categorized fact. Set `wait: true` to block until stored. |
| `remember_conversation` | async | Bulk extract from conversation messages via LLM. Set `wait: true` to block. |
| `get_project_context` | sync | Bootstrap: load all user prefs + project context organized by category. |
| `search_knowledge_graph` | sync | Graph-based entity/relationship search. |
| `list_memories` | sync | List/inspect stored memories with filters. |
| `delete_memories` | sync | Delete by ID or by filters. |

### Claude Code (stdio)

The quickest way to connect — runs the MCP server as a subprocess:

```bash
claude mcp add neuralscape-memory -- uv run --directory /absolute/path/to/neuralscape-service python mcp_server.py
```

Or add manually to your Claude Code MCP settings (`.claude/settings.json` or project settings):

```json
{
  "mcpServers": {
    "neuralscape-memory": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/neuralscape-service", "python", "mcp_server.py"]
    }
  }
}
```

> Requires the ARQ worker running separately: `uv run arq worker.WorkerSettings`

### Docker (Streamable HTTP)

For remote agents or containerized setups, set `MCP_TRANSPORT=http`. The MCP endpoint mounts at `/mcp/` on the same port as the REST API.

Add `MCP_TRANSPORT=http` to the neuralscape service in `docker-compose.yml`:

```yaml
neuralscape:
  environment:
    MCP_TRANSPORT: http
    # ... other env vars
```

Then restart the stack and add the MCP server:

```bash
docker compose up -d
claude mcp add neuralscape-memory --transport http http://localhost:8199/mcp/
```

Or manually:

```json
{
  "mcpServers": {
    "neuralscape-memory": {
      "type": "streamable-http",
      "url": "http://localhost:8199/mcp/"
    }
  }
}
```

### Testing with mcp-cli

You can verify the MCP server independently using [mcp-cli](https://github.com/philschmid/mcp-cli), a lightweight Bun-based CLI:

```bash
# Install
bun install -g https://github.com/philschmid/mcp-cli

# Create mcp_servers.json in the project root
cat > mcp_servers.json << 'EOF'
{
  "mcpServers": {
    "neuralscape": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/neuralscape-service", "python", "mcp_server.py"]
    }
  }
}
EOF

# List all tools
mcp-cli

# Inspect a tool's schema
mcp-cli info neuralscape recall_memories

# Call a read tool
mcp-cli call neuralscape recall_memories '{"query": "testing", "user_id": "ehfaz"}'

# Call a write tool (fire-and-forget)
mcp-cli call neuralscape remember '{"content": "test fact", "user_id": "ehfaz", "category": "interaction"}'

# Call a write tool (blocking)
mcp-cli call neuralscape remember '{"content": "test fact", "user_id": "ehfaz", "category": "interaction", "wait": true}'
```

## Enabling Memory Globally (Claude Code)

There are two approaches to giving Claude Code persistent memory. Use either or both.

### Approach 1: Plugin (recommended)

The neuralscape plugin handles context injection and observation capture automatically via lifecycle hooks. See the [Claude Code Plugin](#claude-code-plugin) section above for installation.

Once installed, the plugin loads on every session across all projects. Add the MCP server (below) alongside it for explicit memory operations like targeted search and manual storage.

### Approach 2: MCP server + CLAUDE.md instructions

For setups where you can't or don't want to use the plugin, you can rely on MCP tools and CLAUDE.md instructions to drive memory behavior.

**Step 1: Add the MCP server to global settings**

Add `neuralscape-memory` to the `mcpServers` key in `~/.claude.json`:

```jsonc
// ~/.claude.json — Streamable HTTP (recommended, shares a single server process)
{
  "mcpServers": {
    "neuralscape-memory": {
      "type": "http",
      "url": "http://localhost:8199/mcp"
    }
  }
}
```

> Make sure `MCP_TRANSPORT=http` is set in the neuralscape service environment.

**Step 2: Add memory instructions to global CLAUDE.md**

Create or append to `~/.claude/CLAUDE.md`:

```markdown
## Neuralscape Memory Layer

Context is automatically injected at session start and tool observations are captured
via the neuralscape plugin hooks. The MCP tools remain available for explicit memory operations.

### When to Store Memories (via MCP tools)

Proactively call `remember` (fire-and-forget, `wait: false`) when the user:
- Reveals a preference, shares a personal fact, or makes a technical decision
- Discusses project architecture, conventions, tech stack, or workflows
- Mentions a dependency, constraint, or compatibility issue

### Identity

Always pass `user_id: "your-user-id"` on every memory call.
Include `project_id` (use the project directory name) when working in a project context.
```

With both the plugin and MCP server in place, Claude Code will automatically inject stored context at session start, capture tool observations in the background, and have MCP tools available for explicit memory operations — across every project.

## REST API

All new endpoints live under `/v1`. Legacy endpoints at root are preserved for backward compatibility.

### Remember (async — returns 202)

```bash
# Extract and store from conversation (LLM-powered)
curl -X POST http://localhost:8199/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "I use Python 3.12 with FastAPI"}],
    "user_id": "ehfaz",
    "project_id": "neuralscape-graphiti"
  }'
# → {"status": "accepted", "task_id": "abc123", "poll_url": "/v1/memories/status/abc123"}

# Store a single pre-categorized fact (no LLM)
curl -X POST http://localhost:8199/v1/memories/raw \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Prefers 4-space indentation",
    "user_id": "ehfaz",
    "category": "preference"
  }'
# → {"status": "accepted", "task_id": "def456", "poll_url": "/v1/memories/status/def456"}

# Poll task status
curl http://localhost:8199/v1/memories/status/{task_id}
# → {"task_id": "abc123", "status": "completed", "result": {"memories": [...]}, "error": null}
```

### Recall (sync — returns 200)

```bash
# Semantic search (searches global + project when project_id given)
curl -X POST http://localhost:8199/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "indentation style",
    "user_id": "ehfaz",
    "project_id": "neuralscape-graphiti",
    "categories": ["preference", "convention"],
    "limit": 10
  }'

# Knowledge graph search
curl -X POST http://localhost:8199/v1/graph/search \
  -H "Content-Type: application/json" \
  -d '{"query": "FastAPI", "user_id": "ehfaz"}'
```

### Context Loading (sync)

```bash
# Full project context (global prefs + project facts, organized by category)
curl "http://localhost:8199/v1/context/neuralscape-graphiti?user_id=ehfaz"

# Global-only context
curl "http://localhost:8199/v1/context/global?user_id=ehfaz"
```

### Manage

```bash
# List with filters
curl "http://localhost:8199/v1/memories?user_id=ehfaz&scope=global&category=preference"

# Get/update/delete single memory
curl http://localhost:8199/v1/memories/{id}
curl -X PUT http://localhost:8199/v1/memories/{id} -d '{"content": "..."}'
curl -X DELETE http://localhost:8199/v1/memories/{id}

# List available categories
curl http://localhost:8199/v1/categories

# Graph introspection
curl "http://localhost:8199/v1/graph/nodes?user_id=ehfaz&project_id=neuralscape-graphiti"
curl "http://localhost:8199/v1/graph/edges?user_id=ehfaz"
curl "http://localhost:8199/v1/graph/episodes?user_id=ehfaz"
curl "http://localhost:8199/v1/graph/communities?user_id=ehfaz"
```

---

## How It Works

```
  [Claude Code]
       │
       ├─── SessionStart hook ──► neuralscape-plugin ──► GET /v1/context/{id}
       │                              (inject context)        │
       ├─── PostToolUse hook ───► neuralscape-plugin ──► POST /v1/memories/raw
       │                          (async, background)         │
       ├─── Stop hook ──────────► neuralscape-plugin ──► POST /v1/memories/raw
       │                          (async, background)         │
       │                                                      │
       │    ┌─────────────────────────────────────────────────┘
       │    │
       │    ▼
       │  ┌──────────────────────────────────────────────┐
       │  │           neuralscape-service                │
       │  │                                              │
       ├─►│  MCP Server (7 tools)   REST API (/v1)      │
       │  │       stdio / HTTP         FastAPI           │
       │  │              │                │              │
       │  │              └──────┬─────────┘              │
       │  │                     │                        │
       │  │  ┌─── reads ────────┤                        │
       │  │  │            writes │                        │
       │  │  ▼                  ▼                         │
       │  │  MemoryService    Redis                      │
       │  │  (sync reads)    (task queue)                │
       │  └──────┬──────────────┬────────────────────────┘
       │         │              │
       │    ┌────┤              │
       │    │    │              ▼
       │    │    │     ┌─────────────────┐
       │    │    │     │  ARQ Worker      │
       │    │    │     │  (separate proc) │
       │    │    │     │  MemoryService   │
       │    │    │     │  (async writes)  │
       │    │    │     └────────┬─────────┘
       │    │    │              │
    ┌──▼────▼──┐ │  ┌───────────┘
    │  Qdrant  │ │  │
    │ (vectors)│ │  ▼
    └──────────┘ │ ┌───────────┐
                 └►│   Neo4j   │
                   │ (Graphiti │
                   │   graph)  │
                   └───────────┘
```

**Plugin capture path**: Claude Code → PostToolUse hook → neuralscape-plugin summarizes tool action → fire-and-forget POST to `/v1/memories/raw` → Redis queue → ARQ Worker → Qdrant + Neo4j. Runs async in the background, never blocks Claude.

**Plugin injection path**: Claude Code → SessionStart hook → neuralscape-plugin calls `GET /v1/context/{projectId}` → formats as markdown → injected as `additionalContext`. Sync, runs once at session start (~1s).

**MCP/API write path**: Client → API/MCP → enqueue to Redis → 202 Accepted → ARQ Worker → Gemini extraction + Qdrant + Neo4j → result stored in Redis → client polls status.

**Read path**: Client → API/MCP → MemoryService → Qdrant + Neo4j → 200 OK with results (synchronous, no queue).

**Maintenance path**: ARQ cron job runs every 6 hours → scrolls all users → removes exact duplicates by hash → removes semantic near-duplicates above cosine threshold → expires related graph edges.

Every memory is stored **twice**: as a vector embedding in Qdrant (for semantic search) and as entities/relationships in a Neo4j knowledge graph via Graphiti (for structured reasoning). Both paths are queried on every search and results are merged.

## Philosophy

Neuralscape is an opinionated agentic memory layer. It pairs vector search with a temporal knowledge graph so agents get both semantic recall and structured entity/relationship reasoning in a single call. It is opinionated about three things: **categories** (13 predefined types that control scope defaults), **scopes** (global vs. project namespace isolation), and **dual-backend architecture** (every search queries both Qdrant and Graphiti, deduplicates, and returns a merged result set). The goal is a single `/v1/search` call that gives an agent everything it needs to remember — no manual backend selection required.

## Core Concepts

### Two-Scope Namespace

Memories live in one of two scopes:

| Scope | Graphiti group_id | Purpose |
|---|---|---|
| **Global** | `"global"` | Cross-project facts: user preferences, skills, personal details |
| **Project** | `"project--{slug}"` | Project-specific: tech stack, conventions, architecture decisions |

When you search with a `project_id`, Neuralscape searches **both** scopes and merges results by relevance score. An agent working on `neuralscape-graphiti` sees your global "prefers 4-space indentation" preference alongside the project-specific "uses FastAPI with Graphiti backend" fact.

### 13 Memory Categories

Since self-hosted mem0 has no native category system, every memory gets a `category` metadata field that controls scope defaults and enables filtered retrieval:

| Group | Categories | Default Scope |
|---|---|---|
| **Semantic** | `preference`, `personal_fact`, `technical_skill`, `domain_knowledge` | Global |
| **Project** | `tech_stack`, `convention`, `architecture`, `dependency` | Project |
| **Episodic** | `decision`, `interaction` | Flexible |
| **Procedural** | `workflow`, `procedure` | Flexible |
| **Working** | `task_context` | Flexible |

### Dual-Backend Search

Every call to `recall_memories` (MCP) or `POST /v1/search` (REST) queries **both** backends in a single operation:

1. **Qdrant** (vector search) — finds semantically similar memories by embedding distance
2. **Graphiti** (knowledge graph) — finds related entity/relationship edges from Neo4j

Results are deduplicated (graph facts that closely match a vector result are removed), then interleaved (vector-1, graph-1, vector-2, graph-2, ...) and returned with a `source` field on each result (`"vector"` or `"graph"`) so agents can see where each fact came from.

If the graph search fails (e.g. Neo4j is temporarily unreachable), vector results are still returned — graph search is non-critical.

### Custom LLM Extraction

When an agent sends a conversation to `POST /v1/memories`, Neuralscape doesn't just pass it through to mem0. Instead:

1. The request is enqueued to Redis and the API returns 202 immediately
2. An ARQ worker picks up the task and calls Gemini with a specialized extraction prompt
3. The LLM returns facts tagged with categories: `[preference] Prefers tabs over spaces`
4. Each fact is parsed and stored with proper scope/category metadata via `mem0.add(infer=False)`
5. The raw conversation is also fed to Graphiti's knowledge graph for entity/relationship extraction
6. Results are stored in Redis and available via status polling

This gives you categorized vector memories **and** a rich knowledge graph from the same input.

### Async Processing

All memory write operations are processed asynchronously via ARQ (async Redis queue):

- **API writes** (`POST /v1/memories`, `POST /v1/memories/raw`) return 202 Accepted with a `task_id`
- **MCP writes** (`remember`, `remember_conversation`) return a `task_id` by default, or block with `wait: true`
- **Workers** run in a separate process, processing tasks from the Redis queue
- **Status polling** via `GET /v1/memories/status/{task_id}` returns `queued`, `processing`, `completed`, or `failed`

### Automatic Deduplication

Since `mem0.add(infer=False)` creates new vectors without checking for existing similar ones, Qdrant accumulates duplicates over time. A periodic dedup cron job (every 6 hours via ARQ) keeps the vector store clean in two phases:

1. **Exact dedup** — Groups memories by their `hash` field (MD5 of content stored by mem0). Keeps the newest in each group, deletes the rest.
2. **Semantic dedup** — For each remaining memory, searches Qdrant for near-duplicates above a cosine similarity threshold (default 0.95). Deletes the older memory in each pair.

Both phases expire related Graphiti graph edges on delete. Graph cleanup is non-critical — failures are logged but don't block dedup.

### Graph Re-ingestion on Update

When a memory's content is updated via `PUT /v1/memories/{id}` or `update_memory`, the new content is automatically re-ingested into the Graphiti knowledge graph. This allows Graphiti's contradiction detection to expire stale edges and create new ones reflecting the updated fact — no manual delete-and-recreate required.

### Agent Isolation

`agent_id` is metadata for provenance tracking, not a scope boundary. All agents (Claude Code, a Cursor plugin, a custom bot) share the same memory space for a given user. Conflicts are handled by Graphiti's temporal edge invalidation (old facts get `invalid_at` timestamps) and mem0's LLM-based deduplication.

## Configuration

All settings are environment variables (loaded from `.env`):

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | | Gemini API key |
| `GEMINI_LLM_MODEL` | `gemini-3-flash-preview` | Model for LLM extraction |
| `GEMINI_LLM_FALLBACK_MODEL` | `gemini-2.5-flash` | Fallback model when primary returns 503 |
| `GEMINI_EMBEDDER_MODEL` | `gemini-embedding-001` | Model for embeddings |
| `NEO4J_URI` | `neo4j://127.0.0.1:7687` | Neo4j connection |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | | Neo4j password |
| `NEO4J_DATABASE` | `memory` | Neo4j database name |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection for ARQ task queue |
| `QDRANT_URL` | (none) | Qdrant server URL (e.g. `http://localhost:6333`). If set, uses server mode. |
| `QDRANT_ON_DISK` | `true` | Persist Qdrant to disk (only used when `QDRANT_URL` is not set) |
| `QDRANT_PATH` | `~/.neuralscape/qdrant` | Qdrant local storage path (only used when `QDRANT_URL` is not set) |
| `QDRANT_COLLECTION` | `neuralscape_memories` | Qdrant collection name |
| `HOST` | `0.0.0.0` | Service bind address |
| `PORT` | `8199` | Service port |
| `DEFAULT_USER_ID` | `default_user` | Fallback user ID when none provided |
| `MCP_TRANSPORT` | `stdio` | MCP transport: `stdio` or `http` |
| `ARQ_QUEUE_NAME` | `neuralscape:queue` | Redis queue key for ARQ workers |
| `ARQ_MAX_RETRIES` | `3` | Max retry attempts per background task |
| `ARQ_JOB_TIMEOUT` | `300` | Max seconds per background task (5 min) |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.95` | Cosine similarity threshold for semantic dedup |
| `DEDUP_BATCH_SIZE` | `100` | Qdrant scroll page size during dedup |
| `DEDUP_CRON_HOURS` | `{0,6,12,18}` | Hours (UTC) when the dedup cron runs |

## Tests

164 tests across 6 files:

| File | What it covers | Services needed |
|---|---|---|
| `test_service.py` | REST endpoint unit tests | None (mocked) |
| `test_memory_service.py` | Business logic (MemoryService) | None (mocked) |
| `test_mcp_tools.py` | MCP tool interface | None (mocked) |
| `test_production_readiness.py` | Config, health check, error handling | None (mocked) |
| `test_dedup.py` | Qdrant dedup (exact, semantic, cron) | None (mocked) |
| `test_async_pipeline.py` | End-to-end async pipeline | Redis, Qdrant, Neo4j |

```bash
# Run all unit tests (no services needed)
cd neuralscape-service
uv run pytest tests/ --ignore=tests/test_async_pipeline.py -v

# Run integration tests (requires running services)
uv run pytest tests/test_async_pipeline.py -v -s
```

## Project Structure

```
neuralscape/
├── docker-compose.yml            # Redis + Qdrant + Neo4j + API + Worker orchestration
├── .dockerignore                 # Build context filters
├── .env.example                  # Env template (copy to .env)
├── .claude-plugin/
│   └── marketplace.json          # Local marketplace for plugin distribution
├── neuralscape-plugin/           # Claude Code plugin (TypeScript)
│   ├── .claude-plugin/
│   │   └── plugin.json           # Plugin manifest
│   ├── hooks/
│   │   └── hooks.json            # Lifecycle hook definitions
│   ├── src/
│   │   ├── session-start.ts      # SessionStart: context injection
│   │   ├── post-tool-use.ts      # PostToolUse: observation capture
│   │   ├── stop.ts               # Stop: session marker
│   │   └── utils.ts              # Shared HTTP client, config, helpers
│   ├── scripts/                  # Built JS (generated by esbuild)
│   ├── package.json
│   └── tsconfig.json
├── neuralscape-service/          # The service (what you deploy)
│   ├── Dockerfile                # Multi-stage build with uv
│   ├── main.py                   # FastAPI app: legacy + v1 endpoints
│   ├── memory_service.py         # Business logic layer (MemoryService class)
│   ├── context_formatter.py      # Format memories as markdown for hook injection
│   ├── mcp_server.py             # MCP server: 7 tools, stdio + HTTP
│   ├── worker.py                 # ARQ worker: background task processing + dedup cron
│   ├── task_manager.py           # Redis-backed task enqueuing + status
│   ├── schemas.py                # Enums, category taxonomy, Pydantic models
│   ├── prompts.py                # LLM extraction prompt, category parser
│   ├── config.py                 # Pydantic settings (env-driven)
│   ├── logging_config.py         # Structured logging setup
│   ├── pyproject.toml            # Dependencies
│   └── tests/
│       ├── test_service.py             # REST endpoint unit tests (mocked)
│       ├── test_async_pipeline.py      # Integration tests (requires running services)
│       ├── test_memory_service.py      # Business logic tests
│       ├── test_mcp_tools.py           # MCP tool tests
│       ├── test_production_readiness.py # Config, health, and error handling tests
│       └── test_dedup.py               # Qdrant dedup tests (exact, semantic, cron)
├── scripts/
│   └── sync-upstream.sh          # Pull upstream changes for git subtree deps
├── mem0/                         # mem0 (git subtree from upstream)
│   └── mem0/memory/
│       └── graphiti_memory.py    # Graphiti adapter (local patches applied)
└── graphiti/                     # graphiti-core (git subtree from upstream)
```

## Architecture Decisions

**Why ARQ over Celery?** ARQ is async-native (both API and workers are `async def`), matching FastAPI + Graphiti's async Neo4j driver. Simple setup (~50 lines of config), built-in retries and result storage in Redis. Celery is designed for CPU-bound distributed workloads at massive scale — overkill for I/O-bound LLM calls and DB writes.

**Why async writes?** Memory storage involves sequential LLM calls (Gemini extraction, embeddings) and database writes (Qdrant vectors, Neo4j graph via Graphiti) taking 5-30s total. Async processing returns control to the client in <50ms while the worker handles the heavy lifting.

**Why Qdrant server mode?** The ARQ worker runs as a separate process from the API server. On-disk Qdrant only supports single-process access. Qdrant server mode (via Docker or standalone) allows both processes to connect concurrently.

**Why custom extraction instead of mem0's built-in?** Self-hosted mem0 doesn't support categories. By doing extraction in our service layer, we tag each fact with a category *before* storage, enabling filtered retrieval and organized context loading.

**Why two storage backends?** Vector search (Qdrant) is for "find memories similar to this query." Knowledge graph (Graphiti/Neo4j) is for "what entities are related to X?" and handles temporal fact invalidation (when facts change over time). Together they provide comprehensive recall.

**Why group_id-based scoping instead of separate databases?** Graphiti partitions data by `group_id` within a single Neo4j database. Using composite IDs (`"global"`, `"project--my-app"`) keeps the infrastructure simple while providing proper namespace isolation. Multi-scope search just queries multiple group_ids.

**Why not use agent_id as a scope boundary?** Multiple agents (Claude Code, a Slack bot, a CI pipeline) should all benefit from the same memory. Agent isolation would fragment knowledge. Instead, `agent_id` is provenance metadata — you can see *who* learned a fact but everyone can use it.

**Why a periodic dedup cron instead of dedup-on-write?** `mem0.add(infer=False)` bypasses mem0's built-in LLM dedup because we do our own extraction. Checking for duplicates on every write would add latency to the async write path and require embedding + search per write. A periodic batch job is simpler, runs during low-traffic hours, and can use higher thresholds without blocking user-facing operations.

**Why git subtrees for mem0 and graphiti?** Both dependencies have local patches (Graphiti adapter scoping, Neo4j driver fixes). Git subtrees keep the full upstream history, allow pulling upstream changes with `scripts/sync-upstream.sh`, and let local patches live as normal commits — no submodule headaches or fork maintenance.
