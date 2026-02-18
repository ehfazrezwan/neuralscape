# NeuralScape Graphiti - Conversation Summary

## Project Overview

This repository (`neuralscape-graphiti`) is a fork/working copy of [Graphiti](https://github.com/getzep/graphiti), a Python framework for building temporally-aware knowledge graphs for AI agents. We've customized it to use **Google Gemini** (via AI Studio API key) for LLM/embeddings and **Neo4j Desktop** with a custom database named `memory`.

## Repository Structure

```
neuralscape-graphiti/
  .git/
  .gitignore              # Root-level gitignore (covers .env, .venv, etc.)
  .dockerignore           # Excludes tests, docs, unused subpackages from Docker build
  .env                    # Root-level env (NEO4J + GOOGLE_API_KEY credentials)
  .env.example            # Template for users to copy to .env
  docker-compose.yml      # Neo4j + neuralscape-service orchestration
  docs/
    conversation-summary.md       # This file
    graphiti-mem0-implementation.md  # Graphiti-mem0 integration plan
  graphiti/                   # All upstream graphiti source lives here
    graphiti_core/            # Core library (the Python package)
    server/                   # REST API server (FastAPI)
    mcp_server/               # MCP server (Model Context Protocol)
    examples/                 # Example scripts
    tests/                    # Test suite
    pyproject.toml            # Root package config for graphiti-core
    Makefile, README.md, etc.
  mem0/                       # Fork of mem0 with Graphiti graph memory adapter
    mem0/                     # Core library (the Python package)
      memory/graphiti_memory.py  # MemoryGraph adapter + _AsyncBridge
      configs/base.py            # MemoryConfig (top-level Pydantic config)
    tests/
      test_graphiti_config.py    # Config + factory unit tests (10 tests)
      test_graphiti_memory.py    # Adapter unit tests (19 tests)
      test_graphiti_integration.py  # Integration tests (3, skipped without env)
    pyproject.toml
  neuralscape-service/        # FastAPI service wrapping mem0 + Graphiti
    Dockerfile                # Multi-stage build using uv
    main.py                   # REST endpoints (sync + async memory, graph queries)
    config.py                 # pydantic-settings config
    mcp_server.py             # MCP server (Model Context Protocol)
    tests/
      conftest.py             # Adds parent dir to sys.path
      test_service.py         # Endpoint tests (13 tests)
    pyproject.toml
```

The entire upstream graphiti codebase was moved into the `graphiti/` subdirectory to allow for a monorepo structure. `mem0/` is a fork of the mem0 library with a custom Graphiti adapter. `neuralscape-service/` is the application layer.

## Branches

- `bootstrap/setting-up` — initial setup (merged to `dev`)
- `dev` — integration branch
- `feature/agentic-memory-layer` — v1 API, MCP, MemoryService (merged to `dev` via PR #2)
- `feature/dockerizing-neuralscape` — Docker containerization (merged to `dev` via PR #4)
- `main` — stable

## What Was Done

### 1. MCP Server: Neo4j Database Passthrough Fix

**Problem**: `DatabaseDriverFactory.create_config()` in `graphiti/mcp_server/src/services/factories.py` returned `{uri, user, password}` for Neo4j but omitted the `database` field, so it always defaulted to `neo4j`.

**Fix**: Added `'database': neo4j_config.database` to the returned dict.

**Problem**: The Neo4j branch in `graphiti/mcp_server/src/graphiti_mcp_server.py` passed `uri/user/password` directly to `Graphiti()`, which internally creates a `Neo4jDriver` with default `database='neo4j'`. The FalkorDB branch correctly created a `FalkorDriver` instance first.

**Fix**: Changed to create a `Neo4jDriver` instance directly (matching the FalkorDB pattern) and pass it via `graph_driver=`.

### 2. Cross-Encoder Support for Non-OpenAI Providers

**Problem**: `Graphiti.__init__()` (in `graphiti/graphiti_core/graphiti.py:221-224`) defaults ALL None clients to OpenAI variants:
- `llm_client=None` -> `OpenAIClient()`
- `embedder=None` -> `OpenAIEmbedder()`
- `cross_encoder=None` -> `OpenAIRerankerClient()`

This means if you pass `llm_client` and `embedder` but forget `cross_encoder`, it crashes with "OPENAI_API_KEY not set".

**Fix**: Both servers now explicitly create a `GeminiRerankerClient` when using Gemini as the LLM provider, using `graphiti_core.cross_encoder.gemini_reranker_client.GeminiRerankerClient`.

### 3. REST Server: Multi-Provider Support

**Files changed**: `graphiti/server/graph_service/config.py`, `graphiti/server/graph_service/zep_graphiti.py`

Previously the REST server was hardcoded to OpenAI. Now:

- `Settings` class has `llm_provider`, `embedding_provider`, `google_api_key`, `small_model_name`, `neo4j_database` fields. `openai_api_key` is now optional.
- `ZepGraphiti.__init__` accepts `graph_driver=`, `embedder=`, `cross_encoder=` params (forwarded to `Graphiti.__init__`).
- Helper functions `_create_llm_client()`, `_create_embedder()`, `_create_cross_encoder()` dispatch on provider setting.
- Both `get_graphiti()` and `initialize_graphiti()` create a `Neo4jDriver` with `database=settings.neo4j_database`.
- No router files needed changes - they use `ZepGraphitiDep` which is unaffected.

### 4. MCP Server Config Switched to Gemini + Neo4j

**File**: `graphiti/mcp_server/config/config.yaml`

- `llm.provider`: `openai` -> `gemini`
- `llm.model`: `gpt-4o-mini` -> `gemini-3-pro-preview`
- `embedder.provider`: `openai` -> `gemini`
- `embedder.model`: `text-embedding-3-small` -> `gemini-embedding-001`
- `embedder.dimensions`: `1536` -> `1024`
- `database.provider`: `falkordb` -> `neo4j`

### 5. Local graphiti-core via Editable Path

Both `graphiti/server/pyproject.toml` and `graphiti/mcp_server/pyproject.toml` have:

```toml
[tool.uv.sources]
graphiti-core = { path = "..", editable = true }
```

This points to the local `graphiti/` directory (which contains `graphiti_core/` and `pyproject.toml`). This is necessary because the PyPI version of `graphiti-core` (0.28.0) does NOT have the `database` parameter on `Neo4jDriver.__init__` - the local repo version does.

The `google-genai` extra was also added to both dependency specs:
- Server: `graphiti-core[google-genai]`
- MCP Server: `graphiti-core[falkordb,google-genai]`

### 6. Repository Restructuring

All files moved from repo root into `graphiti/` subdirectory via `git mv`. A copy of `.gitignore` remains at the repo root.

### 7. mem0 Subproject with Graphiti Graph Memory Adapter

**New directory**: `mem0/`

A fork of [mem0](https://github.com/mem0ai/mem0) with a custom `MemoryGraph` adapter (`mem0/mem0/memory/graphiti_memory.py`) that delegates all graph operations to Graphiti's temporal knowledge graph engine.

Key components:
- **`_AsyncBridge`** — runs a dedicated `asyncio` event loop in a background thread, so Graphiti's async operations (including the Neo4j driver) don't conflict with the caller's event loop.
- **`MemoryGraph`** — implements mem0's graph store interface (`add`, `search`, `get_all`, `delete_all`), translating mem0 `filters` to Graphiti `group_id` and delegating to `graphiti.add_episode()`, `graphiti.search()`, etc.
- **`GraphitiConfig`** (`mem0/mem0/graphs/configs.py`) — Pydantic config for Graphiti-specific fields (Neo4j connection, LLM/embedder/reranker provider settings).
- **`GraphStoreFactory`** (`mem0/mem0/utils/factory.py`) — registers `"graphiti"` provider mapping to `mem0.memory.graphiti_memory.MemoryGraph`.

### 8. neuralscape-service: FastAPI + MCP Service

**New directory**: `neuralscape-service/`

Lightweight FastAPI service wrapping mem0 with Graphiti backend. Endpoints:
- `GET /health` — health check
- `POST /memories` — sync memory addition (blocks during Graphiti entity extraction, 5-15s)
- `POST /memories/async` — non-blocking memory addition, returns task_id immediately
- `GET /memories/status/{task_id}` — poll for async task result (`processing`/`completed`/`failed`)
- `POST /search` — semantic memory search
- `GET /memories` — list all memories for a user
- `DELETE /memories` — delete all memories for a user
- `GET /graph/nodes`, `GET /graph/edges`, `GET /graph/episodes`, `GET /graph/communities` — direct Graphiti graph queries
- `POST /graph/search` — advanced Graphiti search with configurable `SearchConfig`

**Async architecture**: `POST /memories/async` uses `AsyncMemory` (mem0's async variant) with `asyncio.create_task()`. The chain: FastAPI loop -> `create_task(_process())` -> `await async_mem.add()` -> `asyncio.to_thread(self.graph.add, ...)` -> `_bridge.run(coro)` (runs on the bridge's independent loop). No event loop conflicts.

### 9. Docker Containerization

**New files**: `.dockerignore`, `neuralscape-service/Dockerfile`, `docker-compose.yml`, `.env.example`

The full stack can now be started with `docker compose up`. Key design:

- **Multi-stage Dockerfile** — Stage 1 (builder) uses `python:3.12-slim` + `uv` to install all deps via `uv sync --frozen --no-dev`. Stage 2 (runtime) copies only the venv and source code, keeping the image lean.
- **Build context is the project root** — the Dockerfile lives in `neuralscape-service/` but the build context is `.` so it can `COPY` the sibling `graphiti/` and `mem0/` directories for editable installs.
- **`.dockerignore`** — excludes `.venv/`, `__pycache__/`, `.env`, tests, docs, and unused subpackages (`mem0/embedchain/`, `mem0/openmemory/`, `graphiti/examples/`, etc.) to minimize build context (~4MB instead of ~500MB+).
- **Docker Compose** — two services: `neo4j` (5-community with `cypher-shell` healthcheck) and `neuralscape` (waits for healthy Neo4j via `depends_on` + `service_healthy`).
- **Qdrant** — persisted via named volume at `/data/qdrant` inside the container.
- **Neo4j** — community edition with `NEO4J_AUTH=neo4j/neuralscape` for initial password. Data persisted via `neo4j_data` named volume.
- **Env var mapping** — `GEMINI_LLM_MODEL` and `GEMINI_EMBEDDER_MODEL` are mapped from the root `.env`'s `SMALL_MODEL_NAME` and `EMBEDDING_MODEL_NAME` via compose variable substitution with defaults.
- **`.env.example`** — template with `GOOGLE_API_KEY` and `NEO4J_PASSWORD` for users to copy.

**Build gotchas encountered**:
- Both `graphiti/` and `mem0/` pyproject.toml files reference `README.md` (hatchling/poetry `readme` field). These must be copied into the build context or the editable install fails with "Readme file does not exist".
- `mem0/pyproject.toml` also has `license-files = ["LICENSE"]`, requiring the LICENSE file in the build context.
- `config.py` uses `GEMINI_EMBEDDER_MODEL` (pydantic-settings auto-maps from field name), but the root `.env` uses `EMBEDDING_MODEL_NAME`. The compose `environment:` block bridges this with `GEMINI_EMBEDDER_MODEL: ${EMBEDDING_MODEL_NAME:-gemini-embedding-001}`.

### 10. Test Coverage

| File | Tests | Status |
|------|-------|--------|
| `mem0/tests/test_graphiti_config.py` | 10 — config validation, defaults, factory registration | passing |
| `mem0/tests/test_graphiti_memory.py` | 19 — `_AsyncBridge`, init, add, search, get_all, delete_all, group_id mapping, source description | passing |
| `mem0/tests/test_graphiti_integration.py` | 3 — full cycle (direct + via Memory), temporal edge invalidation | skipped without `NEO4J_URI`/`GOOGLE_API_KEY` |
| `neuralscape-service/tests/test_service.py` | 13 — health, sync add, search, list, delete, async add, task status | passing |

Run tests:
```bash
# mem0 unit tests (from neuralscape-service, using its venv)
uv run python -m pytest ../mem0/tests/test_graphiti_config.py ../mem0/tests/test_graphiti_memory.py -v

# Service tests
cd neuralscape-service && uv run python -m pytest tests/ -v

# Integration tests (requires Neo4j + Gemini)
NEO4J_URI=neo4j://127.0.0.1:7687 NEO4J_PASSWORD=... GOOGLE_API_KEY=... \
  uv run python -m pytest ../mem0/tests/test_graphiti_integration.py -v
```

## Commit History

```
# feature/dockerizing-neuralscape branch (merged to dev via PR #4)
f4b0208 feat: add docker-compose orchestration and env template
6e8245e feat: add Dockerfile and build context filters for neuralscape service

# feature/agentic-memory-layer branch (merged to dev via PR #2)
07d7091 docs: add project README, memory layer skill, and conversation summary
1417e76 test: add comprehensive tests for v1 API, MemoryService, and MCP tools
897eb69 feat: rewrite MCP server with 7 tools and dual transport
d32a314 feat: add v1 REST API with scoped memory endpoints
136a339 feat: add MemoryService business logic layer
bf4b7eb feat: add Qdrant vector store config and composite group_id scoping
21631a7 feat: add category taxonomy, scoping enums, and extraction prompt

# bootstrap/setting-up branch (merged to dev via PR #1)
2004f8b test: add endpoint tests for neuralscape-service
2e8db95 feat: add non-blocking POST /memories/async endpoint
e296895 test: add unit and integration tests for Graphiti adapter
30203a7 fix: configure mem0 LLM/embedder for Gemini and route graph endpoints through bridge
4bfaa1f fix: replace per-thread event loop with single async bridge
5188634 docs: add Graphiti-mem0 integration implementation plan
c46a16b feat: add neuralscape-service with REST and MCP interfaces
ae1e20a feat: add mem0 subproject with Graphiti graph memory adapter
e0cd131 refactor: move all repository contents into graphiti/ subdirectory
232a4a5 chore: switch to Gemini + Neo4j and use local graphiti-core
174e166 feat: add multi-provider LLM, embedder, and cross-encoder support to REST server
696771d fix: pass Neo4j database param through MCP server and add cross-encoder support
ab7f316 fix: route execute_query to correct database when using custom db name  (pre-existing)
```

## Environment Setup

### Prerequisites

- **Python 3.12** (managed by uv)
- **uv** package manager (`~/.local/bin/uv`)
- **Docker** + **Docker Compose** (for containerized deployment)
- **Neo4j Desktop** with a database named `memory` running on `neo4j://127.0.0.1:7687` (for local dev; Docker Compose includes its own Neo4j)
- **Google API Key** for Gemini (AI Studio)

### .env Files

These are gitignored. Three locations:

**Root `.env`** (used by `graphiti/examples/quickstart/quickstart_memory.py`):
```env
GOOGLE_API_KEY=<key>
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=memory
```

**`graphiti/mcp_server/.env`**:
```env
GOOGLE_API_KEY=<key>
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=memory
```

**`graphiti/server/.env`**:
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=<key>
MODEL_NAME=gemini-3-pro-preview
SMALL_MODEL_NAME=gemini-3-flash-preview
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL_NAME=gemini-embedding-001
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=memory
```

**`neuralscape-service/.env`**:
```env
GOOGLE_API_KEY=<key>
GEMINI_LLM_MODEL=gemini-2.5-flash
GEMINI_EMBEDDER_MODEL=text-embedding-004
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=memory
```

### Setting Up Virtual Environments

Each server/service has its own `.venv`. From the repo root:

```bash
# REST server
cd graphiti/server
uv sync --extra dev

# MCP server
cd graphiti/mcp_server
uv sync

# Neuralscape service
cd neuralscape-service
uv sync --dev    # includes pytest, pytest-asyncio, httpx
```

**Important**: Always use each server's own `.venv` when running commands:
- REST server: `graphiti/server/.venv/bin/python`, `graphiti/server/.venv/bin/uvicorn`
- MCP server: `graphiti/mcp_server/.venv/bin/python`
- Neuralscape service: `cd neuralscape-service && uv run ...`

### Running with Docker

```bash
# Copy env template and fill in your API key
cp .env.example .env
# Edit .env: set GOOGLE_API_KEY=your-key

# Start full stack (Neo4j + neuralscape service)
docker compose up --build -d

# Verify
docker compose ps
curl http://localhost:8199/health
# → {"status":"ok","service":"neuralscape-memory"}

# Neo4j browser at http://localhost:7474

# Stop
docker compose down
```

To use a local Neo4j Desktop instead of the containerized one, comment out the `neo4j` service in `docker-compose.yml` and change `NEO4J_URI` to `neo4j://host.docker.internal:7687`.

### Running the Servers (Local, without Docker)

Both graphiti servers default to port 8000. Neuralscape service runs on port 8199.

**REST Server** (Graphiti direct):
```bash
cd graphiti/server
.venv/bin/uvicorn graph_service.main:app --host 127.0.0.1 --port 8000 --reload
# Healthcheck: curl http://127.0.0.1:8000/healthcheck
# Returns: {"status":"healthy"}
```

**MCP Server** (Graphiti MCP):
```bash
cd graphiti/mcp_server
.venv/bin/python main.py
# Healthcheck: curl http://localhost:8000/health
# Returns: {"status":"healthy","service":"graphiti-mcp"}
# MCP endpoint: http://localhost:8000/mcp/
```

**Neuralscape Service** (mem0 + Graphiti):
```bash
cd neuralscape-service
uv run uvicorn main:app --host 127.0.0.1 --port 8199 --reload
# Healthcheck: curl http://127.0.0.1:8199/health
# Returns: {"status":"ok","service":"neuralscape-memory"}

# Async memory addition:
curl -X POST http://localhost:8199/memories/async \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I work at Acme"}], "user_id": "test"}'
# Returns: {"status": "accepted", "task_id": "..."}

curl http://localhost:8199/memories/status/<task_id>
# Returns: {"status": "completed", "result": {...}}
```

## Key Learnings / Gotchas

1. **Graphiti defaults everything to OpenAI**: If you pass `None` for `llm_client`, `embedder`, or `cross_encoder`, `Graphiti.__init__` creates OpenAI defaults. You must pass ALL three explicitly when using a non-OpenAI provider.

2. **Neo4jDriver `database` param**: Only available in the local/latest version of graphiti-core, not in the PyPI 0.28.0 release. This is why we use `[tool.uv.sources]` to point at the local repo.

3. **Each server has its own .venv**: Don't use the root `.venv` or cross-use venvs between servers. The root `.venv` is for the graphiti-core library itself.

4. **Config flow**:
   - REST server: `.env` -> `pydantic_settings` `Settings` class -> helper functions create clients -> `Neo4jDriver` + `ZepGraphiti`
   - MCP server: `.env` + `config/config.yaml` -> `GraphitiConfig` -> factory classes create clients -> `Neo4jDriver` + `Graphiti`

5. **MCP server config.yaml supports env var expansion**: Values like `${GOOGLE_API_KEY}` in the YAML are expanded from environment variables (loaded from `.env`).

6. **Available Gemini cross-encoder**: `graphiti_core.cross_encoder.gemini_reranker_client.GeminiRerankerClient` - scores passages on a 0-100 scale (Gemini doesn't support logprobs like OpenAI).

7. **Quickstart reference**: `graphiti/examples/quickstart/quickstart_memory.py` demonstrates the correct Gemini + Neo4j pattern with all components wired up properly.

## Key Learnings / Gotchas (continued)

8. **mem0 always creates a vector store** — `Memory.__init__` unconditionally creates an embedder and vector store (defaults to Qdrant in local/in-memory mode at `~/.mem0/`). The graph store is additive, not a replacement. When `add()` runs, it executes two parallel pipelines: (1) LLM extracts facts -> embeds -> stores in vector DB, and (2) raw text -> Graphiti entity extraction. This means LLM fact extraction runs twice (once per pipeline). If you only want the knowledge graph, use `MemoryGraph` directly (which is what the `/graph/*` endpoints do).

9. **`_AsyncBridge` pattern** — Graphiti's Neo4j driver binds to whichever event loop is active during its creation. The `_AsyncBridge` creates a dedicated background thread with its own event loop, initializes the Neo4j driver on that loop, and routes all async operations through it via `run_coroutine_threadsafe`. This avoids "Future attached to a different loop" errors when FastAPI or mem0's `AsyncMemory` try to use the driver from their own event loops.

10. **`AsyncMemory.from_config()` is sync** — despite the class name, `AsyncMemory.__init__` is synchronous. It creates its own `MemoryGraph` which spawns its own `_AsyncBridge`. The async methods (`add()`, `search()`, etc.) use `asyncio.to_thread()` internally to delegate to the sync graph adapter.

11. **Graphiti supports 4 graph DB backends** — Neo4j, FalkorDB, Kuzu, Neptune. FalkorDB is a first-class, actively maintained backend (not experimental). The driver lives at `graphiti_core/driver/falkordb_driver.py` with full implementations of all 11 operation interfaces. Install via `pip install graphiti-core[falkordb]`.

12. **FalkorDB vector search is brute-force in Graphiti** — FalkorDB has HNSW vector indexes, but Graphiti's FalkorDB driver doesn't use them. Instead, it computes cosine similarity inline via `vec.cosineDistance()` (exact but O(n) per query, vs O(log n) with HNSW). Fine for small graphs, degrades at scale.

13. **FalkorDB lacks multi-statement transactions** — single-query atomicity only. If an `add_episode` fails mid-way, you could get orphaned nodes/edges. Neo4j provides full ACID rollback.

14. **neuralscape-service uv venv** — the venv is uv-managed (no `pip` binary). Install dev dependencies with `uv add --dev <pkg>` or `uv sync --dev`, not `pip install`.

15. **Docker build requires README.md and LICENSE for editable packages** — hatchling (graphiti) and poetry (mem0) validate metadata files during editable installs. If `.dockerignore` excludes `README.md` or `LICENSE`, `uv sync` fails with "Readme file does not exist". Both must be explicitly copied in the Dockerfile.

16. **`host.docker.internal` for local development** — when running neuralscape in Docker against a host-machine Neo4j (e.g. Neo4j Desktop), the compose `NEO4J_URI` must use `neo4j://host.docker.internal:7687`, not `127.0.0.1` (which resolves to the container itself). The compose file defaults to `neo4j://neo4j:7687` (the compose service name) for the full-stack setup.

17. **Env var name mismatch between root .env and config.py** — the root `.env` uses `EMBEDDING_MODEL_NAME` and `SMALL_MODEL_NAME` (matching graphiti server conventions), but `neuralscape-service/config.py` pydantic-settings expects `GEMINI_EMBEDDER_MODEL` and `GEMINI_LLM_MODEL`. The docker-compose `environment:` block bridges this with `GEMINI_EMBEDDER_MODEL: ${EMBEDDING_MODEL_NAME:-gemini-embedding-001}`.

## TODOs

### Vector Store: Switch from Default Qdrant to Production Store

**Priority**: High (before any production deployment)

**Current state**: mem0 defaults to local Qdrant (in-memory/file-based at `~/.mem0/`). This works for development but:
- Data is ephemeral (cleared on restart unless `on_disk=True`)
- Not accessible from other services
- No backup/replication
- Performance uncharacterized at scale

**Options to evaluate**:
1. **Qdrant server** (self-hosted or Qdrant Cloud) — simplest migration, same client library
2. **pgvector** — if we already run Postgres, avoids adding another service
3. **Pinecone** — managed, zero-ops, good for serverless workloads
4. **Disable vector store entirely** — if we decide Graphiti's hybrid search (fulltext + vector via Neo4j/FalkorDB) is sufficient and we don't need mem0's separate fact extraction pipeline. This would require bypassing `Memory` and using `MemoryGraph` directly, losing mem0's LLM-based fact deduplication and update logic.

**Decision needed**: Do we want both pipelines (vector + graph), or should we rely solely on Graphiti's knowledge graph for storage and search? The dual pipeline means double LLM calls per `add()` but provides both semantic similarity search (vector) and structured relational queries (graph).

### Graph Database: Neo4j vs FalkorDB

**Priority**: Medium (evaluate before scaling to multiple concurrent users)

**Current state**: Neo4j Desktop with local database `memory`.

**FalkorDB advantages**: In-memory speed (~10x+ faster queries), horizontal write scaling via Redis Cluster, open source multi-tenancy.

**FalkorDB risks**: No multi-statement ACID transactions, brute-force vector search in Graphiti's driver (O(n) vs O(log n)), weaker persistence guarantees (Redis AOF), smaller community.

**Recommendation**: Stay with Neo4j for now. The async endpoint solves the blocking UX problem. Switching to FalkorDB is a config change (Graphiti already has the driver), so defer until write throughput becomes a bottleneck.

## Files Modified (from upstream)

| File | What Changed |
|------|-------------|
| `graphiti/mcp_server/src/services/factories.py` | Added `database` to Neo4j config dict |
| `graphiti/mcp_server/src/graphiti_mcp_server.py` | Neo4jDriver pattern + cross-encoder creation |
| `graphiti/mcp_server/config/config.yaml` | Switched to gemini/neo4j providers |
| `graphiti/mcp_server/pyproject.toml` | Added google-genai extra + uv source override |
| `graphiti/server/graph_service/config.py` | Added provider fields, made openai_api_key optional |
| `graphiti/server/graph_service/zep_graphiti.py` | Multi-provider support, graph_driver param, Neo4jDriver |
| `graphiti/server/pyproject.toml` | Added google-genai extra + uv source override |
| `mem0/mem0/memory/graphiti_memory.py` | New — Graphiti MemoryGraph adapter + _AsyncBridge |
| `mem0/mem0/graphs/configs.py` | Added GraphitiConfig Pydantic model |
| `mem0/mem0/utils/factory.py` | Registered "graphiti" in GraphStoreFactory |
| `neuralscape-service/main.py` | New — FastAPI service with sync/async memory endpoints + graph endpoints |
| `neuralscape-service/config.py` | New — pydantic-settings config for Gemini + Neo4j |
| `neuralscape-service/Dockerfile` | New — multi-stage Docker build with uv |
| `docker-compose.yml` | New — Neo4j + neuralscape service orchestration |
| `.dockerignore` | New — build context filters |
| `.env.example` | New — env template for Docker users |
