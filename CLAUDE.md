# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Neuralscape is a production-grade agentic memory layer combining mem0 (vector store) + Graphiti (temporal knowledge graph). It exposes both a REST API (FastAPI) and MCP server for AI agents to store and retrieve structured memories. Writes are async (enqueued to Redis/ARQ, return 202), reads are sync (return 200).

## Development Commands

All commands run from `neuralscape-service/` using `uv`:

```bash
# Install/sync dependencies
cd neuralscape-service && uv sync

# Start backing services (Neo4j, Redis, Qdrant)
docker compose up neo4j redis qdrant -d

# Run the API server
uv run python main.py

# Run the ARQ background worker (separate terminal)
uv run arq worker.WorkerSettings

# Run MCP server in stdio mode
uv run python mcp_server.py

# Run unit tests (no running services needed)
uv run pytest tests/ --ignore=tests/test_async_pipeline.py -v

# Run a single test file
uv run pytest tests/test_dedup.py -v

# Run a single test by name
uv run pytest tests/test_service.py -k "test_search_memories" -v

# Run integration tests (requires Neo4j, Redis, Qdrant running)
uv run pytest tests/test_async_pipeline.py -v -s

# Sync upstream subtree dependencies
./scripts/sync-upstream.sh [graphiti|mem0|all]
```

## Architecture

### Service Layer (`neuralscape-service/`)

- **`main.py`** — FastAPI app. Legacy endpoints at root, v1 endpoints at `/v1/*`. Also mounts MCP HTTP transport at `/mcp/`. Health checks at `/health` and `/api/v1/health`.
- **`memory_service.py`** — Core business logic. Handles extraction (via Gemini LLM), storage (Qdrant + Neo4j), search, dedup, and graph re-ingestion. This is the largest file (~1200 LOC).
- **`mcp_server.py`** — MCP server with 7 tools. Supports both stdio and Streamable HTTP transports.
- **`worker.py`** — ARQ background worker. Processes `process_memory_store` and `process_memory_raw` tasks. Runs periodic dedup cron job (every 6 hours).
- **`config.py`** — Pydantic Settings. All configuration via env vars. Builds the mem0 config dict.
- **`schemas.py`** — 13 memory categories organized by type (semantic, project, episodic, procedural, working). Pydantic request/response models.
- **`prompts.py`** — LLM extraction prompts and category parsers.
- **`task_manager.py`** — Redis-backed task enqueuing and status polling for async writes.
- **`logging_config.py`** — Structlog setup (JSON in prod, console in dev).

### Data Flow

- **Write path**: Client → API/MCP → Redis queue → 202 → ARQ Worker → Gemini extraction → Qdrant + Neo4j → Redis status → poll for result
- **Read path**: Client → API/MCP → MemoryService → Qdrant + Neo4j → 200 OK (synchronous)

### Memory Model

Two scopes: `global` (user-wide) and `project` (project-specific, requires `project_id`). Categories auto-assign scope: semantic categories (preference, personal_fact, etc.) default to global; project categories (tech_stack, convention, etc.) default to project.

### Subtree Dependencies

`graphiti/` and `mem0/` are git subtrees from upstream repos, installed as editable packages via `uv` (see `[tool.uv.sources]` in pyproject.toml). After syncing upstream, check for merge conflicts in mem0's `configs.py` / `factory.py`.

**The mem0 graph layer is an NS fork, not an upstream patch.** mem0 deleted its entire self-hostable OSS graph layer in PR #4805 (`a488e190`, 2026-04-14, the v3 pipeline) — `mem0/graphs/` and `graph_memory.py` / `memgraph_memory.py` no longer exist upstream, and graph memory is now a hosted-Platform-only feature. NS restores and maintains that layer plus a `GraphStoreFactory`, and `graphiti_memory.py` is **net-new NS code that never existed upstream** (the adapter presenting Graphiti as a mem0 graph provider). Because mem0 2.x's `Memory` class has no `.graph` concept, the re-attach shim in `memory_service.py` (~`:300-341`) fabricates one. Every sync re-grafts this layer onto a core that deliberately removed it — this is the dominant sync risk, documented in `docs/neuralscape/14-upstream-delta-report.md`.

### Required Services

| Service | Default URL | Purpose |
|---------|------------|---------|
| Neo4j 5 | `neo4j://127.0.0.1:7687` | Knowledge graph (Graphiti) |
| Redis 7 | `redis://localhost:6379` | ARQ task queue |
| Qdrant | `http://localhost:6333` | Vector store |
| Gemini API | via `GOOGLE_API_KEY` | LLM extraction + embeddings |

### Required Environment Variables

```
GOOGLE_API_KEY     # Gemini API key
NEO4J_PASSWORD     # Neo4j password (default user: neo4j)
REDIS_URL          # Redis connection string
QDRANT_URL         # Qdrant server URL (omit for local on-disk mode)
```

## Public Repo — Deployment Secrecy (HARD RULE)

**This is a public GitHub repo. NEVER commit organization- or deployment-specific
details to it.** This applies to every deployment target, including the
Optimizely and Digital Ambiance (DA) deployments.

Forbidden in tracked files:
- Company/org names tied to a deployment (e.g. `optimizely`, `optimizely.com`).
- GCP project IDs, cluster names, state-bucket names, or other infra identifiers
  (e.g. `iiis-492427`, `windmill-gke`, `svc-utility-belt`).
- Internal project slugs / codenames (e.g. `lightpath`, `openclaw`).
- Which external LLM gateway/endpoint or org we route inference through. (The
  generic `LLM_GATEWAY_*` *feature flags* in `config.py` are fine — the
  gateway's identity/URL is not.)

Keep `deploy/` (Terraform + Helm + k8s) **generic and parameterized** with
placeholder defaults (`project_id`, `neuralscape.example.com`, empty
`nodeSelector`, partial TF backend). Real values are supplied only at deploy time
inside the **private** org Git via `terraform.tfvars`, `backend.hcl`, and a
private Helm values / kustomize overlay. Grep before pushing:
`git grep -iE "optimizely|iiis-492427|windmill-gke|svc-utility-belt|lightpath"`.

## Git Workflow

**NEVER commit directly to `dev` or `main`.** Always create a feature branch:

```bash
git checkout -b feature/my-feature dev
# ... make changes, commit ...
# push and create PR targeting dev
git push -u origin feature/my-feature
gh pr create --base dev
```

- All work happens on feature branches (`feature/`, `fix/`, `hotfix/`)
- PRs target `dev`, get reviewed, then merge
- `dev` → `main` via release PRs only

## Testing

Tests are in `neuralscape-service/tests/`. Unit tests mock all external services. Integration tests (`test_async_pipeline.py`) require running Neo4j, Redis, and Qdrant — marked with `@pytest.mark.integration`.

## Docker

`docker compose up` starts the full stack (neo4j, redis, qdrant, neuralscape API, neuralscape-worker). The API runs on port 8199.
