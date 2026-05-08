---
title: Deployment, Config & Observability
date: 2026-05-06
tags: [reference, neuralscape, deployment, docker, config, ops]
source: handwritten
---
 
# Deployment, Config & Observability

Operational reference for running neuralscape locally and in production. Covers the docker compose stack, the shared Dockerfile, environment variables, structured logging, the few helper scripts, the (intentional) absence of CI, and the gotchas you will hit if you skip the env file.

## Overview

Neuralscape ships as a five-service Docker stack: three backing data stores (Neo4j, Redis, Qdrant), the FastAPI server, and a single ARQ background worker. The API and worker share one image built from `neuralscape-service/Dockerfile`; only the entrypoint differs (`uvicorn` vs `arq worker.WorkerSettings`). All services have memory limits, health checks, log rotation, and `restart: unless-stopped`. There is no Kubernetes manifest — the compose file is the deployment unit. See [02-service-architecture](./02-service-architecture.md) for the runtime layout.

## Docker Compose reference

The compose file at `docker-compose.yml:1-159` defines five services. All use json-file logging with `max-size: 10m` and `max-file: 3` for rotation.

| Service | Image | Ports | Memory | Healthcheck |
|---------|-------|-------|--------|-------------|
| neo4j | `neo4j:5-community` | 7474, 7687 | 2G | `cypher-shell ... RETURN 1` (`docker-compose.yml:23`) |
| redis | `redis:7-alpine` | 6379 | 512M | `redis-cli ping` (`docker-compose.yml:45`) |
| qdrant | `qdrant/qdrant:v1.15.0` | 6333, 6334 | 2G | TCP probe on 6333 (`docker-compose.yml:68`) |
| neuralscape | local Dockerfile | 8199 | 2G | urllib GET `/health` (`docker-compose.yml:110`) |
| neuralscape-worker | same Dockerfile | none | 2G | `python -c "sys.exit(0)"` (`docker-compose.yml:150`) |

### Volumes

Three named volumes (`neo4j_data`, `redis_data`, `qdrant_data`) plus a host bind mount for the Obsidian vault: `${OBSIDIAN_VAULT_PATH:-./vault}:/data/vault` (`docker-compose.yml:98`). The vault path is shared by API and worker so both can read/write notes.

### Env injection (API + worker)

Both app services pull from `.env` and override container-network values inline (`docker-compose.yml:87-96` and `:128-136`):

```yaml
NEO4J_URI: neo4j://neo4j:7687
NEO4J_PASSWORD: ${NEO4J_PASSWORD:-neuralscape}
NEO4J_DATABASE: ${NEO4J_DATABASE:-memory}
QDRANT_URL: http://qdrant:6333
REDIS_URL: redis://redis:6379
GEMINI_LLM_MODEL: ${SMALL_MODEL_NAME:-gemini-3-flash-preview}
GEMINI_EMBEDDER_MODEL: ${EMBEDDING_MODEL_NAME:-gemini-embedding-001}
MCP_TRANSPORT: http
OBSIDIAN_VAULT_PATH: /data/vault
```

### Startup ordering

Both app services declare `depends_on` with `condition: service_healthy` for all three data stores (`docker-compose.yml:80-85`, `:120-126`). The API will not start until Neo4j answers `RETURN 1`, Redis answers `PING`, and Qdrant accepts a TCP connection. This eliminates the cold-start race where the worker boots before Neo4j is reachable.

## Dockerfile

Two-stage build at `neuralscape-service/Dockerfile:1-56`. Both stages use `python:3.12-slim`.

**Builder (`Dockerfile:1-21`)** pulls the `uv` binary from `ghcr.io/astral-sh/uv:latest`, copies dependency manifests for graphiti, mem0, and neuralscape-service (`Dockerfile:10-12`), copies the source for the editable subtree packages (`Dockerfile:15-18`), then runs `uv sync --frozen --no-dev` (`Dockerfile:21`) to materialize a `.venv`.

**Runtime (`Dockerfile:23-56`)** copies the venv from the builder (`Dockerfile:29`) and re-copies just the source needed at runtime. It puts the venv on PATH (`Dockerfile:40`), sets container-default env vars `NEO4J_URI=neo4j://neo4j:7687` and `QDRANT_PATH=/data/qdrant` (`Dockerfile:43-44`), creates a non-root `appuser` (uid 1000) with `/data/{qdrant,vault}` writable (`Dockerfile:49-50`), exposes 8199, and starts uvicorn:

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8199"]
```

The compose worker overrides this with `command: ["arq", "worker.WorkerSettings"]` (`docker-compose.yml:119`). One image, two roles — see [07-async-pipeline](./07-async-pipeline.md) for the worker's job semantics.

## Configuration

All settings live in `neuralscape-service/config.py` as a Pydantic `BaseSettings` class loading `.env`. `Settings.validate_required()` (`config.py:59-77`) is called once at startup from `main.py` and aborts boot if any of the four required vars are missing.

### Required (validated at boot)

| Setting | Env var | Default | Source |
|---------|---------|---------|--------|
| google_api_key | `GOOGLE_API_KEY` | empty | `config.py:10`, validated `:67` |
| neo4j_password | `NEO4J_PASSWORD` | empty | `config.py:18`, validated `:68` |
| neo4j_uri | `NEO4J_URI` | `neo4j://127.0.0.1:7687` | `config.py:16`, validated `:70` |
| redis_url | `REDIS_URL` | `redis://localhost:6379` | `config.py:32`, validated `:72` |

### Optional (with defaults)

| Setting | Default | Source |
|---------|---------|--------|
| `gemini_llm_model` | `gemini-3-flash-preview` | `config.py:11` |
| `gemini_llm_fallback_model` | `gemini-2.5-flash` | `config.py:12` |
| `gemini_embedder_model` | `gemini-embedding-001` | `config.py:13` |
| `neo4j_user` | `neo4j` | `config.py:17` |
| `neo4j_database` | `memory` | `config.py:19` |
| `qdrant_url` | `None` (falls back to on-disk) | `config.py:26` |
| `qdrant_on_disk` | `True` | `config.py:27` |
| `qdrant_path` | `~/.neuralscape/qdrant` | `config.py:28` |
| `qdrant_collection` | `neuralscape_memories` | `config.py:29` |
| `arq_queue_name` | `neuralscape:queue` | `config.py:33` |
| `arq_max_retries` | `3` | `config.py:34` |
| `arq_job_timeout` | `300` (sec) | `config.py:35` |
| `llm_max_retries` | `3` | `config.py:38` |
| `llm_retry_base_delay` | `1.0` | `config.py:39` |
| `llm_retry_max_delay` | `30.0` | `config.py:40` |
| `dedup_similarity_threshold` | `0.95` | `config.py:43` |
| `dedup_batch_size` | `100` | `config.py:44` |
| `dedup_cron_hours` | `{0, 6, 12, 18}` | `config.py:45` |
| `neuralscape_api_key` | empty (auth disabled) | `config.py:48` |
| `host` / `port` | `0.0.0.0` / `8199` | `config.py:51-52` |
| `default_user_id` | `default_user` | `config.py:53` |
| `mcp_transport` | `stdio` | `config.py:55` |

`get_mem0_config()` (`config.py:79`) assembles the dict mem0 expects (LLM/embedder/vector_store/graph_store). `parse_redis_settings()` (`config.py:138`) turns `REDIS_URL` into ARQ's `RedisSettings` with sensible reconnect defaults. See [06-storage-backends](./06-storage-backends.md) for what each provider expects.

## Logging

Configured in `neuralscape-service/logging_config.py` via structlog. Two knobs:

- `LOG_LEVEL` (default `INFO`)
- `LOG_FORMAT` (`json` default; `console` for local dev)

The processor chain is `merge_contextvars → add_log_level → add_logger_name → TimeStamper(iso) → StackInfoRenderer → UnicodeDecoder`, then either `JSONRenderer` or `ConsoleRenderer` (`logging_config.py:23-35`). Output goes to stdout so docker captures it. Per-logger noise reduction at `logging_config.py:63-64` clamps `httpcore`, `httpx`, `neo4j`, `urllib3`, and `uvicorn.access` to WARNING — keep this in mind if you are debugging Neo4j driver issues and need to bump that one up.

## Scripts

- `scripts/sync-upstream.sh` (57 lines) — `git subtree pull` for graphiti, mem0, or both. Warns about the `configs.py` / `factory.py` merge conflict zones in mem0, then runs `uv sync` and `pytest`. Usage: `./scripts/sync-upstream.sh [graphiti|mem0|all]`.
- `scripts/migrate-group-ids.cypher` (43 lines) — one-shot Neo4j migration that rewrites the legacy `project:{id}` group_id format to `project--{id}`. Touches `EpisodicNode`, `EntityNode`, `EntityEdge` (`RELATES_TO`), and `CommunityNode`. Verification query at lines 42-43. Run with `cypher-shell -u neo4j -p <pwd> < scripts/migrate-group-ids.cypher`.

## CI

There is no `.github/workflows/` directory at the repo root — neuralscape itself has no CI, no release automation, no published image. The graphiti and mem0 subtrees ship their own workflows (11 and 8 respectively) but those run only against their own upstream repos. Versioning is manual: `version="0.2.0"` is hardcoded in `main.py:175` (and `0.1.0` in `pyproject.toml`); tag releases by hand.

## Production gotchas

- **Default Neo4j credentials.** docker-compose sets `neo4j/neuralscape` if `NEO4J_PASSWORD` is unset (`docker-compose.yml:8`). Always override before exposing the bolt port.
- **Neo4j database name.** Defaults to `memory` (`docker-compose.yml:9`); the database must exist or Graphiti writes will fail with `DatabaseNotFound`.
- **Qdrant collection.** Auto-created on first write — pre-create only if you want non-default HNSW params.
- **Gemini quotas.** Free-tier limits are tight (~10 req/min). On 429/503, `memory_service.py` retries with exponential backoff capped at `llm_retry_max_delay` (default 30s) and falls back to `gemini-2.5-flash`. Bump the cap if you keep exhausting budget. See [05-llm-extraction](./05-llm-extraction.md).
- **API key rotation.** `NEURALSCAPE_API_KEY` is stateless; update `.env` and restart both API and worker.
- **Redis persistence.** Default image has no RDB or AOF tuning; the queue lives in `redis_data` but is best-effort. Configure `appendonly yes` for stronger guarantees.
- **Vault path.** `OBSIDIAN_VAULT_PATH` defaults to `./vault`; `mkdir -p vault` before `compose up` or the bind mount mounts an empty dir.
- **Group-ID migration.** Pre-2024 data uses `project:{id}`; symptom is project-scoped graph search returning nothing. Run `migrate-group-ids.cypher` to fix. See [03-memory-model](./03-memory-model.md).
- **Worker scaling.** Single worker by default. Scale horizontally by spinning up more `neuralscape-worker` replicas pointing at the same Redis — ARQ handles fair dispatch.

## Local dev quickstart

```bash
git clone <repo> neuralscape && cd neuralscape
cp .env.example .env  # set GOOGLE_API_KEY, override NEO4J_PASSWORD
mkdir -p vault
docker compose up --build -d
curl http://localhost:8199/health  # {"status":"ok"}
```

For unit tests without bringing up the stack:

```bash
cd neuralscape-service && uv sync
uv run pytest tests/ --ignore=tests/test_async_pipeline.py -v
```

Integration tests (`tests/test_async_pipeline.py`) require Neo4j, Redis, and Qdrant running. See [10-testing](./10-testing.md).

## Related

- [02-service-architecture](./02-service-architecture.md) — runtime topology these services implement
- [07-async-pipeline](./07-async-pipeline.md) — what the worker container actually does
- [06-storage-backends](./06-storage-backends.md) — provider config consumed by `get_mem0_config()`
- [10-testing](./10-testing.md) — unit vs integration split, what needs the stack
- [00-overview](./00-overview.md) — top-level index
