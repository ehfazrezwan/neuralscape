---
title: Deployment, Config & Observability
date: 2026-05-06
tags: [reference, neuralscape, deployment, docker, config, ops]
source: handwritten
---
 
# Deployment, Config & Observability

Operational reference for running neuralscape locally and in production. Covers the docker compose stack, the shared Dockerfile, environment variables, structured logging, the few helper scripts, the CI and image-build gate, and the gotchas you will hit if you skip the env file.

## Overview

Neuralscape ships as a Docker Compose stack. The always-on services are three data stores (Neo4j, Redis, Qdrant), a Docling document-parsing service, the FastAPI server, and **three** ARQ workers (fast, graph, ingest; see [07-async-pipeline](./07-async-pipeline.md)). Optional services sit behind compose profiles (code-intel sidecars, a tunnel, a test runner), and an autoheal pair restarts wedged containers. The API and the three workers share one image built from `neuralscape-service/Dockerfile`; only the entrypoint differs (`uvicorn` vs `arq worker.<Settings>`). There is also a generic, parameterized Kubernetes deployment under `deploy/` (Terraform, a Helm chart and plain manifests), described below. See [02-service-architecture](./02-service-architecture.md) for the runtime layout.

## Docker Compose reference

The compose file is `docker-compose.yml` at the repo root. Every published port is bound to `127.0.0.1`, and services use json-file log rotation, memory limits and health checks. Read the file for the exact current limits and image pins; the roles are:

| Service | Role | Profile | Notes |
|---------|------|---------|-------|
| neo4j | Knowledge graph (Graphiti) | always | `neo4j:5-community`; `NEO4J_PASSWORD` required |
| redis | ARQ queues, task status, small config stores | always | pinned 7.4 series (see `THIRD_PARTY_NOTICES.md`) |
| qdrant | Vector store | always | pinned version |
| docling | Rich-format document parsing for ingest | always | pinned image, port 5001 |
| neuralscape | FastAPI REST + MCP HTTP transport | always | port 8199; `/health` |
| neuralscape-worker | Fast ARQ worker (`worker.WorkerSettings`) | always | vector writes/reads, conversation tasks |
| neuralscape-graph-worker | Graph ARQ worker (`worker.GraphWorkerSettings`) | always | Graphiti enrichment, dedup / dreaming / synthesis crons |
| neuralscape-ingest-worker | Ingest ARQ worker (`worker.IngestWorkerSettings`) | always | bulk ingest, connector sync, code indexing |
| cbm-bridge | REST shim over a prebuilt code-intel binary | `cbm` | also set `CBM_ENABLED=true` |
| resolver-svc | Precise-neighbors resolver (pyright over LSP) | `resolver` | index-time only; also set `CODE_NEIGHBORS_RESOLVER=lsp` |
| cloudflared | Optional tunnel | `tunnel` | for exposing the API without opening ports |
| neuralscape-test | In-container test runner | `test` | uses the Dockerfile `test` stage |
| autoheal, docker-socket-proxy | Restart containers labelled `com.neuralscape.autoheal.<project>=true` (`<project>` is the compose project name, `neuralscape` by default) | always | the proxy limits what autoheal can ask the Docker socket to do; the per-project label key keeps two stacks on one host from restarting each other's containers |

Other compose files at the repo root cover deployment variants (`docker-compose.prod.yml` adds a Caddy reverse proxy, plus `docker-compose.vps.yml`, `docker-compose.deploy.yml` and `docker-compose.snapshot-test.yml`).

### Volumes

Named volumes hold the data stores and artifacts (`neo4j_data`, `redis_data`, `qdrant_data`, `ingest_data` for uploaded documents, `exemplar_data` for visual exemplars, plus build caches), and a host bind mount carries the Obsidian vault: `${OBSIDIAN_VAULT_PATH:-./vault}:/data/vault`. The API and the workers share the vault and the ingest volume so they can read and write the same files.

### Env injection (API + workers)

The app services pull from `.env` and override container-network values inline (abridged):

```yaml
NEO4J_URI: neo4j://neo4j:7687
NEO4J_PASSWORD: ${NEO4J_PASSWORD:?set NEO4J_PASSWORD in .env}
NEO4J_DATABASE: ${NEO4J_DATABASE:-memory}
QDRANT_URL: http://qdrant:6333
REDIS_URL: redis://redis:6379
GEMINI_LLM_MODEL: ${GEMINI_LLM_MODEL:-gemini-3.1-flash-lite}
GEMINI_EMBEDDER_MODEL: ${EMBEDDING_MODEL_NAME:-gemini-embedding-001}
MCP_TRANSPORT: http
OBSIDIAN_VAULT_PATH: /data/vault
```

### Startup ordering

The app services declare `depends_on` with `condition: service_healthy` for the data stores. The API will not start until Neo4j answers `RETURN 1`, Redis answers `PING`, and Qdrant accepts a TCP connection. This eliminates the cold-start race where a worker boots before Neo4j is reachable.

## Dockerfile

`neuralscape-service/Dockerfile` is a multi-stage build on `python:3.12-slim`, with the `uv` binary pinned by version and digest:

- **`builder`** copies the dependency manifests for graphiti, mem0 and neuralscape-service, copies the source for the editable subtree packages, and runs `uv sync --frozen --no-dev --extra code-graph` to materialize a `.venv`. It also installs the spaCy model wheel (see [`neuralscape-service/vendor/`](../../neuralscape-service/vendor/README.md)).
- **`runtime-src`** copies the top-level service modules and prunes the ones no runtime entry point imports, so they never enter the runtime image's layers.
- **`runtime`** copies the venv and the source needed at runtime, runs as a non-root `appuser`, and starts uvicorn on port 8199.
- **`test`** adds the dev dependencies, the tests and the integration tests, so the suite can run inside the packaged image.

The Dockerfile enumerates its COPY paths per directory. **Any new top-level file or directory under `neuralscape-service/` must be added to the COPY lists in all three of builder, runtime and test**, or it passes host tests and crash-loops in the image; this is the Container Build Gate rule in `CLAUDE.md`, and `tests/test_dockerfile_runtime.py` guards parts of it.

The compose workers override the command: `["arq", "worker.WorkerSettings"]`, `["arq", "worker.GraphWorkerSettings"]` and `["arq", "worker.IngestWorkerSettings"]`. One image, four roles; see [07-async-pipeline](./07-async-pipeline.md) for the workers' job semantics.

## Configuration

All settings live in `neuralscape-service/config.py` as a Pydantic `BaseSettings` class loading `.env`. The tables below list the long-standing core settings only; `config.py` is the source of truth, and it has many more (feature flags, queues, auth, ingest, code-intel). `Settings.validate_required()` in `config.py` is called once at startup from `main.py` and aborts boot if any of the four required vars below is missing. It also aborts on a missing auth configuration, on missing LLM-gateway settings when the gateway is enabled, and on a missing connector vault key when connectors are enabled; read the method for the full list.

### Required (validated at boot)

| Setting | Env var | Default | Source |
|---------|---------|---------|--------|
| google_api_key | `GOOGLE_API_KEY` | empty | `config.py` |
| neo4j_password | `NEO4J_PASSWORD` | empty | `config.py` |
| neo4j_uri | `NEO4J_URI` | `neo4j://127.0.0.1:7687` | `config.py` |
| redis_url | `REDIS_URL` | `redis://localhost:6379` | `config.py` |

### Optional (with defaults)

| Setting | Default | Source |
|---------|---------|--------|
| `gemini_llm_model` | `gemini-3.1-flash-lite` | `config.py` |
| `gemini_llm_fallback_model` | `gemini-2.5-flash` | `config.py` |
| `gemini_embedder_model` | `gemini-embedding-001` | `config.py` |
| `neo4j_user` | `neo4j` | `config.py` |
| `neo4j_database` | `memory` | `config.py` |
| `qdrant_url` | `None` (falls back to on-disk) | `config.py` |
| `qdrant_on_disk` | `True` | `config.py` |
| `qdrant_path` | `~/.neuralscape/qdrant` | `config.py` |
| `qdrant_collection` | `neuralscape_memories` | `config.py` |
| `arq_queue_name` | `neuralscape:queue` | `config.py` |
| `arq_max_retries` | `3` | `config.py` |
| `arq_job_timeout` | `300` (sec) | `config.py` |
| `llm_max_retries` | `3` | `config.py` |
| `llm_retry_base_delay` | `1.0` | `config.py` |
| `llm_retry_max_delay` | `30.0` | `config.py` |
| `dedup_similarity_threshold` | `0.95` | `config.py` |
| `dedup_batch_size` | `100` | `config.py` |
| `dedup_cron_hours` | `{0, 6, 12, 18}` | `config.py` |
| `neuralscape_api_key` | empty (startup refuses unless a secret or `NEURALSCAPE_AUTH_DISABLED=true` is set; see AUTH.md) | `config.py` |
| `neuralscape_api_key_user_id` | empty (the shared key acts as `default_user_id`; see AUTH.md) | `config.py` |
| `neuralscape_legacy_key_trust_body_user_id` | `false` (`true` lets the shared key act as any body `user_id`) | `config.py` |
| `neuralscape_revoked_tokens` | empty (CSV of SHA-256 digests of revoked per-user tokens) | `config.py` |
| `host` / `port` | `0.0.0.0` / `8199` | `config.py` |
| `default_user_id` | `default_user` | `config.py` |
| `mcp_transport` | `stdio` | `config.py` |

`get_mem0_config()` (`config.py`) assembles the dict mem0 expects (LLM/embedder/vector_store/graph_store). `parse_redis_settings()` (`config.py`) turns `REDIS_URL` into ARQ's `RedisSettings` with sensible reconnect defaults. See [06-storage-backends](./06-storage-backends.md) for what each provider expects.

### Third-party library telemetry (off by default)

The vendored mem0 and Graphiti libraries each ship with anonymous usage telemetry enabled by default, which sends events to an external analytics host. Neuralscape turns both off unless the operator opts in:

| Env var | Library | Shipped default | Opt in |
|---------|---------|-----------------|--------|
| `MEM0_TELEMETRY` | mem0 (read once, when mem0 is first imported) | `false` | `true` |
| `GRAPHITI_TELEMETRY_ENABLED` | Graphiti (read on every event) | `false` | `true` |

The default is applied in three layers. The code and image layers keep any explicitly set value, including an empty one; the Compose layer keeps an explicit non-empty value (Compose's `${VAR:-false}` treats an empty value as unset). Either way an empty value means off in both libraries:

- **Code:** `neuralscape-service/third_party_telemetry.py` fills each variable missing from the process environment: from the service `.env` (the file `config.Settings` reads, relative to the working directory) when that file sets it, otherwise `false`. Only these two names are taken from the file, and the process environment wins over it, as it does for `Settings`. Both keys are declared settings, so a service `.env` containing them is valid. `main`, `worker`, `mcp_server`, `memory_service` and the `memory` package import it before anything that can load mem0 or Graphiti, so scripts and library use get the same default. `tests/test_third_party_telemetry.py` enforces that import order.
- **Image:** the runtime and test stages of the Dockerfile set both variables to `false`.
- **Observable state:** `GET /health` reports `checks.third_party_telemetry` as `{"mem0": "off"|"on", "graphiti": "off"|"on"}` (on/off only, never an endpoint, key or id, and it never changes the status code or the `status` value), and the API and every worker log one line at startup, `third_party_telemetry mem0=off graphiti=off`. Graphiti switches itself off under pytest; the report ignores that and states what the library would do in a real process.
- **Deployment templates:** `docker-compose.yml` and `docker-compose.prod.yml` pass `${MEM0_TELEMETRY:-false}` and `${GRAPHITI_TELEMETRY_ENABLED:-false}`, so setting either one in the Compose project's `.env` (the repository root) opts in. The Helm chart's `config` values set both to `"false"`.

Inside the shipped image both variables are already set in the process environment, so a `.env` file inside the container cannot opt in; use `docker run -e …` or the Compose project's `.env`. The service `.env` route applies to a local `uv run`.

The guard covers the service processes. The benchmark harness `neuralscape-bench/trackb/mem0_tracka/run.py` imports mem0 directly; set `MEM0_TELEMETRY=false` in its environment when running it.

The vendored subtrees are not modified, so an upstream sync cannot silently turn telemetry back on.

To opt in, set either variable to `true` in the process environment, in the Compose project's `.env` (for `docker compose`), or in `neuralscape-service/.env` (for a local `uv run` from that directory).

## Logging

Configured in `neuralscape-service/logging_config.py` via structlog. Two knobs:

- `LOG_LEVEL` (default `INFO`)
- `LOG_FORMAT` (`json` default; `console` for local dev)

Both (and `NEURALSCAPE_EXTENSIONS`) are settings: the process environment wins when it sets them, otherwise the service `.env` applies. Both env templates (`.env.example`, `.env.production`) load as `neuralscape-service/.env` for a local run; keys marked "Compose-only" in them, and keys owned by the extension settings (`DREAMING_*`, `STRATEGY_SYNTHESIZER_*`, the conversation compiler's keys), are accepted there, while any other unknown key is still rejected (`tests/test_env_templates.py`).

The processor chain is `merge_contextvars → add_log_level → add_logger_name → TimeStamper(iso) → StackInfoRenderer → UnicodeDecoder`, then either `JSONRenderer` or `ConsoleRenderer` (`logging_config.py`). Output goes to stdout so docker captures it. Per-logger noise reduction in `logging_config.py` clamps `httpcore`, `httpx`, `neo4j`, `urllib3`, and `uvicorn.access` to WARNING — keep this in mind if you are debugging Neo4j driver issues and need to bump that one up.

## Scripts

- `scripts/sync-upstream.sh` — pulls upstream into the graphiti and mem0 subtrees from a locally built filtered mirror, then runs `uv sync` and `pytest`. Warns about the `configs.py` / `factory.py` merge conflict zones in mem0. Usage: `./scripts/sync-upstream.sh [graphiti|mem0|all]`. `scripts/check-upstream-patches.sh` with `scripts/upstream-patches.txt` checks that NS patches to the subtrees survive a sync.
- `scripts/backup-volumes.sh` — consistent backup of the data volumes; `--help` carries the restore steps.
- `scripts/migrate-group-ids.cypher` and `scripts/migrate_wiki_layout.py` — one-shot migrations for old data layouts. Read the header of each before running it; they may be retired or changed independently of this page.

## CI and the image gate

CI lives in `.github/workflows/`:

- **`service-ci`** runs on every pull request and on pushes to `dev` and `main`. It enforces the Container Build Gate from `CLAUDE.md`: subtree-prune guard, host tests, a `container-gate` job that builds the image and runs the suite inside it, and a runtime-image build. Its aggregate `service-gate` job succeeds only when every qualification job succeeds (skipped, cancelled or failed jobs do not satisfy it).
- **`plugin-ci`** guards the plugin's committed esbuild bundles, manifest version drift and coverage thresholds.
- **`sidecars-ci`** covers the sidecar images (`resolver-svc`, `cbm-bridge`), renders the Helm chart, and runs the acceptance harness's own self-tests.
- **`bench-ci`** covers `neuralscape-bench/`.
- **`deploy`** builds the runtime image on pushes to `main` and rolls it onto the production host; it is the only deploy path.
- **`dependabot-auto-merge`** auto-merges low-risk dependency PRs once required checks pass.

Versioning is manual: the API version string is set in `main.py` and the packaging version in `pyproject.toml`; tag releases by hand. Merges into `dev` need a green `service-gate`; releases go `dev` to `main` via a release PR.

## Kubernetes (generic)

`deploy/` holds a generic, parameterized deployment onto an existing Kubernetes cluster: Terraform for secrets, IAM and the Helm release, a Helm chart (`deploy/charts/neuralscape`: API with an HPA, fast, graph and ingest workers, Docling, a service account, an ingest volume claim), and plain manifests for Neo4j, Qdrant and Redis. It is deliberately free of any environment- or organization-specific value: the real project, cluster, domain, registry and state-backend settings are supplied at deploy time from private configuration. Start with `deploy/README.md`.

## Production gotchas

- **Neo4j credentials.** There is no default password: `docker compose` refuses to start until `NEO4J_PASSWORD` is set in `.env`. Set it before the first start (Neo4j keeps the password it was first started with in `neo4j_data`). Every published port is bound to `127.0.0.1`.
- **Backups.** `scripts/backup-volumes.sh` takes a consistent backup of all five data volumes (offline Neo4j dump, Qdrant snapshots, Redis BGSAVE, ingest/exemplar tar) with retention; its `--help` text carries the restore steps.
- **Neo4j database name.** Defaults to `memory`; the database must exist or Graphiti writes will fail with `DatabaseNotFound`.
- **Qdrant collection.** Auto-created on first write — pre-create only if you want non-default HNSW params.
- **Gemini quotas.** Free-tier limits are tight (~10 req/min). On 429/503, `retry_transient` (`memory/retry.py`) retries with exponential backoff capped at `llm_retry_max_delay` (default 30s) and falls back to `gemini-2.5-flash`. Bump the cap if you keep exhausting budget. See [05-llm-extraction](./05-llm-extraction.md).
- **API key rotation.** `NEURALSCAPE_API_KEY` is stateless; update `.env` and restart both API and worker.
- **Redis persistence.** Default image has no RDB or AOF tuning; the queue lives in `redis_data` but is best-effort. Configure `appendonly yes` for stronger guarantees.
- **Vault path.** `OBSIDIAN_VAULT_PATH` defaults to `./vault`; `mkdir -p vault` before `compose up` or the bind mount mounts an empty dir.
- **Group-ID migration.** Very old data used a `project:{id}` group id; the symptom is project-scoped graph search returning nothing. Read `scripts/migrate-group-ids.cypher` and [03-memory-model](./03-memory-model.md) (current group-id format) before running anything.
- **Worker scaling.** One replica of each of the three workers by default. Scale horizontally by adding replicas of the worker that is the bottleneck, pointing at the same Redis; ARQ handles fair dispatch within a queue.

## Local dev quickstart

```bash
git clone <repo> neuralscape && cd neuralscape
cp .env.example .env  # set GOOGLE_API_KEY and a strong NEO4J_PASSWORD (required)
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
- [07-async-pipeline](./07-async-pipeline.md) — what the worker containers actually do
- `deploy/README.md` — the generic Kubernetes deployment
- [06-storage-backends](./06-storage-backends.md) — provider config consumed by `get_mem0_config()`
- [10-testing](./10-testing.md) — unit vs integration split, what needs the stack
- [00-overview](./00-overview.md) — top-level index
