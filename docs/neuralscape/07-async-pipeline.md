---
title: Async Write Pipeline
date: 2026-05-06
tags: [reference, neuralscape, async, arq, redis, worker]
source: handwritten
---

# Async Write Pipeline

> **Status (as of 2026-11-16, dev 6a81341f):** current reference. Code citations name modules and functions, not line numbers; the implementation lives in the `neuralscape-service/memory/` package behind the `memory_service.py` facade. Where a snippet is marked abridged it is a simplified excerpt, so read the named module for the real code.

Neuralscape splits memory operations along a sharp boundary: **writes are asynchronous**, **reads are synchronous**. A POST to `/v1/memories` returns `202 Accepted` with a `task_id`; the actual LLM extraction, embedding, vector upsert, and graph ingestion happen later in a separate ARQ worker process. A POST to `/v1/search` runs synchronously against Qdrant and Neo4j and returns `200 OK`. This page documents the full write-side machinery: the queue lifecycle, the worker, deterministic deduplication of enqueues, the status-polling protocol, retry behaviour, and failure modes.

## Overview

Why split writes from reads at all? Memory ingestion is expensive. A single conversation may invoke Gemini twice (extraction + dedup judging), batch-embed N facts, write each to Qdrant, then re-ingest the whole conversation as episodes into the Graphiti temporal graph. End-to-end latency can run from a few hundred milliseconds to tens of seconds. Forcing a client to wait synchronously for that would block conversational UIs and time out under load.

Reads, by contrast, are a single Qdrant ANN lookup with optional Neo4j re-ranking — fast enough to serve in-line. So Neuralscape funnels writes through Redis + ARQ and keeps reads on the request thread.

The full picture:

- **Client** issues `POST /v1/memories` (extraction) or `POST /v1/memories/raw` (pre-categorized fact).
- **API** (`main.py`) hashes the payload, enqueues an ARQ job, and returns `202` with `{task_id, poll_url}`.
- **Worker** (`arq worker.WorkerSettings` for the fast path; graph enrichment and bulk ingest run on `GraphWorkerSettings` and `IngestWorkerSettings`) — separate processes — dequeues, runs the actual storage logic in [04-memory-service-core](./04-memory-service-core.md), emits `memory_stored` events to extensions, and writes the result to Redis.
- **Client** polls `GET /v1/memories/status/{task_id}` until `status == "completed"` (or `"failed"`).

Cron jobs are spread over the three workers (see the table below): the conversation-compile check and memory expiry on the fast worker; deduplication, the dreaming sweep and the strategy-playbook synthesizer on the graph worker; connector sync on the ingest worker.

## Architecture

The end-to-end sequence for a write:

1. Client → `POST /v1/memories` with `{messages, user_id, project_id?, agent_id?, run_id?}`.
2. API constructs args, computes a SHA-256 job ID, calls `task_manager.enqueue_store(...)`.
3. ARQ pushes the job onto Redis list `arq:queue:neuralscape:queue` keyed by job ID.
4. API returns `202 {task_id, poll_url}` immediately.
5. Worker process (`arq worker.WorkerSettings`) blocks on `BRPOPLPUSH` against the queue, dequeues the job, marks state `in_progress` at `arq:job:<job_id>`.
6. Worker invokes `process_memory_store(ctx, messages, user_id, ...)`:
   - `MemoryService.extract_and_store()` → Gemini → batch embed → Qdrant upsert → Graphiti episode add.
   - For each returned memory, `await registry.emit_event("memory_stored", ...)` so the conversation-compiler extension can write to the Obsidian vault.
   - Single `_rebuild_category_index()` call after all events.
7. Worker stores the return value at `arq:result:<job_id>` and marks state `complete`.
8. Client polls `GET /v1/memories/status/{task_id}` → eventually `200 {status: "completed", result: {memories: [...]}}`.

The API process and the worker process share **only** Redis. They each maintain their own `MemoryService` instance, their own Qdrant client, their own Neo4j driver, their own extension registry. There is no in-process state shared between them — Redis is the single source of truth for task state. See [02-service-architecture](./02-service-architecture.md) for the broader process topology.

## WorkerSettings & ARQ Config

There are three ARQ worker classes in `neuralscape-service/worker.py`, each consuming its own queue so that slow graph work or a bulk ingest cannot starve latency-sensitive writes and reads. Run each as its own process with `arq worker.<Class>`:

| Class | Queue setting (default name) | Runs | Crons |
|---|---|---|---|
| `WorkerSettings` (fast) | `arq_queue_name` (`neuralscape:queue`) | `process_memory_store`, `process_memory_raw`, `process_memory_raw_batch`, `process_memory_retag`, `process_conversation_flush`, `process_conversation_compile` | `auto_compile_check`, `expire_old_memories_cron` |
| `GraphWorkerSettings` (slow) | `graph_queue_name` (`neuralscape:graph`) | `process_graph_enrichment`, `run_dream_sweep`, `process_session_summary` | dedup (`dedup_all_memories`, hours from `DEDUP_CRON_HOURS`; disabled when the list is empty, `[]`), `dream_sweep_cron`, `synthesize_strategy_playbooks_cron` |
| `IngestWorkerSettings` (bulk) | `ingest_queue_name` (`neuralscape:ingest`) | `process_ingest_document`, `process_ingest_file`, `process_ingest_okf_bundle`, `process_connector_sync`, `process_code_index` | `connector_sync_cron` |

For example, `process_memory_raw` does the vector write and defers Graphiti enrichment to a `process_graph_enrichment` job on the graph queue, so the task result reflects the vector write while enrichment finishes later on the graph worker. The shape of the fast worker's settings is:

```python
class WorkerSettings:
    functions = [process_memory_store, process_memory_raw, process_memory_raw_batch,
                 process_memory_retag, process_conversation_flush, process_conversation_compile]
    cron_jobs = [cron(auto_compile_check, ...), cron(expire_old_memories_cron, ...)]
    redis_settings = parse_redis_settings()
    queue_name = settings.arq_queue_name        # "neuralscape:queue"
    max_jobs = settings.arq_max_jobs            # 10 default (ARQ_MAX_JOBS)
    job_timeout = settings.arq_job_timeout      # 300s default
    max_tries = settings.arq_max_retries        # 3 default
```

Key knobs:

- `max_jobs` (`ARQ_MAX_JOBS`, default 10) — task coroutines run concurrently per worker process. The graph worker reads `GRAPH_MAX_JOBS` (default 4) and the ingest worker `INGEST_MAX_JOBS` (default 3). Prefer adding worker replicas to raising these.
- `job_timeout = 300s` — a single task that exceeds 5 minutes is killed and counted as a failure (subject to `max_tries`).
- `max_tries = 3` — if the task raises any uncaught exception, ARQ re-enqueues it up to two more times with default backoff. After the third failure the job is marked permanently failed.
- `queue_name = "neuralscape:queue"` — the API and worker must agree on this. Mismatched queue names silently mean enqueued jobs never run.

Redis connection settings are parsed by `parse_redis_settings()` in `neuralscape-service/config.py` (imported by `worker.py`): 5 connection retries, 2-second delay between retries, 10-second socket timeout. Default URL `redis://localhost:6379`. The same `parse_redis_settings()` helper is used by both the API's `TaskManager` and `WorkerSettings`, ensuring consistency.

The cron schedule for `dedup_all_memories` resolves to hours `{0, 6, 12, 18}` UTC (every 6 hours), with `unique=True` so concurrent worker replicas can't double-run it. `auto_compile_check` fires at `:30` between 6 PM and midnight UTC.

## Task Functions

### process_memory_store

`process_memory_store` in `neuralscape-service/worker.py`. Signature:

```python
async def process_memory_store(
    ctx, messages, user_id,
    project_id=None, agent_id=None, run_id=None,
) -> dict
```

Steps:

1. Pull the cached `MemoryService` from `ctx["service"]` (set in `startup`).
2. Call `service.extract_and_store(...)` — the full extraction pipeline described in [05-llm-extraction](./05-llm-extraction.md) and [04-memory-service-core](./04-memory-service-core.md).
3. For each fact returned, emit a `memory_stored` event through the [09-plugin-system](./09-plugin-system.md) registry. Extensions (notably the conversation-compiler) react by writing markdown into the Obsidian vault.
4. After the per-fact loop, call `_rebuild_category_index(registry)` exactly once. This is deliberate: the index walks the vault, so doing it per-fact would be O(N²).
5. Return `{"memories": [m.model_dump(exclude_none=True) for m in memories]}`.

### process_memory_raw

`process_memory_raw` in `neuralscape-service/worker.py`. Signature mirrors the request body for `POST /v1/memories/raw`. The notable wrinkle is the **idempotency check near the top of the function** (abridged; the real check has grown):

```python
existing = service.search(query=content, user_id=user_id, project_id=project_id, limit=3)
for mem in existing:
    if mem.memory.strip().lower() == content.strip().lower():
        logger.info(f"Skipping duplicate memory for user {user_id}: ...")
        return {"memories": [mem.model_dump(exclude_none=True)], "deduplicated": True}
```

Before storing, the worker performs a vector search and case-insensitively compares the top hits against the new content. If an exact (whitespace- and case-normalised) match exists, it short-circuits and returns the existing memory with `deduplicated: True`. This is in addition to the deterministic-job-ID dedup at the enqueue layer — they catch different cases. Job-ID dedup catches a client retrying the *same* request; idempotency catches a client posting the same fact twice through different code paths or sessions.

Failures of the idempotency search are logged and swallowed: better to risk a duplicate than to drop a write because Qdrant blipped.

### process_memory_retag

The bulk memory-edit job (`POST /v1/memories/retag`, MCP `retag_memories`). Runs on the **fast queue** because the retag itself is a Qdrant scroll + per-point `set_payload` (milliseconds each); when a `set_project_id` op moves memories between graph `group_id` partitions, the service returns one graph job per moved memory and the worker fans them out onto the **graph queue** via `_enqueue_graph_jobs` — the slow Graphiti re-ingest never runs on the fast worker. Job id is a canonical-JSON hash of `{filters, ops}` + caller, so two different retags can't collide while an identical replay coalesces. Single-memory edits (`PATCH /v1/memories/{id}`) don't have a worker function at all: the vector write is synchronous in the request, and only the graph half is enqueued via `TaskManager.enqueue_graph_enrichment` (job id keyed on the edit's *target state* — memory + visibility + project + content hash).

## TaskManager: Deterministic Job IDs

`neuralscape-service/task_manager.py` is the API-side wrapper around the ARQ Redis pool. Its single most important behaviour is **deterministic job IDs**, generated by `_generate_job_id` in `task_manager.py` (abridged):

```python
def _generate_job_id(content: str, user_id: str) -> str:
    h = hashlib.sha256(f"{user_id}:{content}".encode()).hexdigest()[:16]
    return f"ns-{h}"
```

`TaskManager.enqueue_store` derives its key from the message contents and prefixes `store:` (it also folds in the scope and `occurred_at`) before hashing; `enqueue_raw` prefixes `raw:`; the other enqueue methods use their own prefixes. The 16-hex-char prefix gives 64 bits of collision resistance — more than enough for a per-user namespace.

Why deterministic? ARQ skips enqueues whose `_job_id` already exists, returning `None`. So if a flaky client retries the same `POST /v1/memories` three times, all three calls produce the *same* job ID, only the first creates a queue entry, and the API returns the same `task_id` every time. Polling that ID converges to the same final result. This makes the write endpoint safely retriable from the client side without any token or `Idempotency-Key` header.

The trade-off: if a user posts genuinely identical content twice on purpose, the second call returns the first call's task ID rather than creating a fresh job. In practice this is what callers want — the data they wanted to store is already stored.

## HTTP Status Lifecycle

The state machine the client observes through `/v1/memories/status/{task_id}`:

| ARQ JobStatus | API status   | Meaning                                                              |
|---------------|--------------|----------------------------------------------------------------------|
| deferred      | queued       | Scheduled but not yet on the active queue                            |
| queued        | queued       | On the queue, awaiting a worker                                      |
| in_progress   | processing   | A worker has picked it up                                            |
| complete + ok | completed    | Task returned successfully; `result` field populated                 |
| complete + ng | failed       | Task raised after exhausting retries; `error` populated              |
| not_found    | not_found    | Job ID unknown to ARQ — surface as HTTP 404 in the API               |

**Ownership.** With a per-user token, a task's status (state, result and error) is readable only by the user who enqueued it. The enqueue path records the owner under `ns:task-owner:{task_id}` (a set, kept for two days so it outlives ARQ's one-day queue expiry and one-hour result retention), and both `GET /v1/memories/status/{task_id}` and the legacy `GET /memories/status/{task_id}` answer any other user with the same 404 as an unknown id. The owner is checked before the job is looked up. With the shared API key, or with auth disabled, requests carry no caller identity and any caller may write as any user, so status reads are not owner-checked.

Every enqueue that hands a task id to a caller records the owner, not only the `TaskManager` ones: the conversation-compiler `/flush` and `/compile` routes and its auto-compile on session end, and dreaming sweeps started through `POST /v1/extensions/dreaming/run` or the `schedule_dream` MCP tool. The compiler routes record the body's `user_id` only when the caller is that user; a per-user token naming a different user records no owner, so nobody can poll that task with a per-user token. Jobs that workers enqueue for themselves (graph enrichment, session summaries, the connector-sync cron) hand no id to a caller and record no owner. The set member and its TTL are written in one `MULTI`/`EXEC`, so an owner key is never left without an expiry. If Redis fails during the owner check, both status routes answer `503` (with `Retry-After`) for every id alike, before the job is looked up. A `404` would tell the real owner that their task is gone, so a client might resubmit the write. A `503` is retryable and reveals nothing about whether the task exists or who owns it.

Queue accounting and read access are recorded separately. A raw batch whose items belong to more than one user (only possible with the shared key) is counted in every writer's queue status, but its result holds all items, so it grants no per-user read: only shared-key (or auth-disabled) callers can poll it. A single-user batch is readable by that user as usual. A graph re-ingest started by a memory edit writes to the memory owner's graph partition, but the task belongs to the user who made the edit (the one who receives `graph_task_id`), so a teammate's allowed edit of a shared memory is pollable by that teammate. Per-task status reads also search the graph queue, so a graph task reports `queued` while it waits instead of `not_found`; `GET /v1/queue/status` counts are computed over the same queues as before. Connector syncs use a job id derived from the connector id, its owner and the vault record's `generation` (assigned at registration, kept when the same owner re-registers, renewed when the connector is re-created or registered by another owner); the API route and the sync cron derive it the same way, so their syncs of one connector instance coalesce, while a re-created or re-owned connector never shares a cached job, result or grant with the previous one. A re-created or re-owned connector also starts with empty sync state (cursor, `last_synced_at`, per-resource revisions); only a same-owner re-register keeps it. Vault writes are atomic compare-and-set operations on the stored record, and a sync writes only its sync-state fields and only while the stored record is still the generation it read, so a sync job for an older generation never writes the newer instance's cursor and never reverts a concurrent delete, re-registration or credential change. With a per-user token, `POST /v1/connectors` for a connector id owned by another user answers the same 404 as the other connector routes; only shared-key (or auth-disabled) callers can register over another owner's connector, which starts a new instance.

The mapping lives in `task_manager.py` (the state mapping and the success-vs-failure branch in the status poll). Note ARQ does not have a separate `failed` enum value: a failed job is `complete` with `info.success == False`, and `TaskManager.get_status` rewrites it to `"failed"`.

A successful poll returns:

```json
{
  "task_id": "ns-3f2a9b...",
  "status": "completed",
  "result": {
    "memories": [
      {
        "id": "uuid-...",
        "memory": "User prefers TypeScript over JavaScript",
        "category": "preference",
        "scope": "global",
        "project_id": null,
        "tags": ["lang"],
        "source": "vector",
        "created_at": "2026-05-06T12:34:56Z",
        "updated_at": null
      }
    ],
    "deduplicated": false
  },
  "error": null
}
```

ARQ stores job state at `arq:job:<job_id>` and the result at `arq:result:<job_id>`. There is no explicit TTL set in `WorkerSettings`, so ARQ's default of roughly 500 days applies. For high-volume installs this means stale keys accumulate forever — a janitor cron is **not implemented** and would be worth adding.

### The Redis-down sync fallback

There is one important footgun in the store endpoints in `main.py` (`v1_store_memories` and its raw/batch siblings). If `task_manager.enqueue_store` raises `ConnectionError` or `OSError` — i.e. Redis is unreachable — the API does not fail. Instead it falls back to running the storage **synchronously on the request thread** and returns `200` with the `StoreMemoryResponse` body directly:

```python
except (ConnectionError, OSError) as e:
    logger.warning(f"Redis unavailable, falling back to sync store: {e}")
    memories = await asyncio.to_thread(_service.extract_and_store, ...)
    return JSONResponse(status_code=200,
                       content=StoreMemoryResponse(memories=memories)...)
```

This is a survival mechanism: the service stays usable when Redis is down, at the cost of multi-second response times. But it is also a **silent contract change**. A client that always expects `202 + task_id` will be confused by a `200` containing the actual memories. Clients should branch on the status code, not assume async semantics.

## Periodic Cron Jobs

The crons are split across the three workers. The two long-standing ones:

**`dedup_all_memories`** (graph worker) runs at the hours in `DEDUP_CRON_HOURS` (default `00:00`, `06:00`, `12:00`, `18:00` UTC; set it as a comma list `0,6` or a JSON list; blank means the default and `[]` disables it) and is not subject to an ARQ timeout, because it offloads blocking work to threads. It iterates `service.get_all_user_ids()` (paginated by `dedup_batch_size`) and runs `service.dedup_memories(uid)` for each. The two-phase algorithm — exact MD5 hash followed by semantic cosine ≥ 0.95 — is documented in [04-memory-service-core](./04-memory-service-core.md). Per-user errors are caught inside the `dedup_all_memories` loop (`worker.py`) so one user's failure does not abort the sweep; the failed user gets `{"user_id": uid, "error": str(e)}` in the per-user list. The summary aggregates `total_exact_removed` and `total_semantic_removed` for the run.

**`auto_compile_check`** runs at `:30` past every hour from 18:00 to 23:00 UTC. It scans for un-compiled conversation dates in the vault and fires `compile_all_pending` to consolidate them. This is an extension-driven feature; see the conversation-compiler extension docs for details.

The crons are marked `unique=True` so multiple worker replicas cannot run them concurrently — ARQ uses a Redis SET-NX lock per cron name.

## Failure Modes

**Transient errors from LLM/Neo4j/Qdrant.** Detected by `_is_transient()` in `memory/retry.py`, which matches `503`, `429`, `UNAVAILABLE`, `RESOURCE_EXHAUSTED`, the substring `"rate limit"`, and `"timeout"`. The wrapper `retry_transient()` in `memory/retry.py` retries with `min(base * 2^attempt + jitter, max)`, defaulting to base 1s, max 30s, and 3 retries (`llm_retry_*` settings in `config.py`). After exhausting the primary model, it falls back to the configured fallback model (`gemini-2.5-flash` by default). This handles **most** transient blips inside a single task invocation.

**ARQ-level retries.** If a task still raises after `retry_transient` is done, ARQ catches the exception, increments the try counter, and re-enqueues. With `max_tries = 3`, a task can run up to three times total before being marked failed. ARQ uses its default backoff schedule between retries.

**Non-transient errors.** A `ValueError`, `KeyError`, or category-validation failure is raised straight up. ARQ still applies `max_tries = 3`, but each retry will hit the same deterministic failure, so the job effectively fails immediately from the user's perspective.

**Poison messages.** If the function signature changes (e.g. add a required arg) and an old job is dequeued, ARQ raises a `TypeError` and marks the job failed.

**Worker crash mid-task.** This is the trickiest case. ARQ's semantics here are **at-most-once per attempt, at-least-once across retries**. A job in `in_progress` whose worker dies will eventually be reclaimed once `job_timeout = 300s` elapses — another worker (or the restarted one) picks it up and runs it again. If the original worker had already written facts to Qdrant before crashing, those writes persist; the retry will write them again. The idempotency check in `process_memory_raw` mitigates this for raw stores. For `process_memory_store`, the LLM extraction is non-deterministic and could produce slightly different facts on the retry — duplicates are caught by the exact-hash dedup in `_batch_store_facts` (`memory/write.py`) and by the dedup cron (`dedup_memories`, `memory/delete.py`).

**Redis disconnect.** Covered above: API falls back to synchronous storage and returns `200`. The worker, by contrast, exits cleanly when Redis is unreachable and is expected to be restarted by the orchestrator.

**Stale job cleanup.** Not implemented. Redis keys at `arq:job:*` and `arq:result:*` persist for ARQ's default ~500-day TTL. For long-lived production deployments, consider a 7-day cleanup cron.

## Observability

Logging is structured JSON via structlog (see `logging_config.py`). Notable worker log lines:

- `"ARQ worker starting up..."` — `worker.py`
- `"Dedup [{uid}]: removed {removed} duplicates"` per user — `worker.py`
- `"Skipping duplicate memory for user ..."` — `worker.py` (idempotency hit)
- `"Redis unavailable, falling back to sync store"` — API-side fallback warning

The `/health` endpoint in `main.py` returns `503` if the vector store is unreachable and `200` with `degraded` if Redis alone is down. There is **no Prometheus, StatsD, or OpenTelemetry SDK wired in**. Counters and percentiles must be derived from log scraping. Adding metrics is a known gap; see [11-deployment](./11-deployment.md) for runbook-level monitoring guidance.

## Integration Tests

`neuralscape-service/tests/test_async_pipeline.py` exercises the full path end-to-end and **requires running services**: Redis on 6379, Qdrant on 6333, Neo4j on 7687, the API on 8199, and an ARQ worker process. The helper `poll_until_done(task_id)` polls the status endpoint every 5 seconds, up to 300 seconds.

Test classes:

- `TestHealth` — sanity check.
- `TestAsyncRawMemoryStore` — POST raw, poll, assert completion + content.
- `TestAsyncConversationExtraction` — POST messages, poll, assert extracted facts.
- `TestTaskStatusPolling` — explicit state transitions: queued → processing → completed.
- `TestSyncSearch` — async writes do not break sync reads.
- `TestLegacyAsyncEndpoint` — backwards compat for the older `/memories/async` endpoint.

Run with `uv run pytest tests/test_async_pipeline.py -v -s`. See [10-testing](./10-testing.md) for the full test taxonomy and unit-vs-integration split.

## Related

- [04-memory-service-core](./04-memory-service-core.md)
- [05-llm-extraction](./05-llm-extraction.md)
- [02-service-architecture](./02-service-architecture.md)
- [11-deployment](./11-deployment.md)
- [09-plugin-system](./09-plugin-system.md)
- [06-storage-backends](./06-storage-backends.md)
