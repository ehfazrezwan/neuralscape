---
title: Async Write Pipeline
date: 2026-05-06
tags: [reference, neuralscape, async, arq, redis, worker]
source: handwritten
---

# Async Write Pipeline

Neuralscape splits memory operations along a sharp boundary: **writes are asynchronous**, **reads are synchronous**. A POST to `/v1/memories` returns `202 Accepted` with a `task_id`; the actual LLM extraction, embedding, vector upsert, and graph ingestion happen later in a separate ARQ worker process. A POST to `/v1/search` runs synchronously against Qdrant and Neo4j and returns `200 OK`. This page documents the full write-side machinery: the queue lifecycle, the worker, deterministic deduplication of enqueues, the status-polling protocol, retry behaviour, and failure modes.

## Overview

Why split writes from reads at all? Memory ingestion is expensive. A single conversation may invoke Gemini twice (extraction + dedup judging), batch-embed N facts, write each to Qdrant, then re-ingest the whole conversation as episodes into the Graphiti temporal graph. End-to-end latency can run from a few hundred milliseconds to tens of seconds. Forcing a client to wait synchronously for that would block conversational UIs and time out under load.

Reads, by contrast, are a single Qdrant ANN lookup with optional Neo4j re-ranking — fast enough to serve in-line. So Neuralscape funnels writes through Redis + ARQ and keeps reads on the request thread.

The full picture:

- **Client** issues `POST /v1/memories` (extraction) or `POST /v1/memories/raw` (pre-categorized fact).
- **API** (`main.py`) hashes the payload, enqueues an ARQ job, and returns `202` with `{task_id, poll_url}`.
- **Worker** (`arq worker.WorkerSettings`) — a separate process — dequeues, runs the actual storage logic in [04-memory-service-core](./04-memory-service-core.md), emits `memory_stored` events to extensions, and writes the result to Redis.
- **Client** polls `GET /v1/memories/status/{task_id}` until `status == "completed"` (or `"failed"`).

Two cron jobs piggyback on the same worker: a 6-hour deduplication sweep and an evening conversation-compile check.

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

The full worker configuration lives in `neuralscape-service/worker.py:272-306`:

```python
class WorkerSettings:
    functions = [
        process_memory_store,
        process_memory_raw,
        process_conversation_flush,
        process_conversation_compile,
    ]
    cron_jobs = [
        cron(dedup_all_memories, hour=settings.dedup_cron_hours,
             minute=0, timeout=1800, unique=True, max_tries=1),
        cron(auto_compile_check, hour={18,19,20,21,22,23},
             minute=30, timeout=600, unique=True, max_tries=1),
    ]
    redis_settings = parse_redis_settings()
    queue_name = settings.arq_queue_name        # "neuralscape:queue"
    max_jobs = 10
    job_timeout = settings.arq_job_timeout      # 300s default
    max_tries = settings.arq_max_retries        # 3 default
```

Key knobs:

- `max_jobs = 10` — ten task coroutines run concurrently per worker process. Scale by adding worker replicas, not by raising this.
- `job_timeout = 300s` — a single task that exceeds 5 minutes is killed and counted as a failure (subject to `max_tries`).
- `max_tries = 3` — if the task raises any uncaught exception, ARQ re-enqueues it up to two more times with default backoff. After the third failure the job is marked permanently failed.
- `queue_name = "neuralscape:queue"` — the API and worker must agree on this. Mismatched queue names silently mean enqueued jobs never run.

Redis connection settings are parsed in `neuralscape-service/config.py:138-158`: 5 connection retries, 2-second delay between retries, 10-second socket timeout. Default URL `redis://localhost:6379`. The same `parse_redis_settings()` helper is used by both the API's `TaskManager` and `WorkerSettings`, ensuring consistency.

The cron schedule for `dedup_all_memories` resolves to hours `{0, 6, 12, 18}` UTC (every 6 hours), with `unique=True` so concurrent worker replicas can't double-run it. `auto_compile_check` fires at `:30` between 6 PM and midnight UTC.

## Task Functions

### process_memory_store

`neuralscape-service/worker.py:30-66`. Signature:

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

`neuralscape-service/worker.py:69-126`. Signature mirrors the request body for `POST /v1/memories/raw`. The notable wrinkle is the **idempotency check at lines 84-96**:

```python
existing = service.search(query=content, user_id=user_id, project_id=project_id, limit=3)
for mem in existing:
    if mem.memory.strip().lower() == content.strip().lower():
        logger.info(f"Skipping duplicate memory for user {user_id}: ...")
        return {"memories": [mem.model_dump(exclude_none=True)], "deduplicated": True}
```

Before storing, the worker performs a vector search and case-insensitively compares the top hits against the new content. If an exact (whitespace- and case-normalised) match exists, it short-circuits and returns the existing memory with `deduplicated: True`. This is in addition to the deterministic-job-ID dedup at the enqueue layer — they catch different cases. Job-ID dedup catches a client retrying the *same* request; idempotency catches a client posting the same fact twice through different code paths or sessions.

Failures of the idempotency search are logged and swallowed: better to risk a duplicate than to drop a write because Qdrant blipped.

## TaskManager: Deterministic Job IDs

`neuralscape-service/task_manager.py` is the API-side wrapper around the ARQ Redis pool. Its single most important behaviour is **deterministic job IDs**, generated at line 167-170:

```python
def _generate_job_id(content: str, user_id: str) -> str:
    h = hashlib.sha256(f"{user_id}:{content}".encode()).hexdigest()[:16]
    return f"ns-{h}"
```

`enqueue_store` (line 40-72) joins all message contents with `|` and prefixes `store:` before hashing. `enqueue_raw` (line 74-106) prefixes `raw:`. The 16-hex-char prefix gives 64 bits of collision resistance — more than enough for a per-user namespace.

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

The mapping lives in `task_manager.py:17-23` plus the success-vs-failure branch at `task_manager.py:122-129`. Note ARQ does not have a separate `failed` enum value: a failed job is `complete` with `info.success == False`, and `TaskManager.get_status` rewrites it to `"failed"`.

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

There is one important footgun in `main.py:612-696`. If `task_manager.enqueue_store` raises `ConnectionError` or `OSError` — i.e. Redis is unreachable — the API does not fail. Instead it falls back to running the storage **synchronously on the request thread** and returns `200` with the `StoreMemoryResponse` body directly:

```python
except (ConnectionError, OSError) as e:
    logger.warning(f"Redis unavailable, falling back to sync store: {e}")
    memories = await asyncio.to_thread(_service.extract_and_store, ...)
    return JSONResponse(status_code=200,
                       content=StoreMemoryResponse(memories=memories)...)
```

This is a survival mechanism: the service stays usable when Redis is down, at the cost of multi-second response times. But it is also a **silent contract change**. A client that always expects `202 + task_id` will be confused by a `200` containing the actual memories. Clients should branch on the status code, not assume async semantics.

## Periodic Cron Jobs

Two crons share the worker:

**`dedup_all_memories`** (`worker.py:213-240`) runs at `00:00`, `06:00`, `12:00`, `18:00` UTC with a 30-minute timeout. It iterates `service.get_all_user_ids()` (paginated by `dedup_batch_size`) and runs `service.dedup_memories(uid)` for each. The two-phase algorithm — exact MD5 hash followed by semantic cosine ≥ 0.95 — is documented in [04-memory-service-core](./04-memory-service-core.md). Per-user errors are caught at line 227-229 so one user's failure does not abort the sweep; the failed user gets `{"user_id": uid, "error": str(e)}` in the per-user list. The summary aggregates `total_exact_removed` and `total_semantic_removed` for the run.

**`auto_compile_check`** runs at `:30` past every hour from 18:00 to 23:00 UTC. It scans for un-compiled conversation dates in the vault and fires `compile_all_pending` to consolidate them. This is an extension-driven feature; see the conversation-compiler extension docs for details.

Both crons are marked `unique=True` so multiple worker replicas cannot run them concurrently — ARQ uses a Redis SET-NX lock per cron name.

## Failure Modes

**Transient errors from LLM/Neo4j/Qdrant.** Detected by `_is_transient()` at `memory_service.py:25-28`, which matches `503`, `429`, `UNAVAILABLE`, `RESOURCE_EXHAUSTED`, the substring `"rate limit"`, and `"timeout"`. The wrapper `retry_transient()` at `memory_service.py:97-150` retries with `min(base * 2^attempt + jitter, max)`, defaulting to base 1s, max 30s, and 3 retries (config in `config.py:38-40`). After exhausting the primary model, it falls back from `gemini-3-flash-preview` to `gemini-2.5-flash`. This handles **most** transient blips inside a single task invocation.

**ARQ-level retries.** If a task still raises after `retry_transient` is done, ARQ catches the exception, increments the try counter, and re-enqueues. With `max_tries = 3`, a task can run up to three times total before being marked failed. ARQ uses its default backoff schedule between retries.

**Non-transient errors.** A `ValueError`, `KeyError`, or category-validation failure is raised straight up. ARQ still applies `max_tries = 3`, but each retry will hit the same deterministic failure, so the job effectively fails immediately from the user's perspective.

**Poison messages.** If the function signature changes (e.g. add a required arg) and an old job is dequeued, ARQ raises a `TypeError` and marks the job failed.

**Worker crash mid-task.** This is the trickiest case. ARQ's semantics here are **at-most-once per attempt, at-least-once across retries**. A job in `in_progress` whose worker dies will eventually be reclaimed once `job_timeout = 300s` elapses — another worker (or the restarted one) picks it up and runs it again. If the original worker had already written facts to Qdrant before crashing, those writes persist; the retry will write them again. The idempotency check in `process_memory_raw` mitigates this for raw stores. For `process_memory_store`, the LLM extraction is non-deterministic and could produce slightly different facts on the retry — duplicates land at line 84-96 of memory_service.py and are caught there.

**Redis disconnect.** Covered above: API falls back to synchronous storage and returns `200`. The worker, by contrast, exits cleanly when Redis is unreachable and is expected to be restarted by the orchestrator.

**Stale job cleanup.** Not implemented. Redis keys at `arq:job:*` and `arq:result:*` persist for ARQ's default ~500-day TTL. For long-lived production deployments, consider a 7-day cleanup cron.

## Observability

Logging is structured JSON via structlog (see `logging_config.py`). Notable worker log lines:

- `"ARQ worker starting up..."` — `worker.py:245`
- `"Dedup [{uid}]: removed {removed} duplicates"` per user — `worker.py:226`
- `"Skipping duplicate memory for user ..."` — `worker.py:93` (idempotency hit)
- `"Redis unavailable, falling back to sync store"` — API-side fallback warning

The `/health` endpoint at `main.py:227-275` returns `503` if the vector store is unreachable and `200` with `degraded` if Redis alone is down. There is **no Prometheus, StatsD, or OpenTelemetry SDK wired in**. Counters and percentiles must be derived from log scraping. Adding metrics is a known gap; see [11-deployment](./11-deployment.md) for runbook-level monitoring guidance.

## Integration Tests

`neuralscape-service/tests/test_async_pipeline.py` exercises the full path end-to-end and **requires running services**: Redis on 6379, Qdrant on 6333, Neo4j on 7687, the API on 8199, and an ARQ worker process. The helper `poll_until_done(task_id)` at lines 46-57 polls the status endpoint every 5 seconds, up to 300 seconds.

Test classes:

- `TestHealth` (65-69) — sanity check.
- `TestAsyncRawMemoryStore` (77-142) — POST raw, poll, assert completion + content.
- `TestAsyncConversationExtraction` (150-187) — POST messages, poll, assert extracted facts.
- `TestTaskStatusPolling` (195-227) — explicit state transitions: queued → processing → completed.
- `TestSyncSearch` (234-248) — async writes do not break sync reads.
- `TestLegacyAsyncEndpoint` (256-284) — backwards compat for the older `/memories/async` endpoint.

Run with `uv run pytest tests/test_async_pipeline.py -v -s`. See [10-testing](./10-testing.md) for the full test taxonomy and unit-vs-integration split.

## Related

- [04-memory-service-core](./04-memory-service-core.md)
- [05-llm-extraction](./05-llm-extraction.md)
- [02-service-architecture](./02-service-architecture.md)
- [11-deployment](./11-deployment.md)
- [09-plugin-system](./09-plugin-system.md)
- [06-storage-backends](./06-storage-backends.md)
