---
title: Memory Service Core
date: 2026-05-06
tags: [reference, neuralscape, memory-service, extraction, dedup]
source: handwritten
---

# Memory Service Core

> **Status (as of 2026-11-16, dev 6a81341f):** current reference. Code citations name modules and functions, not line numbers; the implementation lives in the `neuralscape-service/memory/` package behind the `memory_service.py` facade. Where a snippet is marked abridged it is a simplified excerpt, so read the named module for the real code.

## Overview

`MemoryService` is the single business-logic class that backs both the FastAPI surface in `main.py` and the MCP tools in `mcp_server.py`. Everything memory-related funnels through it: LLM-driven extraction, dual-write into Qdrant and Neo4j, hybrid recall, scope inference, deduplication, soft-delete edge expiration, and junk episode cleanup. The implementation is split across the `neuralscape-service/memory/` package (`write.py`, `search.py`, `reads.py`, `edit.py`, `delete.py`, `graph_admin.py`, `core.py` and smaller helpers), with `neuralscape-service/memory_service.py` as the facade that composes them. If you need to understand how a fact moves from a chat transcript into vector space and a temporal knowledge graph and back out again on a recall, start with `memory/write.py` and `memory/search.py`.

The class is constructed once at app startup (lazy state, thread-safe via `_init_lock`) and then shared across requests. It owns three handles to the underlying stack: a mem0 `Memory` instance (which wraps Qdrant + a SQLite history DB), a `Graphiti` instance for the Neo4j temporal graph, and a `_bridge` adapter that lets sync code hop onto Graphiti's asyncio loop without blocking the FastAPI worker. See `_get_memory` and the init helpers in `neuralscape-service/memory/core.py` (`CoreMixin`) for the lazy-init plumbing.

## Public API surface

`MemoryService` is defined in `neuralscape-service/memory_service.py`, which is a thin facade: the class is assembled from mixins that live in the `memory/` package, and `memory_service.py` re-exports the helpers tests and callers import. The table lists the core methods and the module that now implements each. It is not exhaustive: the class has grown well beyond the original twenty (batch writes, `get_memories` / `timeline`, standards, supersession, provenance, remediation, conversion), so list the mixin modules for the full set.

| Group | Method | Module (in `neuralscape-service/memory/`) | Purpose |
|---|---|---|---|
| Write | `extract_and_store` | `write.py` | LLM extract → batch embed → Qdrant + graph |
| Write | `store_raw` | `write.py` | Pre-categorized fact, no LLM |
| Read | `search` | `search.py` | Hybrid: vector + graph + merge (plus fusion/rerank legs) |
| Read | `search_graph` | `search.py` | Graphiti search only |
| Read | `get_project_context` | `reads.py` | Global + project bucketed by category |
| Read | `get_global_context` | `reads.py` | Global only, bucketed by category |
| CRUD | `get_memory` | `reads.py` | Single memory by ID |
| CRUD | `list_memories` | `reads.py` | Filter by scope/category/project |
| CRUD | `patch_memory` | `edit.py` | Presence-keyed partial update; permission-gated; returns a deferred graph job |
| CRUD | `retag_memories` | `edit.py` | Bulk metadata ops over a filter set; per-row permission/validity skip counters |
| CRUD | `delete_memory` | `delete.py` | Vector delete + soft-delete edges |
| CRUD | `delete_memories` | `delete.py` | Bulk delete with filters |
| Graph | `get_graph_nodes` | `graph_admin.py` | Entity nodes by group_id |
| Graph | `get_graph_edges` | `graph_admin.py` | Edges (facts) by group_id |
| Graph | `get_graph_episodes` | `graph_admin.py` | Episodic nodes by group_id |
| Graph | `get_graph_communities` | `graph_admin.py` | Community nodes |
| Graph | `delete_episode` | `graph_admin.py` | Cypher DETACH DELETE |
| Ops | `delete_junk_episodes` | `graph_admin.py` | Pattern-match cleanup |
| Ops | `dedup_memories` | `delete.py` | Two-phase: hash + cosine |
| Ops | `get_all_user_ids` | `reads.py` | Scroll for cron iteration |
| Lifecycle | `close` | `core.py` | Tear down Graphiti driver |

## Extraction pipeline

`extract_and_store` is the canonical write path for the conversational use case (Claude Code transcripts, Cursor sessions, etc.). It runs a four-stage pipeline: prompt build → Gemini call with retry → JSON parse + category extract → junk filter → batch store + separate graph ingest. The whole thing is synchronous from the worker's perspective; async-ness lives one layer up in ARQ.

### Stage 1 — Prompt build

`build_extraction_messages()` from `prompts.py` packs the inbound `[{role, content}, ...]` list into a single system+conversation message for Gemini. The system prompt is `CODING_ASSISTANT_EXTRACTION_PROMPT` at `prompts.py`, which enumerates the 13 memory categories and instructs the model to emit `[category] fact` lines. Junk-filter rules ("don't store raw tool output") are baked directly into the prompt — the post-extraction `_is_junk_fact` check is a defense-in-depth backstop, not the primary filter.

### Stage 2 — Gemini call with retry

```python
response = retry_transient(
    client.models.generate_content,
    model=settings.gemini_llm_model,
    contents=extraction_messages[0]["content"],
    config=GenerateContentConfig(
        http_options=HttpOptions(timeout=60_000),  # milliseconds
    ),
    operation="LLM extraction",
    fallback_model=settings.gemini_llm_fallback_model,
)
parsed_facts = parse_extraction_response(response.text)
```
(abridged from `extract_and_store` in `memory/write.py`)

The `genai.Client` is cached per process via `_get_genai_client()` in `memory/core.py`. Defaults are set in `config.py` (`gemini_llm_model`, currently `gemini-3.1-flash-lite`, with `gemini-2.5-flash` as the fallback). The 60-second HTTP timeout is set in milliseconds — easy to mis-read.

### Stage 3 — Parse + categorize

`parse_extraction_response()` (`prompts.py`) handles the JSON-wrapped response, peels markdown code fences, and falls back to line-by-line bracket parsing if JSON decoding fails. `parse_category_from_fact()` (`prompts.py`) then extracts the `[category]` prefix from each fact. If a category is invalid or missing, it defaults to `personal_fact` rather than failing the whole batch.

### Stage 4 — Junk filter

```python
parsed_facts = [
    (cat, content) for cat, content in parsed_facts
    if not _is_junk_fact(content)
]
```
(abridged from `extract_and_store` in `memory/write.py`)

`_is_junk_fact()` in `memory/junk.py` drops anything under 10 characters or matching `_JUNK_PATTERNS` (same module): `"Ran command:"`, `"Edited file:"`, `"Read file:"`, `"Tool result:"`, etc. These are the raw tool-call log lines that occasionally leak past the prompt instruction.

### Stage 5 — Batch store + graph ingest

The crucial optimization is `_batch_store_facts()` (`memory/write.py`): a single `embed_batch()` call followed by a single Qdrant `insert()` covers all facts at once. This deliberately bypasses mem0's per-fact `m.add()` pipeline, which would otherwise trigger a separate graph ingestion per fact — multiplying LLM token cost.

Graph ingestion happens once, separately, with the cleaned conversation:

```python
cleaned_messages = _clean_conversation_for_graph(messages)
raw_text = "\n".join(
    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
    for msg in cleaned_messages
)
try:
    if self._graphiti and self._bridge and raw_text.strip():
        retry_transient(
            self._memory.graph.add,
            data=raw_text,
            filters={"user_id": user_id, "group_id": group_id},
            operation="graph storage (extract_and_store)",
        )
```
(abridged from `extract_and_store` in `memory/write.py`)

`_clean_conversation_for_graph()` (`memory/junk.py`) strips junk lines per-message before joining, so the graph never ingests `Ran command: ls -la`-style noise. See [05-llm-extraction](./05-llm-extraction.md) for prompt details.

## Dual-write storage path

Every successful write fans out to two stores: Qdrant (vectors + payload) and Neo4j (temporal graph via Graphiti). The writes are non-transactional. Order matters: vector-first, graph-second.

### Qdrant payload

```python
payload = {
    "data": content,
    "hash": hashlib.md5(content.encode()).hexdigest(),
    "created_at": now_iso,
    "user_id": user_id,
    "agent_id": agent_id,
    "run_id": run_id,
    "metadata": {
        "scope": scope.value if isinstance(scope, MemoryScope) else scope,
        "category": category,
        "project_id": fact_project_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "source": source,
    },
}
```
(abridged from `_batch_store_facts` in `memory/write.py`; the real payload carries more fields)

The `hash` field is the dedup primary key for Phase 1 (exact dedup). `source` is `"explicit"` when written via `store_raw` or `"conversation"` when written via `extract_and_store`.

### The `metadata.` prefix gotcha

This is the most common foot-gun in the codebase and the subject of fix `c90f347`/`6c94f20`. Because `category`, `scope`, and `project_id` live nested under the `metadata` field in the Qdrant payload, **filter keys must be prefixed with `metadata.`** to match. The pattern is consistent across `search`, `list_memories`, `get_project_context`, etc.:

```python
filters = {}
if categories:
    filters["metadata.category"] = {"in": categories}
if scope:
    filters["metadata.scope"] = scope
```
(abridged from `search` in `memory/search.py`)

If you forget the prefix you get an empty result set with no error — Qdrant silently doesn't match anything. The same applies to `IsNullCondition` in `_list_null_category_memories` in `memory/reads.py`.

### Neo4j (Graphiti) write

Graph writes use mem0's wrapper, which delegates to Graphiti:

```python
retry_transient(
    self._memory.graph.add,
    data=content,
    filters={"user_id": user_id, "group_id": group_id},
    operation="store_raw graph add",
)
```
(abridged from `store_raw` in `memory/write.py`)

`group_id` comes from `_build_group_id(visibility, user_id, project_id, workspace)` in `memory/groups.py`. See the Scope section below for the canonical format.

### Failure modes

The two stores are written serially with no rollback. The cases to keep in mind:

- **Vector OK, graph fail**: warning logged in `extract_and_store` (or `store_raw`) in `memory/write.py`, response returns successfully. The fact lives in Qdrant only — searchable via vector but invisible to graph queries until something else re-ingests it. Treated as non-critical.
- **Vector fail**: error propagates; nothing is written. The caller (worker) marks the task failed.
- **Graphiti has no ingest dedup**: re-ingesting the same content creates fresh edges every time, which is why the post-hoc `dedup_memories` cron and `delete_junk_episodes` exist.

Idempotency on the vector side comes from Qdrant's upsert-by-ID semantics: writing the same UUID twice replaces the prior point. Graphiti edges, by contrast, accumulate.

## Hybrid search

`search` (`memory/search.py`) is a vector-first hybrid that augments results with graph edge facts. It has two distinct branches based on whether `project_id` is given without an explicit `scope`.

### Dual-scope vector search

When the caller passes `project_id` and no `scope`, the method issues two parallel-style searches and merges:

```python
if project_id and not scope:
    project_filters = {**filters, "metadata.project_id": project_id}
    project_results = m.search(
        query=query, user_id=user_id, limit=limit, filters=project_filters,
    )
    global_filters = {**filters, "metadata.scope": "global"}
    global_results = m.search(
        query=query, user_id=user_id, limit=limit, filters=global_filters,
    )
    all_results = self._merge_results(project_results, global_results)
    vector_responses = self._results_to_responses(all_results[:limit])
```
(abridged from `search` in `memory/search.py`)

`_merge_results` (`memory/search.py`) deduplicates by ID and sorts by score descending. This way the caller gets project context plus global preferences in one ranked list.

### Graph search via the bridge

`search_graph` (`memory/search.py`) wraps a Graphiti `g.search_()` call with `EDGE_HYBRID_SEARCH_RRF` (Reciprocal Rank Fusion across BM25, vector, and edge text). Group IDs are taken from `_get_group_ids(caller_user_id, project_id)`, which returns the caller's private groups plus the shared (and, when enabled, standard) pools. Because Graphiti is async-only, the call goes through `_run_on_bridge` (`memory/core.py`) which schedules the coroutine on the bridge's dedicated event loop with a 30-second timeout and cancels on timeout.

### Result merge

After both phases run, `_deduplicate_responses` (`memory/search.py`) lowercases and strips each vector response into a `seen_content` set, then drops any graph response whose content is an exact-or-substring match. The remaining unique results are fused into one merit-ranked list: both legs carry cosine scores in the same space, so graph rows rank among the vector hits on score, with no fixed quota or cap, and the list is truncated to `limit`. (The original design interleaved the two legs one to one; that was replaced because it let unscored graph rows evict ranked vector hits.) Graph search failures are caught and logged as non-critical (`memory/search.py`) so an unreachable Neo4j never breaks recall — vector-only results still flow back to the caller.

## Scope and group_id

This section is the operational summary; the canonical model lives in [03-memory-model](./03-memory-model.md).

Two scopes, `global` (user-wide) and `project` (project-scoped), are carried in the Qdrant payload (`metadata.scope`). The Graphiti `group_id` is derived from visibility, owner, project and workspace by `_build_group_id` in `memory/groups.py`:

| Case | group_id |
|---|---|
| Private, no project | `user--{user_id}` |
| Private, project | `user--{user_id}--project--{project_id}` |
| Shared, no project | `shared` |
| Shared, project | `shared--project--{project_id}` |

A non-default workspace appends `--ws--{workspace}`. `_get_group_ids` returns the set of group ids a read should cover. [03-memory-model](./03-memory-model.md) has the canonical description.

> **Historical note.** The first design used the bare tokens `global` and `project--{project_id}` for every user. The double hyphen was introduced by PR `2edd9b5`, when mem0 (`project:{id}`) and `MemoryService` (`project--{id}`) disagreed and project-scoped graph searches silently missed. The per-user and `shared` namespaces replaced the bare tokens later, to stop users sharing one graph namespace.

### Scope inference

When a fact comes through extraction without an explicit scope, `_batch_store_facts` infers it (`memory/write.py`):

1. `default_scope_for_category(category)` (`schemas.py`) provides the default — semantic categories (preference, personal_fact) → global; project categories (tech_stack, convention) → project.
2. If `project_id` is present and the category isn't in `GLOBAL_CATEGORIES`, force scope to `PROJECT`.
3. If category is in `PROJECT_CATEGORIES` but no `project_id` given, call `_infer_project_id(content)` (`memory/hashing.py`), which pattern-matches against known project slugs (deployment-specific, from `KNOWN_PROJECT_SLUGS`).
4. If inference fails, fall back to global with a warning rather than rejecting the fact.

## Two-phase deduplication

`dedup_memories` (`memory/delete.py`) is the cron-scheduled cleanup that catches anything the live write path didn't. Worker schedule: every 6 hours (`worker.py`).

### Phase 1 — Exact (MD5 hash)

`_scroll_all_user_memories` (`memory/delete.py`) paginates through Qdrant's `scroll()` API to gather every point for the user (mem0's wrapper doesn't support pagination, so this bypasses it). All points are then bucketed by `payload["hash"]`. Any group of size ≥2 keeps the newest by `created_at` and deletes the rest:

```python
for h, group in hash_groups.items():
    if len(group) < 2:
        continue
    # Sort by created_at descending — keep the first (newest)
    group.sort(
        key=lambda x: x["payload"].get("created_at", ""),
        reverse=True,
    )
    for dup in group[1:]:
        mid = dup["id"]
        if mid in deleted_ids:
            continue
        try:
            self._delete_qdrant_memory_with_graph_cleanup(mid, dup["payload"])
            deleted_ids.add(mid)
            exact_removed += 1
        except Exception as e:
            logger.warning(f"Failed to delete exact dup {mid}: {e}")
```
(abridged from `dedup_memories` in `memory/delete.py`)

Each delete uses `_delete_qdrant_memory_with_graph_cleanup` (`memory/delete.py`), which deletes the Qdrant point and best-effort soft-deletes any related graph edges via `_expire_graph_edges_for_memory`.

### Phase 2 — Semantic (cosine similarity)

For every memory not already deleted, embed the text, run a top-5 vector search filtered to the user, and for each hit above the cosine threshold delete the older of the two:

```python
for hit in hits:
    hit_id = str(hit["id"]) if isinstance(hit, dict) else str(hit.id)
    hit_score = hit["score"] if isinstance(hit, dict) else hit.score
    hit_payload = hit.get("payload", {}) if isinstance(hit, dict) else (hit.payload or {})

    if hit_id == mid or hit_id in deleted_ids:
        continue
    if hit_score < threshold:
        continue

    mem_created = mem["payload"].get("created_at", "")
    hit_created = hit_payload.get("created_at", "")
    older_id, older_payload = (
        (hit_id, hit_payload) if hit_created <= mem_created else (mid, mem["payload"])
    )
```
(abridged from `dedup_memories` in `memory/delete.py`)

Threshold defaults to `0.95` (`config.py`, key `dedup_similarity_threshold`). Per-memory failures don't abort the run — they're logged and skipped (`dedup_memories`, `memory/delete.py`).

Return shape: `{user_id, exact_duplicates_removed, semantic_duplicates_removed, total_checked}`.

## Graph re-ingestion

The graph re-ingest on an edit (in `patch_memory`, `memory/edit.py`; formerly `update_memory`) is the only path that triggers a deliberate re-ingestion into Graphiti. The motivation is contradiction handling: when a user updates a stored fact (e.g., "prefers dark mode" → "prefers light mode"), the new content needs to flow through Graphiti's edge-extraction pipeline so it can detect the contradiction and expire the stale edge.

```python
# HISTORICAL: the pre-split update_memory snippet, kept to show the idea.
# Its _build_group_id(scope, project_id) call no longer exists; the current
# signature is _build_group_id(visibility, user_id, project_id, workspace).
if content and existing and self._graphiti and self._bridge:
    try:
        metadata = existing.get("metadata", {}) or {}
        scope = metadata.get("scope", "global")
        project_id = metadata.get("project_id")
        user_id = existing.get("user_id", "")
        group_id = _build_group_id(scope, project_id)
        retry_transient(
            self._memory.graph.add,
            data=content,
            filters={"user_id": user_id, "group_id": group_id},
            operation=f"graph re-ingestion for {memory_id}",
        )
    except Exception as e:
        logger.warning(f"Graph re-ingestion failed for {memory_id} (non-critical): {e}")
```
(historical, from the pre-split `update_memory`; the current logic lives in `patch_memory` in `memory/edit.py` and builds the group id with the current `_build_group_id`)

The vector update is the source of truth and propagates errors. The graph re-ingest is best-effort: if it fails, the vector reflects the new content and the graph keeps the old edges until the next manual cleanup. Note that the existing memory record is fetched **before** the update (in `memory/edit.py`) so its `user_id` and metadata can be reused for the graph filters.

## Failure modes and retries

`retry_transient` (`memory/retry.py`) is the single source of truth for transient-error handling and is used everywhere LLM, embedding, or graph-write calls are made.

```python
last_exc = None
for attempt in range(max_retries + 1):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        last_exc = e
        if not _is_transient(e):
            raise
        if attempt == max_retries:
            break  # exhausted retries — try fallback below
        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
        logger.warning(...)
        time.sleep(delay)
```
(abridged from `retry_transient` in `memory/retry.py`)

Transient detection (`_is_transient`, `memory/retry.py`) does substring matching against `_TRANSIENT_PATTERNS` (same module): `"503"`, `"429"`, `"UNAVAILABLE"`, `"RESOURCE_EXHAUSTED"`, `"rate limit"`, `"overloaded"`, `"capacity"`, `"timed out"`, `"timeout"`. Anything else is non-retryable and propagates immediately.

Backoff: `delay = min(base * 2^attempt + jitter, max)`. Defaults from `config.py`: `base=1s`, `max=30s`, `max_retries=3`. A jitter of 0–1s prevents thundering-herd when the upstream comes back.

If primary retries exhaust and `fallback_model` is set, the function is called once more with `kwargs[model_kwarg]` swapped to the fallback. This is what handles a primary-model outage, for example the primary model → `gemini-2.5-flash` in `extract_and_store`.

### Partial-failure semantics

Each call site has explicit semantics for what counts as critical:

- **Extraction**: LLM error → return `[]` (`extract_and_store`, `memory/write.py`). The whole batch is dropped, the worker task succeeds with zero stored memories.
- **Graph ingest in extract path**: warning only (`extract_and_store`, `memory/write.py`). Vectors persist; graph misses this batch.
- **Update**: vector update propagates errors; graph re-ingest is non-critical (`memory/edit.py`).
- **Delete**: vector is primary; `_expire_graph_edges_for_memory` is best-effort (`memory/delete.py`).
- **Dedup**: per-item failures logged but the loop continues (`dedup_memories`, `memory/delete.py`); semantic search failures skip just that memory (`memory/delete.py`).

## Junk episode cleanup

`delete_junk_episodes` (`memory/graph_admin.py`) is the graph-side complement to `_is_junk_fact`. It scans Graphiti episodic nodes for content that looks like raw tool/conversation logs and deletes them via Cypher.

The detection pattern in `_find_junk_episodes` (`memory/graph_admin.py`) flags two cases: content starting with `"assistant:"` (raw conversation logs that leaked past the prompt) and anything matching `_JUNK_RE` from the same regex used by the extraction filter.

When `project_id` is `None`, the cleanup iterates the global group plus all known projects:

```python
if project_id is not None:
    project_ids_to_scan = [project_id]
else:
    # None = global group, then all known projects
    project_ids_to_scan = [None] + settings.known_projects
```
(abridged from `delete_junk_episodes` in `memory/graph_admin.py`)

The known-project list comes from the `KNOWN_PROJECT_SLUGS` env setting (`settings.known_projects`); the same list backs `_infer_project_id`'s slug matcher.

Two modes: `dry_run=True` returns a per-group breakdown of junk counts plus 5 sample contents per group without deleting; the production mode hard-deletes each junk episode via `delete_episode` (`memory/graph_admin.py`), which runs `MATCH (e:Episodic {uuid: $uuid}) DETACH DELETE e` against the Neo4j driver. Returns include both per-group breakdown and total count.

This is one of the few hard-delete paths in the codebase (most graph cleanup is soft-delete via `expired_at`). Episodic nodes are append-only event logs by design, so once they're identified as junk there's no audit value in keeping them.

## Related

- [03-memory-model](./03-memory-model.md) — canonical scope, category, and group_id semantics
- [05-llm-extraction](./05-llm-extraction.md) — extraction prompt, category parser, junk patterns
- [06-storage-backends](./06-storage-backends.md) — Qdrant payload schema and Neo4j/Graphiti shape
- [07-async-pipeline](./07-async-pipeline.md) — how `extract_and_store` is invoked from ARQ workers
- [02-service-architecture](./02-service-architecture.md) — the full FastAPI + MCP + worker topology
