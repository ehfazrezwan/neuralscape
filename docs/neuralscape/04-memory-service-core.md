---
title: Memory Service Core
date: 2026-05-06
tags: [reference, neuralscape, memory-service, extraction, dedup]
source: handwritten
---

# Memory Service Core

## Overview

`MemoryService` is the single business-logic class that backs both the FastAPI surface in `main.py` and the MCP tools in `mcp_server.py`. Everything memory-related funnels through it: LLM-driven extraction, dual-write into Qdrant and Neo4j, hybrid recall, scope inference, deduplication, soft-delete edge expiration, and junk episode cleanup. At ~1631 lines and 20 public methods, `neuralscape-service/memory_service.py` is the largest file in the codebase and the densest concentration of behavior. If you need to understand how a fact moves from a chat transcript into vector space and a temporal knowledge graph and back out again on a recall, this is the file to read.

The class is constructed once at app startup (lazy state, thread-safe via `_init_lock`) and then shared across requests. It owns three handles to the underlying stack: a mem0 `Memory` instance (which wraps Qdrant + a SQLite history DB), a `Graphiti` instance for the Neo4j temporal graph, and a `_bridge` adapter that lets sync code hop onto Graphiti's asyncio loop without blocking the FastAPI worker. See `neuralscape-service/memory_service.py:196-226` for the lazy-init plumbing.

## Public API surface

The 20 public methods cluster into six functional groups. Most callers (REST handlers, ARQ worker tasks, MCP tools) only touch the first two groups.

| Group | Method | File:line | Purpose |
|---|---|---|---|
| Write | `extract_and_store` | `memory_service.py:272-366` | LLM extract → batch embed → Qdrant + graph |
| Write | `store_raw` | `memory_service.py:368-460` | Pre-categorized fact, no LLM |
| Read | `search` | `memory_service.py:587-678` | Hybrid: vector + graph + merge |
| Read | `search_graph` | `memory_service.py:680-749` | Graphiti EDGE_HYBRID_SEARCH_RRF only |
| Read | `get_project_context` | `memory_service.py:755-801` | Global + project bucketed by category |
| Read | `get_global_context` | `memory_service.py:803-833` | Global only, bucketed by category |
| CRUD | `get_memory` | `memory_service.py:839-845` | Single memory by ID |
| CRUD | `list_memories` | `memory_service.py:847-873` | Filter by scope/category/project |
| CRUD | `patch_memory` | `memory_service.py` | Presence-keyed partial update; permission-gated; returns a deferred graph job |
| CRUD | `retag_memories` | `memory_service.py` | Bulk metadata ops over a filter set; per-row permission/validity skip counters |
| CRUD | `delete_memory` | `memory_service.py:914-929` | Vector delete + soft-delete edges |
| CRUD | `delete_memories` | `memory_service.py:931-992` | Bulk delete with filters |
| Graph | `get_graph_nodes` | `memory_service.py:1055-1084` | Entity nodes by group_id |
| Graph | `get_graph_edges` | `memory_service.py:1086-1120` | Edges (facts) by group_id |
| Graph | `get_graph_episodes` | `memory_service.py:1122-1155` | Episodic nodes by group_id |
| Graph | `get_graph_communities` | `memory_service.py:1157-1185` | Community nodes |
| Graph | `delete_episode` | `memory_service.py:1187-1203` | Cypher DETACH DELETE |
| Ops | `delete_junk_episodes` | `memory_service.py:1216-1290` | Pattern-match cleanup |
| Ops | `dedup_memories` | `memory_service.py:1509-1614` | Two-phase: hash + cosine |
| Ops | `get_all_user_ids` | `memory_service.py:1484-1507` | Scroll for cron iteration |
| Lifecycle | `close` | `memory_service.py:263-266` | Tear down Graphiti driver |

The full `MemoryService` class spans `memory_service.py:190-1631`.

## Extraction pipeline

`extract_and_store` is the canonical write path for the conversational use case (Claude Code transcripts, Cursor sessions, etc.). It runs a four-stage pipeline: prompt build → Gemini call with retry → JSON parse + category extract → junk filter → batch store + separate graph ingest. The whole thing is synchronous from the worker's perspective; async-ness lives one layer up in ARQ.

### Stage 1 — Prompt build

`build_extraction_messages()` from `prompts.py:123-144` packs the inbound `[{role, content}, ...]` list into a single system+conversation message for Gemini. The system prompt is `CODING_ASSISTANT_EXTRACTION_PROMPT` at `prompts.py:25-66`, which enumerates the 13 memory categories and instructs the model to emit `[category] fact` lines. Junk-filter rules ("don't store raw tool output") are baked directly into the prompt — the post-extraction `_is_junk_fact` check is a defense-in-depth backstop, not the primary filter.

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
(`memory_service.py:301-311`)

The `genai.Client` is cached per process via `_get_genai_client()` at `memory_service.py:252-261`. Default model is `gemini-3-flash-preview` with `gemini-2.5-flash` as the fallback (see `config.py`). The 60-second HTTP timeout is set in milliseconds — easy to mis-read.

### Stage 3 — Parse + categorize

`parse_extraction_response()` (`prompts.py:92-120`) handles the JSON-wrapped response, peels markdown code fences, and falls back to line-by-line bracket parsing if JSON decoding fails. `parse_category_from_fact()` (`prompts.py:69-89`) then extracts the `[category]` prefix from each fact. If a category is invalid or missing, it defaults to `personal_fact` rather than failing the whole batch.

### Stage 4 — Junk filter

```python
parsed_facts = [
    (cat, content) for cat, content in parsed_facts
    if not _is_junk_fact(content)
]
```
(`memory_service.py:318-321`)

`_is_junk_fact()` at `memory_service.py:49-54` drops anything under 10 characters or matching `_JUNK_PATTERNS` (`memory_service.py:32-43`): `"Ran command:"`, `"Edited file:"`, `"Read file:"`, `"Tool result:"`, etc. These are the raw tool-call log lines that occasionally leak past the prompt instruction.

### Stage 5 — Batch store + graph ingest

The crucial optimization is `_batch_store_facts()` (`memory_service.py:462-581`): a single `embed_batch()` call followed by a single Qdrant `insert()` covers all facts at once. This deliberately bypasses mem0's per-fact `m.add()` pipeline, which would otherwise trigger a separate graph ingestion per fact — multiplying LLM token cost.

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
(`memory_service.py:350-362`)

`_clean_conversation_for_graph()` (`memory_service.py:57-76`) strips junk lines per-message before joining, so the graph never ingests `Ran command: ls -la`-style noise. See [05-llm-extraction](./05-llm-extraction.md) for prompt details.

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
(`memory_service.py:523-538`)

The `hash` field is the dedup primary key for Phase 1 (exact dedup). `source` is `"explicit"` when written via `store_raw` (`memory_service.py:420`) or `"conversation"` when written via `extract_and_store` (`memory_service.py:339`).

### The `metadata.` prefix gotcha

This is the most common foot-gun in the codebase and the subject of fix `c90f347`/`6c94f20`. Because `category`, `scope`, and `project_id` live nested under the `metadata` field in the Qdrant payload, **filter keys must be prefixed with `metadata.`** to match. The pattern is consistent across `search`, `list_memories`, `get_project_context`, etc.:

```python
filters = {}
if categories:
    filters["metadata.category"] = {"in": categories}
if scope:
    filters["metadata.scope"] = scope
```
(`memory_service.py:615-619`)

If you forget the prefix you get an empty result set with no error — Qdrant silently doesn't match anything. The same applies to `IsNullCondition` in `_list_null_category_memories` at `memory_service.py:1015`.

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
(`memory_service.py:441-446`)

`group_id` comes from `_build_group_id(scope, project_id)` at `memory_service.py:175-179`. See the Scope section below for the canonical format.

### Failure modes

The two stores are written serially with no rollback. The cases to keep in mind:

- **Vector OK, graph fail**: warning logged at `memory_service.py:363-364` (or `:447-448` for `store_raw`), response returns successfully. The fact lives in Qdrant only — searchable via vector but invisible to graph queries until something else re-ingests it. Treated as non-critical.
- **Vector fail**: error propagates; nothing is written. The caller (worker) marks the task failed.
- **Graphiti has no ingest dedup**: re-ingesting the same content creates fresh edges every time, which is why the post-hoc `dedup_memories` cron and `delete_junk_episodes` exist.

Idempotency on the vector side comes from Qdrant's upsert-by-ID semantics: writing the same UUID twice replaces the prior point. Graphiti edges, by contrast, accumulate.

## Hybrid search

`search` (`memory_service.py:587-678`) is a vector-first hybrid that augments results with graph edge facts. It has two distinct branches based on whether `project_id` is given without an explicit `scope`.

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
(`memory_service.py:622-643`)

`_merge_results` (`memory_service.py:1616-1631`) deduplicates by ID and sorts by score descending. This way the caller gets project context plus global preferences in one ranked list.

### Graph search via the bridge

`search_graph` (`memory_service.py:680-749`) wraps a Graphiti `g.search_()` call with `EDGE_HYBRID_SEARCH_RRF` (Reciprocal Rank Fusion across BM25, vector, and edge text). Group IDs are taken from `_get_group_ids(project_id)` so global edges are always included. Because Graphiti is async-only, the call goes through `_run_on_bridge` (`memory_service.py:228-250`) which schedules the coroutine on the bridge's dedicated event loop with a 30-second timeout and cancels on timeout.

### Result merge

After both phases run, `_deduplicate_responses` (`memory_service.py:1340-1380`) lowercases and strips each vector response into a `seen_content` set, then drops any graph response whose content is an exact-or-substring match. The remaining unique results are interleaved (vector-1, graph-1, vector-2, graph-2, …) and truncated to `limit`. Graph search failures are caught and logged as non-critical (`memory_service.py:673-674`) so an unreachable Neo4j never breaks recall — vector-only results still flow back to the caller.

## Scope and group_id

This section is the operational summary; the canonical model lives in [03-memory-model](./03-memory-model.md).

Two scopes — `global` (user-wide) and `project` (project-scoped) — map to two `group_id` formats in Graphiti:

| Scope | group_id |
|---|---|
| Global | `"global"` |
| Project | `"project--{project_id}"` (double-hyphen) |

```python
def _build_group_id(scope: str, project_id: str | None = None) -> str:
    if scope == MemoryScope.PROJECT and project_id:
        return f"project--{project_id}"
    return "global"

def _get_group_ids(project_id: str | None = None) -> list[str]:
    group_ids = ["global"]
    if project_id:
        group_ids.append(f"project--{project_id}")
    return group_ids
```
(`memory_service.py:175-187`)

The double-hyphen format is non-negotiable: prior to PR `2edd9b5`, mem0 used `project:{id}` (single colon) while `MemoryService` used `project--{id}`, causing all project-scoped graph searches to silently miss. The fix aligned both sides.

### Scope inference

When a fact comes through extraction without an explicit scope, `_batch_store_facts` infers it (`memory_service.py:500-520`):

1. `default_scope_for_category(category)` (`schemas.py:73-79`) provides the default — semantic categories (preference, personal_fact) → global; project categories (tech_stack, convention) → project.
2. If `project_id` is present and the category isn't in `GLOBAL_CATEGORIES`, force scope to `PROJECT`.
3. If category is in `PROJECT_CATEGORIES` but no `project_id` given, call `_infer_project_id(content)` (`memory_service.py:88-94`), which pattern-matches against known project slugs (deployment-specific, from `KNOWN_PROJECT_SLUGS`).
4. If inference fails, fall back to global with a warning rather than rejecting the fact.

## Two-phase deduplication

`dedup_memories` (`memory_service.py:1509-1614`) is the cron-scheduled cleanup that catches anything the live write path didn't. Worker schedule: every 6 hours (`worker.py:213-240`).

### Phase 1 — Exact (MD5 hash)

`_scroll_all_user_memories` (`memory_service.py:1432-1465`) paginates through Qdrant's `scroll()` API to gather every point for the user (mem0's wrapper doesn't support pagination, so this bypasses it). All points are then bucketed by `payload["hash"]`. Any group of size ≥2 keeps the newest by `created_at` and deletes the rest:

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
(`memory_service.py:1535-1552`)

Each delete uses `_delete_qdrant_memory_with_graph_cleanup` (`memory_service.py:1467-1482`), which deletes the Qdrant point and best-effort soft-deletes any related graph edges via `_expire_graph_edges_for_memory`.

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
(`memory_service.py:1579-1594`)

Threshold defaults to `0.95` (`config.py:43`, key `dedup_similarity_threshold`). Per-memory failures don't abort the run — they're logged and skipped (`memory_service.py:1575-1577`, `:1602-1603`).

Return shape: `{user_id, exact_duplicates_removed, semantic_duplicates_removed, total_checked}`.

## Graph re-ingestion

`update_memory` (`memory_service.py:875-912`) is the only path that triggers a deliberate re-ingestion into Graphiti. The motivation is contradiction handling: when a user updates a stored fact (e.g., "prefers dark mode" → "prefers light mode"), the new content needs to flow through Graphiti's edge-extraction pipeline so it can detect the contradiction and expire the stale edge.

```python
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
(`memory_service.py:896-910`)

The vector update is the source of truth and propagates errors. The graph re-ingest is best-effort: if it fails, the vector reflects the new content and the graph keeps the old edges until the next manual cleanup. Note that the existing memory record is fetched **before** the update (`memory_service.py:889`) so its `user_id` and metadata can be reused for the graph filters.

## Failure modes and retries

`retry_transient` (`memory_service.py:97-157`) is the single source of truth for transient-error handling and is used everywhere LLM, embedding, or graph-write calls are made.

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
(`memory_service.py:123-138`)

Transient detection (`_is_transient`, `memory_service.py:25-28`) does substring matching against `_TRANSIENT_PATTERNS` at `memory_service.py:22`: `"503"`, `"429"`, `"UNAVAILABLE"`, `"RESOURCE_EXHAUSTED"`, `"rate limit"`, `"overloaded"`, `"capacity"`, `"timed out"`, `"timeout"`. Anything else is non-retryable and propagates immediately.

Backoff: `delay = min(base * 2^attempt + jitter, max)`. Defaults from `config.py`: `base=1s`, `max=30s`, `max_retries=3`. A jitter of 0–1s prevents thundering-herd when the upstream comes back.

If primary retries exhaust and `fallback_model` is set, the function is called once more with `kwargs[model_kwarg]` swapped to the fallback. This is what handles `gemini-3-flash-preview` → `gemini-2.5-flash` at `memory_service.py:309`.

### Partial-failure semantics

Each call site has explicit semantics for what counts as critical:

- **Extraction**: LLM error → return `[]` (`memory_service.py:312-314`). The whole batch is dropped, the worker task succeeds with zero stored memories.
- **Graph ingest in extract path**: warning only (`memory_service.py:363-364`). Vectors persist; graph misses this batch.
- **Update**: vector update propagates errors; graph re-ingest is non-critical (`memory_service.py:909-910`).
- **Delete**: vector is primary; `_expire_graph_edges_for_memory` is best-effort (`memory_service.py:924-927`).
- **Dedup**: per-item failures logged but the loop continues (`memory_service.py:1551-1552`, `:1602-1603`); semantic search failures skip just that memory (`memory_service.py:1575-1577`).

## Junk episode cleanup

`delete_junk_episodes` (`memory_service.py:1216-1290`) is the graph-side complement to `_is_junk_fact`. It scans Graphiti episodic nodes for content that looks like raw tool/conversation logs and deletes them via Cypher.

The detection pattern in `_find_junk_episodes` (`memory_service.py:1205-1214`) flags two cases: content starting with `"assistant:"` (raw conversation logs that leaked past the prompt) and anything matching `_JUNK_RE` from the same regex used by the extraction filter.

When `project_id` is `None`, the cleanup iterates the global group plus all known projects:

```python
if project_id is not None:
    project_ids_to_scan = [project_id]
else:
    # None = global group, then all known projects
    project_ids_to_scan = [None] + ALL_KNOWN_PROJECTS
```
(`memory_service.py:1239-1243`)

The known-project list comes from the `KNOWN_PROJECT_SLUGS` env setting (`settings.known_projects`); the same list backs `_infer_project_id`'s slug matcher.

Two modes: `dry_run=True` returns a per-group breakdown of junk counts plus 5 sample contents per group without deleting; the production mode hard-deletes each junk episode via `delete_episode` (`memory_service.py:1187-1203`), which runs `MATCH (e:Episodic {uuid: $uuid}) DETACH DELETE e` against the Neo4j driver. Returns include both per-group breakdown and total count.

This is one of the few hard-delete paths in the codebase (most graph cleanup is soft-delete via `expired_at`). Episodic nodes are append-only event logs by design, so once they're identified as junk there's no audit value in keeping them.

## Related

- [03-memory-model](./03-memory-model.md) — canonical scope, category, and group_id semantics
- [05-llm-extraction](./05-llm-extraction.md) — extraction prompt, category parser, junk patterns
- [06-storage-backends](./06-storage-backends.md) — Qdrant payload schema and Neo4j/Graphiti shape
- [07-async-pipeline](./07-async-pipeline.md) — how `extract_and_store` is invoked from ARQ workers
- [02-service-architecture](./02-service-architecture.md) — the full FastAPI + MCP + worker topology
