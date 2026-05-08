---
title: Storage Backends & Subtree Dependencies
date: 2026-05-06
tags: [reference, neuralscape, qdrant, neo4j, graphiti, mem0]
source: handwritten
---

# Storage Backends & Subtree Dependencies

## Overview

Neuralscape persists memory across two physically distinct stores, glued together by the `mem0` orchestration layer. Each fact written by the service ends up in **both** systems, indexed differently:

- **Qdrant** — vector store for semantic similarity search. Holds embedded fact text plus rich payload metadata (scope, category, project, tags, hash, timestamps). Used for nearest-neighbour retrieval.
- **Neo4j + Graphiti** — temporal knowledge graph. Holds entities, relationships, episodes, and communities. Used for graph traversal, hybrid search, and temporal reasoning.
- **mem0** — Python library that exposes a unified `Memory` API and dispatches each `add` / `search` call into both backends. mem0 owns the embedder and LLM clients; Graphiti owns its own copies, configured side-by-side.

Both `mem0` and `graphiti-core` are pulled in as **git subtrees** (not submodules) under `mem0/` and `graphiti/`, then installed as editable packages via `uv` workspace sources. Custom commits live directly on top of upstream history — there are no `.patch` files.

This page documents every moving piece: the configuration dict that wires everything up, the Qdrant payload schema and filter quirks, the Neo4j database/group-id conventions, the Graphiti API surface used by `MemoryGraph`, the async-bridge that makes a synchronous service talk to an async Neo4j driver, and the upstream-sync workflow.

## Architecture

Every successful write fans out across both stores. Reads fan in.

**Write path** (see [04-memory-service-core](./04-memory-service-core.md)):

1. `MemoryService.extract_and_store` calls Gemini to extract structured facts from a conversation.
2. `_batch_store_facts` embeds all facts in a **single** Gemini batch call, then upserts them into Qdrant via `vector_store.insert(vectors, ids, payloads)` — one network call per extraction (`memory_service.py:545-553`).
3. The cleaned conversation text is then handed to Graphiti via `self._memory.graph.add(data=raw_text, filters={"user_id", "group_id"})`. Graphiti runs its own LLM extraction pass internally (entity + relation extraction, dedup, edge invalidation) and writes nodes/edges into Neo4j.

The two stores hold **different shapes of the same conversation**: Qdrant gets categorized, hashed, dedup-friendly fact strings; Neo4j gets the raw narrative as a temporal episode plus the entities/relations Graphiti derives from it.

**Read path**: search hits Qdrant for vector similarity (with optional dual-scope merge) and Graphiti for graph-aware hybrid retrieval (semantic + keyword + traversal). Results are merged in `MemoryService` before returning. The `metadata.*` payload prefix and the `project--{id}` group-id format are the two cross-store conventions that keep results from leaking between scopes.

See [03-memory-model](./03-memory-model.md) for scope/category semantics and [07-async-pipeline](./07-async-pipeline.md) for how writes get queued through Redis/ARQ before reaching either store.

## Subtree dependencies

The repo carries two large vendored projects in-tree:

- **`graphiti/`** — full copy of `getzep/graphiti` v0.28.2. Contains `graphiti_core/` (the library), `server/` (a standalone API), `mcp_server/` (Graphiti's own MCP), and `tests/`. Neuralscape only consumes `graphiti_core` via the editable install.
- **`mem0/`** — full copy of `mem0ai/mem0` v1.0.10. Contains `mem0/memory/` (with the custom `graphiti_memory.py` adapter), `mem0/vector_stores/qdrant.py` (with the metadata-prefix filter fix), `mem0/configs/`, `mem0/embedchain/`, and `tests/`.

These were brought in via `git subtree add`, which means upstream history is squashed into local commits. Updates are pulled with `git subtree pull --squash`, and any local changes ride on top as ordinary commits.

### Custom commits on top of upstream

There are four neuralscape-specific commits layered onto subtree history:

- `6c94f20` — `fix(search): prefix metadata keys in Qdrant filters for vector search`. Required because mem0 stores user payload under a nested `metadata` object, but its filter builder previously emitted top-level keys.
- `2edd9b5` — `fix(graphiti): align group_id format to use double-hyphen for project scoping`. Standardises on `project--{id}` to avoid collisions with the `:` separator Graphiti uses internally.
- `f591781` — `feat(conversation-compiler): rebuild category index after memory_stored vault writes`. Hooks into the worker event stream.
- `2cf1da8` — `feat(worker): emit memory_stored events`. Producer side of the same event.

There are no `.patch` files anywhere. Every modification is a real commit visible in `git log`.

### uv sources wiring

Both subtrees are declared as editable path dependencies in the service's pyproject:

```toml
# neuralscape-service/pyproject.toml:33-35
[tool.uv.sources]
graphiti-core = { path = "../graphiti", editable = true }
mem0ai = { path = "../mem0", editable = true }
```

mem0 itself depends on `graphiti-core` and overrides it the same way:

```toml
# mem0/pyproject.toml:166-167
[tool.uv.sources]
graphiti-core = { path = "../graphiti", editable = true }
```

mem0 also declares an optional extra so that downstream consumers can pull a Graphiti-enabled build without the editable override:

```toml
# mem0/pyproject.toml:71-72
graphiti = ["graphiti-core[google-genai]>=0.28.0"]
```

The net effect: `uv sync` from `neuralscape-service/` resolves both `mem0ai` and `graphiti-core` to the in-tree subtrees, and any edit you make under `mem0/` or `graphiti/` is immediately picked up without reinstall.

## mem0 configuration

The single function that wires every backend together is `Settings.get_mem0_config()` in `neuralscape-service/config.py:79-132`. It returns the dict consumed by `Memory.from_config(...)`.

```python
# neuralscape-service/config.py:92-132 (abridged)
return {
    "llm":          {"provider": "gemini", "config": {"model": ..., "api_key": ...}},
    "embedder":     {"provider": "gemini", "config": {"model": ..., "embedding_dims": 768}},
    "vector_store": {"provider": "qdrant",   "config": qdrant_config},
    "graph_store":  {"provider": "graphiti", "config": {...}},
    "version":      "v1.1",
}
```

Section by section:

- **`llm`** — Gemini, used by mem0 itself (not Graphiti). The same `gemini_llm_model` value is reused, but mem0's built-in extraction pipeline is **bypassed**: `MemoryService` calls Gemini directly via `google.genai.Client` for category-aware extraction (see [05-llm-extraction](./05-llm-extraction.md)). mem0's LLM is only used for residual operations like `Memory.update`.
- **`embedder`** — Gemini's `gemini-embedding-001` at 768 dimensions. This single embedder is used for every Qdrant write and read. Graphiti has its own embedder (`graphiti_embedder_*` keys) so the two stores can in principle diverge — though in this config both point at the same Gemini model.
- **`vector_store`** — Qdrant. The dict is built dynamically: if `QDRANT_URL` is set the config gets `url` (server mode), otherwise it gets `path` and `on_disk` (local mode). Collection name is hardcoded to `neuralscape_memories` and dim to 768.
- **`graph_store`** — Graphiti, with its own `graphiti_llm_*`, `graphiti_embedder_*`, and `graphiti_reranker_*` triplets. `database: "memory"` is non-default — Graphiti must run against a Neo4j database named `memory`, not the default `neo4j`. `store_raw_episode_content=True` keeps the verbatim conversation text on each episode node; `update_communities=False` skips Leiden community recomputation on every write.
- **`version: "v1.1"`** — selects the v1.1 schema in mem0 (post-breaking-change).

`Memory.from_config(config)` is called lazily and **once** in `MemoryService._get_memory()` (`memory_service.py:203-221`), guarded by `self._init_lock` so concurrent cold-start requests don't double-instantiate. Once initialized, the underlying Graphiti instance is cached as `self._memory.graph.graphiti` and the async bridge as `self._memory.graph._bridge`.

## Qdrant

**Collection.** Single collection `neuralscape_memories` (config.py:29). 768-dim vectors. COSINE distance — that's the mem0 default and is not overridden.

**Storage modes.** Same `QdrantClient` API in both modes; the switch is purely config:

- **Local on-disk**: `QdrantClient(path="~/.neuralscape/qdrant")`. No server process required. Used in dev and CI.
- **Server**: `QdrantClient(url="http://localhost:6333")`. Used in Docker compose and production.

The mem0 wrapper at `mem0/mem0/vector_stores/qdrant.py:57-72` handles both. One important caveat: **payload indexes are only created in server mode**. Local on-disk Qdrant doesn't support `create_payload_index`, so categorical filters fall back to scan-based filtering.

**Payload schema.** Every Qdrant point carries:

```python
{
    "data": "<fact text>",
    "hash": "<md5 of content>",
    "created_at": "<ISO 8601 UTC>",
    "user_id": "...",
    "agent_id": "..." | None,
    "run_id": "..." | None,
    "metadata": {
        "scope": "global" | "project",
        "category": "<one of 13>",
        "project_id": "..." | None,
        "agent_id": "..." | None,
        "run_id": "..." | None,
        "source": "conversation" | "explicit" | "graph",
        "tags": ["..."] | None,
    },
}
```

Origin: `memory_service.py:407-424` (single-fact path) and `memory_service.py:523-537` (batch path). `hash` is MD5 of the content string and is used for fact-level dedup before re-embedding (see `_batch_store_facts`).

**The metadata-prefix gotcha.** This is the single most common foot-gun in the codebase. Because user data lives nested under `metadata`, every filter must use a dotted key:

```python
filters = {
    "metadata.project_id": "neuralscape",
    "metadata.scope": "project",
    "metadata.category": {"in": ["tech_stack", "convention"]},
}
```

Top-level keys (`user_id`, `agent_id`, `run_id`) stay un-prefixed because they live at the payload root. The fix that aligned this was commit `6c94f20`. Filter operators are the standard Qdrant set, exposed by `mem0/mem0/vector_stores/qdrant.py:142-283`: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `nin`, `contains`, `icontains`, plus boolean `AND`, `OR`, `NOT`.

**Dual-scope search.** When a search supplies `project_id` but no explicit `scope`, `MemoryService.search` runs **two** Qdrant queries — one for `metadata.scope = "project"` with the matching `project_id`, one for `metadata.scope = "global"` — then merges results by score and dedups by ID (`memory_service.py:622-643`). This is why a project-scoped agent transparently sees both project-specific and user-wide memories.

**Batching.** `_batch_store_facts` (`memory_service.py:545-553`) issues a single `embed_batch()` call to Gemini and a single `vector_store.insert(vectors, ids, payloads)` call to Qdrant per extraction, regardless of how many facts the LLM returned. This keeps write amplification linear in extractions, not facts.

## Neo4j + Graphiti

**Database.** Single Neo4j instance over the bolt protocol at `neo4j://127.0.0.1:7687`. The target database is `memory` — **not** the Neo4j default `neo4j`. The database must already exist before the service starts; it isn't auto-created. User defaults to `neo4j`; password is required.

**Indices and constraints.** On first use, `MemoryGraph._ensure_indices` runs `graphiti.build_indices_and_constraints()` on the bridge loop (`mem0/mem0/memory/graphiti_memory.py:192-200`). Defined in `graphiti/graphiti_core/graph_queries.py`, this creates:

- **EntityNode**: range indices on `uuid`, `group_id`, `name`, `created_at`; fulltext on `name + summary`.
- **EpisodicNode**: range indices on `uuid`, `group_id`, `created_at`, `valid_at`; fulltext on `content`.
- **CommunityNode**: range index on `uuid`; fulltext on `name`.
- **SagaNode**: range indices on `uuid`, `group_id`, `name`.
- **RELATES_TO** edge: range indices on `uuid`, `group_id`, `name`, `created_at`, `expired_at`, `valid_at`, `invalid_at`; fulltext on `name + fact`.
- **MENTIONS / HAS_MEMBER / HAS_EPISODE / NEXT_EPISODE**: range indices on `uuid`, `group_id`.

The temporal indices (`valid_at`, `invalid_at`) are what make Graphiti's bi-temporal queries fast.

**group_id scoping.** All Graphiti reads/writes are scoped by `group_id`. Format (locked in by commit `2edd9b5`):

- Global: `"global"`
- Project: `"project--{project_id}"` (double-hyphen on purpose — single-hyphen project IDs would otherwise be ambiguous)
- Multi-group search: `["global", "project--{id}"]` so a project search includes overarching memories.

Built by `_build_group_id` and `_get_group_ids` in `memory_service.py:175-187`, with a parallel implementation in `mem0/mem0/memory/graphiti_memory.py:202-228` that handles the legacy `user_id`-as-group fallback for older callers.

**Episode model.** Every conversation becomes one episode:

- `name = f"mem0_episode_{datetime.isoformat()}"`
- `episode_body` = cleaned conversation text (role-prefixed lines)
- `source = EpisodeType.text` (not JSON, not audio)
- `reference_time = datetime.now(timezone.utc)` — anchors temporal reasoning
- `group_id` = scope-derived as above
- `update_communities = False` — skipped by default

`add_episode` returns a list of `EntityEdge` objects: `(source_node_uuid, target_node_uuid, name, fact)` — the relations Graphiti extracted from the episode.

**Async bridge.** Graphiti's Neo4j driver is async, but `MemoryService` is synchronous and called from FastAPI's threadpool. To bridge the two, `MemoryGraph.__init__` spins up a dedicated daemon thread running its own `asyncio` event loop (`mem0/mem0/memory/graphiti_memory.py:102-123`):

```python
class _AsyncBridge:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()
```

The Neo4j driver is constructed **on** that loop so it binds correctly. Every subsequent `add_episode` / `search` call goes through `self._bridge.run(coro)`. This avoids the `RuntimeError: Future attached to a different loop` that would otherwise hit on every request, and it keeps Graphiti's async machinery encapsulated behind a sync facade.

## Graphiti API surface

The neuralscape codebase touches a small slice of Graphiti — there's no usage of sagas, custom search recipes, or community APIs beyond the disabled toggle. The methods called are:

- **`graphiti.add_episode(name, episode_body, source_description, reference_time, source=EpisodeType.text, group_id, update_communities=False)`** — entry point for all graph writes. Internally runs entity extraction, edge extraction, dedup, and edge invalidation. Returns extracted entities + edges.
- **`graphiti.search(query, group_ids, num_results)`** — hybrid search combining semantic embedding similarity, keyword (BM25-style) match, and graph traversal. Returns `EntityEdge` results, each with `source_node_uuid`, `target_node_uuid`, `name` (the relationship label), and `fact` (the textual description).
- **`graphiti.build_indices_and_constraints()`** — schema bootstrap. Idempotent; `_ensure_indices` swallows the "already exists" failure path.
- **`EntityEdge.get_by_group_ids(driver, group_ids, limit)`** — used to enumerate all relationships for a given scope (e.g. for the list/list-all endpoints).
- **`Node.delete_by_group_id(driver, group_id)`** — purges every node tagged with a given group_id. Used when a project is deleted.
- **`graphiti.close()`** — driver shutdown, called by `MemoryService.close()` via the bridge.

Community detection (`update_communities=True`) is wired but disabled by config to keep write latency predictable; turning it on triggers Leiden recomputation on every episode.

## Subtree sync workflow

Pulling upstream changes is scripted via `neuralscape-service/scripts/sync-upstream.sh`:

```bash
./scripts/sync-upstream.sh [graphiti|mem0|all]
```

Steps the script performs:

1. `git subtree pull --prefix=graphiti upstream-graphiti main --squash` (or the mem0 equivalent against `upstream-mem0`).
2. Prints a warning enumerating the **conflict hot zones** in mem0: `mem0/mem0/configs/*.py`, `mem0/mem0/utils/factory.py`, and `mem0/mem0/memory/graphiti_memory.py`. These are the files most likely to need manual resolution because every neuralscape commit touches one of them.
3. After the subtree pull, the operator must `uv sync` to re-lock dependencies and run the unit tests with `uv run pytest tests/ --ignore=tests/test_async_pipeline.py` to catch adapter regressions before integration tests.

There is no automatic rebase of the four custom commits — `git subtree pull --squash` lays down a single squashed merge commit, and any prior local edits to subtree files come along as part of regular history. Conflicts in `configs.py` typically come from upstream renaming a field; conflicts in `factory.py` come from upstream registering new providers; conflicts in `graphiti_memory.py` are usually superficial (logging, type hints).

## Local vs remote modes

**Qdrant** is the only backend with a real mode switch:

- `QDRANT_URL` set (`http://localhost:6333` or remote) → server mode, full payload-index support, suitable for production.
- `QDRANT_URL` unset → local on-disk at `QDRANT_PATH` (default `~/.neuralscape/qdrant`). No external dependency, but no payload indexes — filter performance degrades on large collections.

Same Python API (`QdrantClient`) handles both; mem0 picks the constructor args from config. Migration from local to server is one-way and manual (re-import or use Qdrant's snapshot tooling).

**Neo4j** has no equivalent mode switch. It's always a single live instance over bolt — no clustering, no embedded mode. The `database` setting is hardcoded to `memory` and is the only Neo4j knob the service exposes.

See [11-deployment](./11-deployment.md) for Docker compose specifics and how the on-disk Qdrant path is mounted into the container.

## Environment variables

The five env vars that govern storage behaviour:

| Variable | Required | Default | Effect |
|---|---|---|---|
| `GOOGLE_API_KEY` | yes | — | Gemini key reused by mem0 LLM, mem0 embedder, Graphiti LLM, Graphiti embedder, Graphiti reranker. One key, five clients. |
| `NEO4J_PASSWORD` | yes | — | Neo4j auth. Service refuses to start without it (`Settings.validate_required`). |
| `REDIS_URL` | yes | `redis://localhost:6379` | ARQ queue for async writes — not strictly a storage backend, but required for writes to land. |
| `QDRANT_URL` | optional | unset | If set, server mode. If unset, local on-disk mode at `QDRANT_PATH`. |
| `QDRANT_PATH` | optional | `~/.neuralscape/qdrant` | Local on-disk path; ignored when `QDRANT_URL` is set. |

`NEO4J_URI`, `NEO4J_USER`, and `NEO4J_DATABASE` all have sensible defaults (`neo4j://127.0.0.1:7687`, `neo4j`, `memory`). Override them only for non-standard topologies.

## Related

- [04-memory-service-core](./04-memory-service-core.md) — how `MemoryService` orchestrates writes across both stores.
- [03-memory-model](./03-memory-model.md) — scopes, categories, and how they map onto Qdrant payload and Graphiti `group_id`.
- [05-llm-extraction](./05-llm-extraction.md) — the Gemini extraction step that feeds Qdrant; Graphiti runs its own extraction on the raw text.
- [11-deployment](./11-deployment.md) — docker compose layout for Neo4j, Qdrant, Redis.
- [07-async-pipeline](./07-async-pipeline.md) — Redis/ARQ enqueue path that fronts both backends on writes.
