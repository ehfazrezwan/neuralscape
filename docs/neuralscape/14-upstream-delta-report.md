# Neuralscape Upstream-Delta Report: How Far Behind, What We Patched, and the Sync Risk Surface

*Refreshed 2026-11-16 against dev as it stands (the vendored `mem0/` 2.0.2 and `graphiti/` 0.28.2 subtrees) and live upstream state. Sections 1, 2 and 5 are current. Sections 3 and 4 are the upstream-change analysis measured on 2026-06-22 against mem0 2.0.7 and graphiti 0.29.2; they are kept because the API-stability findings still guide a sync, but upstream has moved since (section 1), so re-check them before relying on a version-specific claim. The earlier edition of this report described a roughly 20-file Graphiti saga/prompt patch cluster that no longer exists on dev; it is gone from this edition.*

---

## 1. How far behind we are (measured 2026-11-16)

| Subtree | Vendored | Recorded upstream base | Upstream now | Gap | Evidence |
|---|---|---|---|---|---|
| **mem0** | `mem0ai` 2.0.2 (`mem0/pyproject.toml`) | `c9e8482a` (2026-05-14), synced 2026-05-15 by `cd27fc57` / PR #38 | `f1791daa` (HEAD, 2026-11-06); latest release Python SDK **v2.2.1** (2026-09-25) | **500 commits ahead**; 2.0.2 -> 2.2.1 on the release line | `git ls-remote https://github.com/mem0ai/mem0 HEAD`; `gh release list -R mem0ai/mem0 -L 5`; `gh api repos/mem0ai/mem0/compare/c9e8482a...HEAD --jq .ahead_by` |
| **graphiti** | `graphiti-core` 0.28.2 (`graphiti/pyproject.toml`) | `feafc422` (2026-04-01), squashed in by `9cb6dc1b` (2026-04-02), merged by `6d87b441` | `3eafcc81` (HEAD, 2026-11-15); latest release **v0.31.2** (2026-10-20) | **264 commits ahead**; 0.28.2 -> 0.31.2 | `git ls-remote https://github.com/getzep/graphiti HEAD`; `gh release list -R getzep/graphiti -L 5`; `gh api repos/getzep/graphiti/compare/feafc422...HEAD --jq .ahead_by` |

Both counts include CI, bot and dependency-bump commits; they say "how stale", not "how much changed that matters". On 2026-06-22 the same measurement gave 168 (mem0) and 91 (graphiti) commits behind, so the gap has roughly tripled in five months.

**A resync is out of scope without the owner.** Nothing on dev is broken by being behind. A sync means re-grafting the mem0 graph layer onto a core that deleted it (section 2), a human decision on Graphiti's extraction-prompt changes (section 3), and a full container-gate and integration run. Do not start one as part of cleanup work; the owner decides whether and when. When one happens, `scripts/sync-upstream.sh` runs the marker check and the fork smoke tests described in section 5.

## 2. What NeuralScape carries on top of upstream (current)

### 2.1 graphiti: small, fully marked

`git diff 9cb6dc1b:graphiti_core HEAD:graphiti/graphiti_core` (`9cb6dc1b` is the squash of upstream `feafc422`, the last Graphiti sync) touched **10 files** (9 patched, 1 NS-owned) once the fork-marker change (PR #475) landed; `scripts/upstream-patches.txt` is the authoritative patched-file list (a test re-derives the graphiti set from this diff) and supersedes this table:

| File | What the delta does | Marker |
|---|---|---|
| `graphiti_core/llm_client/gemini_client.py` | default models changed to `gemini-3.1-flash-lite`, transient-error detection that walks the cause/context chain, output-token ceiling for that model, original error text preserved on re-raise, fallback-model failover on 503/overload that passes the fallback model per call (`_generate_response(model_override=...)`, never mutating `self.model`), and a bounded Gemini HTTP timeout on the client | `NEURALSCAPE PATCH` (6 sites) |
| `graphiti_core/llm_client/config.py` | adds `fallback_model` to `LLMConfig` and the `GEMINI_HTTP_TIMEOUT_MS = 60_000` constant (milliseconds) | `NEURALSCAPE PATCH` |
| `graphiti_core/embedder/gemini.py`, `graphiti_core/cross_encoder/gemini_reranker_client.py` | `genai.Client(..., http_options=HttpOptions(timeout=GEMINI_HTTP_TIMEOUT_MS))` so a stalled call cannot hang a worker slot | `NEURALSCAPE PATCH` |
| `graphiti_core/utils/maintenance/edge_operations.py`, `node_operations.py` | optional bounded-decision fast path behind `HYBRID_GRAPH_ENABLED` (PR #471) | `NEURALSCAPE PATCH` |
| `graphiti_core/llm_client/hybrid_decision_client.py` | NS-only hybrid decision client (not upstream; listed `owned` in the manifest) | n/a |
| `graphiti_core/graphiti.py` | `add_episode(update_communities=True)` flattens the per-node `(communities, edges)` results instead of unpacking N pairs into two names (upstream raised `ValueError` after the episode was stored unless exactly two entities resolved) | `NEURALSCAPE PATCH` |
| `graphiti_core/driver/neo4j_driver.py` | `kwargs.setdefault('database_', ...)` so queries run on the configured database (upstream put it in `params`) | `NEURALSCAPE PATCH` |
| `graphiti_core/search/search.py` | `search()` returns empty results, and runs no query, for an empty group list (`[]` or `['']`); `None` keeps its meaning (no group filter). Upstream treated `['']` like `None`, so a caller on Graphiti's default `''` group sees empty results from this entry point; NS never uses the `''` group | `NEURALSCAPE PATCH` |
| `graphiti_core/utils/maintenance/graph_data_operations.py` | `retrieve_episodes()` returns no episodes for an empty group list; `None` keeps its meaning (no group filter) | `NEURALSCAPE PATCH` |

The saga lifecycle, `reference_time` edge field, extraction/dedup prompt patches and `summarize_sagas.py` that the 2026-06-22 edition listed are **not in the tree**: the April sync took upstream's own implementations, and the diff above proves it. There is no saga or prompt conflict cluster to resolve on a sync. `graphiti/` is pruned to `graphiti_core` plus packaging (`d3978abe`, 2026-07-05).

### 2.2 mem0: an NS-maintained graph layer plus a few patched files

- **The OSS graph layer is an NS fork, not an upstream patch.** mem0 deleted its self-hostable graph layer in PR #4805 (`a488e190`, 2026-04-14, the v3 pipeline): `mem0/graphs/` and the `*_memory.py` graph providers no longer exist upstream, and graph memory is now a hosted-Platform-only feature. NS restored and maintains `mem0/mem0/graphs/` and `memory/{graph,memgraph,kuzu,apache_age}_memory.py`, plus `GraphStoreFactory` in `utils/factory.py`. Only the `graphiti` provider is wired in production.
- **`memory/graphiti_memory.py` is NS-original.** It never existed upstream (`MemoryGraph` presenting Graphiti as a mem0 graph provider, and `_AsyncBridge`, the single background event loop all graph calls run on). It carries the small-model default, the reranker `config=` fix, the episode-identity return value, the idempotent episode name, a bounded bridge wait, a first-community build when `UPDATE_COMMUNITIES` is on (`_ensure_group_communities`), and a hard error instead of the legacy `project:{id}` / `global` / bare-`user_id` group fallback (every service call passes an explicit `group_id`).
- **The re-attach shim.** mem0 2.x `Memory` has no `.graph`, so `neuralscape-service/memory/core.py` `_get_memory` fabricates the config namespace the adapter reads and sets `Memory.graph` itself. Every sync re-grafts this onto a core that deliberately removed it; it is the dominant sync risk.
- **Patched mem0 core/embedding files:**

| File | What the delta does |
|---|---|
| `mem0/mem0/memory/main.py` | lock plus cold-start tolerance for the lazy `<collection>_entities` store (this refresh; before it, concurrent first touches raced on Qdrant create-collection) |
| `mem0/mem0/graphs/configs.py` | `graphiti_llm_small_model` and related config fields for the Graphiti graph store |
| `mem0/mem0/memory/graphiti_memory.py` | the adapter itself (above) |
| `mem0/mem0/embeddings/openai.py`, `mem0/mem0/configs/embeddings/base.py` | `embedding_batch_size`: chunk size and per-item fallback for OpenAI-compatible endpoints that accept one input per request |
| `mem0/mem0/vector_stores/qdrant.py`, `mem0/mem0/configs/vector_stores/qdrant.py` | `timeout`: per-request timeout forwarded to `QdrantClient` (service setting `QDRANT_TIMEOUT`, default 30 s; qdrant-client's own default is about 5 s) |
| `mem0/mem0/embeddings/gemini.py`, `mem0/mem0/llms/gemini.py`, `mem0/mem0/configs/embeddings/base.py`, `mem0/mem0/configs/llms/base.py` | `request_timeout_s`: per-request timeout (seconds) turned into `HttpOptions.timeout` on the Gemini client (service setting `GEMINI_REQUEST_TIMEOUT_S`, default 120 s); unset keeps the SDK default |
| `mem0/mem0/utils/factory.py` | adds `GraphStoreFactory`, marked `NEURALSCAPE PATCH` above the class |

- **Fork environment knob:** `GRAPHITI_BRIDGE_TIMEOUT_S` (documented in `.env.example`) bounds one `_AsyncBridge.run` call; default 960 s, above the 900 s graph/ingest worker `job_timeout`. Before the patch the wait was unbounded.

### 2.3 The authoritative list

`scripts/upstream-patches.txt` is the list of patched files, and `scripts/check-upstream-patches.sh` fails a sync when a listed file loses its `NEURALSCAPE PATCH` marker (PR #475). The manifest has two entry kinds: `patched` (upstream file we modified; must keep a marker) and `owned` (NS-only restored graph layer; must exist). The graphiti base (`9cb6dc1b`) and the mem0 base (`cd27fc57`, the `c9e8482a` sync; the upstream object itself is not in this repo, so the mem0 list is pinned) are recorded in its header and must be bumped after a sync. If the tables above disagree with the file, the file wins and this report is stale.

## 3. Upstream churn that mattered when last measured (2026-06-22, mem0 2.0.7 / graphiti 0.29.2)

Retained from the previous edition. Re-verify against the versions in section 1 before acting on any item.

### 3.1 mem0 2.0.2 → 2.0.7 — what matters to NS

**Vector backends**
- **[NEUTRAL]** Roster **unchanged** — 24 providers, byte-identical `VectorStoreFactory.provider_to_class` at both tags.
- **[WIN] #5380** — expose Qdrant `https` option (`qdrant.py:40,73`). NS's only vector backend; lets it force TLS on remote Qdrant.
- **[NEUTRAL]** Per-backend `get()`-None / filter / reset fixes (#5625, #5562, #5531, #5485, …) — all non-Qdrant; latent until pluggable.

**Scoring / search**
- **[HAZARD→NEUTRAL for NS today] #5391** (`7ac8ab15`) "normalize scores to similarity (higher=better) across all backends" — new contract in `vector_stores/base.py:17-24`. **Qdrant is NOT in the commit's file list** (verified); Qdrant already returns cosine similarity and NS already sorts descending. Not a hazard for the current deployment; it becomes a **pluggability enabler**. Prior analysis calling it a "blocking gate on the sync" is **wrong** (3.3.1).
- **[WIN] #5102 / #5423** — `Memory.search(..., explain=True)` returns per-result **`score_details`** (`utils/scoring.py:118`): `semantic_score, bm25_score, entity_boost, raw_score, max_possible_score, final_score, threshold`. **Correction:** field is `score_details`, *not* `score_breakdown`, and there is **no `temporal_boost` key**. Usable on NS's private-pool `m.search()` path only.
- **[NEUTRAL]** Hardened errors: #5258 reject empty queries, #5482 invalid filters → 400, #5444 warn on hybrid→semantic degrade.

**Embeddings / LLM**
- **[NEUTRAL for graph] #5609 / #5415** — native `embed_batch` for GoogleGenAI/VertexAI/Ollama. **Does NOT resolve NS's graph-embedder batching blocker** — NS's graph path uses *graphiti's* `GeminiEmbedder`, never mem0's. graphiti's `embedder/openai.py create_batch` is **byte-identical vendored↔HEAD**. Helps only mem0's vector-side embedder.
- **[WIN, marginal] #4030** — Vertex-AI as first-class LLM provider + proper `GeminiConfig`. Applies only to mem0's extraction LLM, not graphiti.

**Reranker** — **[WIN, latent] #5636** export all 5 rerankers; **#5560** respect `top_k`; **#5635** clamp scores. NS uses graphiti reranking; latent until NS exposes mem0's `RerankerFactory`.

**Server/security** — **[WIN] #5447** httpx≥0.28 proxy repair via `utils/http.py:build_http_client`; carries a breaking rename (3.3.2).

---

### 3.2 graphiti 0.28.2 → 0.29.2 — what matters to NS

**Drivers**
- **[NEUTRAL]** Roster unchanged (neo4j, falkordb, kuzu, neptune). `driver/driver.py` **zero diff**; the `# Legacy ... Phase 1` marker at `:97` predates 0.28.2. Neo4j driver: no diff. Neptune: no diff (still least-exercised).
- **[WIN, conditional] FalkorDB hardening** — **#1549** is *load-bearing for NS*: default group_id `'\_'`→`'_'` (old was an invalid RediSearch token) + group_id RediSearch escaping rewritten to escape **all** non-alphanumerics (`falkordb_driver.py:127,419`). NS's hyphenated multi-axis group_ids **silently failed to match on FalkorDB pre-0.29.2.** Plus #1531 NUL-byte stripping, #1125/#1536 `falkordblite` embedded. **FalkorDB only becomes viable for NS as of this version.**
- **[DO-NOT-ADOPT] #1548** — **Kuzu deprecated** (DeprecationWarning, dropped from default test matrix). Credible roster narrows to Neo4j (default) + FalkorDB (now viable) + Neptune (unproven).

**Combined extraction + custom types**
- **[WIN, available NOW, no sync] custom entity/edge ontology** — `add_episode()` already accepts `entity_types`/`edge_types`/`edge_type_map` in **both** 0.28.2 and HEAD. NS passes **none** (`graphiti_memory.py:320-328` passes only `group_id`). Projecting NS's 13-category taxonomy (`schemas.py:79-158`) is the **highest-value, lowest-risk graph win and needs no graphiti sync.**
- **[WIN, needs sync + adapter rewrite] #1432/#1443** — combined node+edge extraction (1 LLM call vs 2) via `extract_nodes_and_edges_bulk(use_combined_extraction=True)`, reachable **only from `add_episode_bulk`**, not `add_episode` (`bulk_utils.py:271` default `False`). Capturing it means migrating the adapter to the bulk path — behavioral change (bulk skips cross-episode edge-invalidation/dedup), not a flag flip.
- **[WIN, free on upgrade] #1498** — strengthened extract prompts (attribute-hallucination guards, entity-precision) + 250-char attribute cap. NS gets these automatically on its existing `add_episode` path.

**Sagas** — **[NEUTRAL]** `SagaNode` gained summary fields + `summarize_saga()` + `summarize_sagas.py`. NS never creates sagas — purely additive, no action. (The earlier edition called this the dominant merge-conflict zone because NS then carried a parallel saga implementation; it no longer does, see 2.1.)

**Attribute/group_id safety** — **[WIN, protective] #1498** — attribute write changed to guarded `if k not in entity_data` (`nodes.py:567`, `edges.py:362`), so a hallucinated `group_id` attribute can't overwrite the core field. NS not exposed today (passes `group_id` as first-class kwarg). Resolves prior open question.

**Search** — **[NEUTRAL]** `search.py` +677/−323 is **entirely OpenTelemetry tracing** (`NoOpTracer` default). `search_config.py`, `search_config_recipes.py`, `search_filters.py`, `search_helpers.py`, `search_utils.py` all **byte-identical**. No recipe/RRF/MMR/reranker algorithm changed.

**LLM providers** — **[NEUTRAL] #1551** default → `gpt-5.5`; **#1537** OpenAIGenericClient json_schema default. NS not exposed (explicitly sets gemini models, builds `OpenAIClient` directly). But #1537's json_schema mode is the one lever toward gateway-routing graphiti's LLM.

---

### 3.3 Breaking changes & required NS edits

#### 3.3.1 LEAD: Score-normalization (#5391) — **NOT a hazard for current NS; prior analysis §8.3 is wrong.**
Qdrant is **not in #5391's file list** (verified `git log v2.0.2..HEAD -- mem0/vector_stores/qdrant.py` → only #5380); Qdrant already returns cosine similarity ~[0,1] higher=better; NS already lives in the normalized regime (vendored `main.py` defaults `threshold=0.1`, rejects thresholds outside [0,1]). **Every NS score site is already correct and stays correct** (Qdrant only):

| `memory_service.py` | Code | Verdict |
|---|---|---|
| `:827` | `"score": getattr(hit,"score",None)` (raw `query_points`) | native similarity — safe |
| `:1290` | `deduped.sort(...score..., reverse=True)` | higher=better — safe |
| `:1362/1434` | `_GRAPH_ENRICH_THRESHOLD=0.7; if score<0.7: skip` | correct direction — safe |
| `:2809/2879` | dedup `0.95`; `if hit_score<threshold: continue` | safe |
| `:2923` | `merged.sort(...score..., reverse=True)` | safe |

**The real rule:** #5391 gates **backend-pluggability, not the sync.** If NS ever selects pgvector/milvus *while pinned to 2.0.2*, the `0.95`/`0.7` thresholds rank backwards on distance-based backends. Sync mem0 ≥2.0.7 **before** shipping any non-Qdrant backend; then thresholds work unchanged everywhere. **Downgrade from "blocking" to "verify-only / pluggability-enabler."**

#### 3.3.2 `config.http_client` semantics flip (#5447) — *most likely real mem0 break.*
`LlmFactory` now reads `config.http_client_proxies`; `config.http_client` now returns a built `httpx.Client`, not the proxies dict. **Action:** audit `config.py` mem0 config-dict build for any `config.http_client` read expecting a dict.

#### 3.3.3 Empty-query rejection (#5258) / invalid-filter 400 (#5482).
mem0 native `search("")` now raises. **Action:** audit NS recall paths that could pass `""` to mem0 native `search()`.

#### 3.3.4 graphiti attribute-merge guard (#1498) — non-breaking for NS, confirmed safe.

#### 3.3.5 Episode index 0-based (#1432) — not exposed (NS sends one episode per `add_episode`).

#### 3.3.6 Model/field additions (graphiti) — additive, non-breaking. `add_episode` gained `saga`/`custom_extraction_instructions` (keyword-defaulted) — NS's call at `:320` is forward-compatible. `MAX_SUMMARY_CHARS` 500→1000 is cosmetic (verify wiki synthesizer has no 500 assumption).

#### 3.3.7 graphiti patches that are NOT obsoleted — **keep all three** (corrects prior optimism):
- **small_model/gpt-4.1-nano** (`graphiti_memory.py:41-51`): HEAD still hard-defaults `DEFAULT_SMALL_MODEL='gpt-4.1-nano'` (`openai_base_client.py:35`; #1551 only changed the *main* default). Gateway can't provision nano. **Keep.**
- **Reranker `config=` fix** (`:99-104`): `OpenAIRerankerClient.__init__` at HEAD still takes `config`, not `api_key=`. **Keep.**
- **Embedder-on-AI-Studio** (config.py:306-321): graphiti's `create_batch` byte-identical vendored↔HEAD — still sends multi-input that Vertex rejects with 400, no per-item fallback. **Keep.** Upgrading unlocks **zero** of these three.

---

## 4. Recommended sync plan (for the owner, when a sync is chosen)

**Sync mem0 first, graphiti second.** mem0 has a handful of small patches plus the orphaned graph layer; graphiti's NS delta is three files, so its mechanical cost is low and the work is deciding what to do with Graphiti's changed extraction prompts and saga/edge behaviour (a human prompt-engineering call, not a `git merge`).

1. **Pre-flight.** Read the section 3 audits (`config.http_client` consumers, empty-query search, `limit` -> `top_k`). Confirm `scripts/upstream-patches.txt` lists every patched file.
2. **Run the sync** with `scripts/sync-upstream.sh` (filtered mirror, core paths only). It runs `uv sync`, the marker check, the mocked unit suite, then the fork smoke tests.
3. **Re-apply** whatever the marker check reports as lost, then re-run.
4. **Gate.** The repo's Container Build Gate applies in full (Docker build plus in-container suite on the exact tree), and the fork smoke tests must pass against real Qdrant and Neo4j with `GOOGLE_API_KEY` set: `cd neuralscape-service && NS_INTEGRATION_REQUIRED=1 pytest integration_tests/test_fork_smoke.py -v`.
5. **Highest-risk item:** the re-attach shim. The smoke tests assert that `Memory.graph` is a `MemoryGraph`, that the bridge runs real Neo4j queries, and that one real episode round-trips through add, search and an edge query by `group_id`.

## 5. The safety net that now exists

Before this refresh the two forks had **zero tests** (pruned with upstream's tests in `d3978abe`), and every service test mocked Qdrant, Neo4j and Gemini, so a sync that changed `Memory` internals, a Qdrant signature (the `top_k` rename already broke `memory/delete.py` once) or Graphiti's episode pipeline would pass the suite and fail in production.

- `neuralscape-service/integration_tests/test_fork_smoke.py`: real-store smoke tests for `_memory_from_config` (collection created with the BM25 sparse slot, idempotent), `vector_store.insert` / `query_points` / `search(top_k=)` / `delete`, the BM25 leg, the lazy entity store under concurrent first touch, the re-attach shim and bridge against real Neo4j, and one real Graphiti episode (add, search, edge query). Skips without stores; fails when `NS_INTEGRATION_REQUIRED=1`.
- Unit tests for the patches this refresh made: `tests/test_fork_entity_store_lock.py`, `tests/test_fork_bridge_timeout.py`, `tests/test_cold_start_schedule.py`.
- `scripts/sync-upstream.sh` runs the smoke tests (when `QDRANT_URL` is set) and tells the operator to run them otherwise.

## 6. Open questions / verify-before-sync

1. **Graph re-attach shim survival.** Whether a newer mem0 `Memory` still lets the shim set `.graph` and expose `.graph.graphiti` / `._bridge`. The smoke tests cover it; run them against the new checkout before merging a sync.
2. **`config.http_client` consumers (mem0 #5447)** and **empty-query rejection (#5258)**: audit as in section 3.
3. **Strategic option (unchanged):** since NS already reaches past mem0 into Graphiti's bridge directly, retiring the "impersonate a mem0 graph provider" indirection in favour of an NS-owned graph port would remove the shim as a permanent re-graft. Fold into the pluggable-backend decision (`13-pluggable-backend-gap-analysis.md`).

## 7. Vendored but intentionally unused

Each inert-looking capability of the two forks was traced to its callers and sorted into one of two kinds. **(A) NeuralScape work that was started** was wired to a real caller and proven; **(B) upstream code NeuralScape never adopted** stays vendored (it keeps syncs mechanical) and is fenced by `neuralscape-service/tests/test_vendored_unused_fence.py`, which fails when NS code starts importing it.

### 7.1 Kind A: wired and proven

| Capability | Evidence it is NS work | State on dev | Proof |
|---|---|---|---|
| Graph communities (`UPDATE_COMMUNITIES`) | NS setting (`config.py`), passed into `add_episode` by the adapter, read back by `GET /v1/graph/communities` and the `get_graph_communities` tool | **Was broken when on**: (1) upstream `add_episode` crashed after storing the episode unless exactly two entities resolved; (2) nothing created a group's first community (`update_community` only joins existing ones), so the endpoint stayed empty. Both fixed (`NEURALSCAPE PATCH` in `graphiti.py` and `graphiti_memory.py`); the first build is scoped to the written group because `Graphiti.build_communities` calls `remove_communities`, which deletes every group's communities. The build is idempotent across workers (uuids derived from group and cluster, MERGE saves) and a community with no `HAS_MEMBER` edge counts as not built and is rebuilt | `tests/test_fork_inert_capabilities.py`, `integration_tests/test_graph_communities_flag_on.py` |
| Cross-encoder reranker (`_create_cross_encoder`, `graphiti_reranker_provider`) | NS adapter code and NS settings; reachable because `POST /graph/search` and `POST /v1/graph/search` build a `SearchConfig` from the caller's `search_config`, so `"reranker": "cross_encoder"` reaches `cross_encoder.rank` | Works; no named recipe uses it, so it only runs on request | `tests/test_fork_inert_capabilities.py`, `integration_tests/test_graph_cross_encoder_search_config.py` |
| mem0 SQLite history | The service itself calls `m.db.add_history` on every write (`memory/write.py`) and mem0 does on `_update_memory` / `_delete_memory`, but nothing reads it | Writes disabled by default (`MEM0_HISTORY_ENABLED=false`): `get_mem0_config` points mem0 at a throwaway in-memory database and `_memory_from_config` swaps in a no-op store. Set `MEM0_HISTORY_ENABLED=true` for the file-backed history | `tests/test_fork_inert_capabilities.py` |

The dreaming communities cron in `docs/UPSTREAM_EDGE_EXPERIMENTS.md` E-3 (`DREAMING_COMMUNITIES`, a periodic rebuild off the write path) and the `RERANK_ENABLED` fused-list reranker of E-1 were never started (no code anywhere) and remain out of scope.

### 7.2 Kind B: fenced, not deleted

| Vendored code | Why it is not NS's | Fence |
|---|---|---|
| mem0 `Memory.add` / `search` / `update`, the hybrid `_search_vector_store` (lemmatisation, BM25 + entity boost), the `<collection>_entities` store (with its NS lock patch) | The legacy `POST /memories` and `POST /search` routes now call `MemoryService.extract_and_store` / `search`; the service writes and reads Qdrant itself. The only `Memory` methods it calls are `get`, `get_all`, `delete`, `delete_all`, `_update_memory`, plus the `vector_store`, `db` and `embedding_model` attributes. `_remove_memory_from_entity_store` is a no-op until something creates the entity store | AST pin on every `Memory` handle in the service |
| mem0 vector stores other than Qdrant, LLMs and embedders other than Gemini / OpenAI, `client/`, `proxy/`, `reranker/`, the legacy graph providers (`graph_memory`, `memgraph`, `kuzu`, `apache_age`, `graphs/neptune`) | `get_mem0_config` names only `qdrant`, `gemini` / `openai` and `graphiti`; `GraphStoreFactory` still lists the others but nothing selects them | import-graph pin; config provider pin; runtime check that building the adapter's clients loads none of them |
| Graphiti FalkorDB, Kuzu and Neptune drivers; Anthropic, Azure, Groq, GLiNER2 and generic OpenAI LLM clients; Azure and Voyage embedders; the BGE reranker | Only the Neo4j driver and the Gemini / OpenAI clients are constructed | import-graph pin |
| Graphiti `add_episode_bulk` and `bulk_utils`, sagas, `content_chunking`, `namespaces`, `migrations`, the OpenTelemetry tracer | No service call site (E-4 in `UPSTREAM_EDGE_EXPERIMENTS.md` is the unstarted plan for bulk) | import-graph pin plus a method-name pin (`add_episode_bulk`, `build_communities`, `remove_communities`, `saga=`) |
| 15 of Graphiti's 16 named search recipes | The service imports `EDGE_HYBRID_SEARCH_RRF` only. The recipes' building blocks (node-distance, MMR, episode-mentions and cross-encoder rerankers) are reachable through a caller's `search_config` and are not fenced | pin that the only imported recipe is `EDGE_HYBRID_SEARCH_RRF` |

To adopt a fenced piece, wire and test it, then move it out of the lists in `test_vendored_unused_fence.py` and out of this table in the same PR. Fencing adds no markers to the vendored trees, so `scripts/upstream-patches.txt` is unchanged by it; it gained one line only for the `graphiti.py` communities patch.

### 7.3 Dead code in the service itself

`main._get_async_memory()` builds an `AsyncMemory` that nothing calls (the legacy routes use `MemoryService`); only the shutdown hook references its handle. It was left alone here because `main.py` belongs to the routes owner.

