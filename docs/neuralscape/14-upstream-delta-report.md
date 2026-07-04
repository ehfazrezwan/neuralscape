# Neuralscape Upstream-Delta Report: How Far Behind, What Upgrading Buys, and the Sync Risk Surface

*Verified offline on 2026-06-22 against freshly cloned upstream repos (mem0 HEAD 2.0.7 @ 2026-06-21, graphiti HEAD 0.29.2 @ 2026-06-20) and the vendored subtrees. Every claim was checked against actual source — not web research. This report corrects the prior architecture gap analysis (`13-pluggable-backend-gap-analysis.md`) in three places where it was over-cautious or wrong.*

---

## 1. How far behind we are

| Subtree | Vendored | Upstream HEAD | Behind by | HEAD date | Substantive churn (excl. CI/bot/dependabot) |
|---|---|---|---|---|---|
| **mem0** | 2.0.2 | 2.0.7 | 168 commits | 2026-06-21 | Scoring contract, Vertex LLM provider, embed_batch, reranker exports, httpx≥0.28, server error semantics. **Backend rosters unchanged.** |
| **graphiti** | 0.28.2 | 0.29.2 | 91 commits (~60 are CLA/dependabot/CI) | 2026-06-20 | ~28 files `graphiti_core`, +3750/−835, dominated by 3 PRs: #1432 (combined extraction), #1443 (custom edge types in combined prompt), #1498 (saga episode-time watermarks + extraction-prompt rewrite). **Driver roster unchanged.** |

**Verdict.** We are two minor versions behind on each, but the *functional* surface NS depends on is remarkably stable: no factory key renamed, no `create()` signature changed, `add_episode`/`search` byte-identical, vector-store + embedder + driver rosters unchanged on both sides. **The risk is not in upstream's API — it is in our local fork.** Two structural facts dominate:

1. **mem0 deleted its entire graph layer upstream** (PR #4805, v3 pipeline). `mem0/graphs/` and all five `*_memory.py` files (incl. `graphiti_memory.py`) **do not exist in 2.0.2 or HEAD**. NS ships them as a permanent fork. A sync won't conflict on these files; it will silently leave them orphaned while the mem0 internals they hook into evolve.
2. **graphiti independently re-implemented the exact features NS hand-patched** — `reference_time` edges, `SagaNode` summarization fields, `summarize_sagas.py` — with *different code* (all confirmed present at HEAD). This converts most graphiti patches from "clean re-apply" into genuine add/add and semantic conflicts.

Bottom line: **the upgrade is low-risk for our running Qdrant+Neo4j deployment; the merge is the hard part, concentrated in our forked graph layer.**

---

## 2. mem0 2.0.2 → 2.0.7 — what matters to NS

**Vector backends**
- **[NEUTRAL]** Roster **unchanged** — 24 providers, byte-identical `VectorStoreFactory.provider_to_class` at both tags.
- **[WIN] #5380** — expose Qdrant `https` option (`qdrant.py:40,73`). NS's only vector backend; lets it force TLS on remote Qdrant.
- **[NEUTRAL]** Per-backend `get()`-None / filter / reset fixes (#5625, #5562, #5531, #5485, …) — all non-Qdrant; latent until pluggable.

**Scoring / search**
- **[HAZARD→NEUTRAL for NS today] #5391** (`7ac8ab15`) "normalize scores to similarity (higher=better) across all backends" — new contract in `vector_stores/base.py:17-24`. **Qdrant is NOT in the commit's file list** (verified); Qdrant already returns cosine similarity and NS already sorts descending. Not a hazard for the current deployment; it becomes a **pluggability enabler** (§6). Prior analysis calling it a "blocking gate on the sync" is **wrong** (§4.1).
- **[WIN] #5102 / #5423** — `Memory.search(..., explain=True)` returns per-result **`score_details`** (`utils/scoring.py:118`): `semantic_score, bm25_score, entity_boost, raw_score, max_possible_score, final_score, threshold`. **Correction:** field is `score_details`, *not* `score_breakdown`, and there is **no `temporal_boost` key**. Usable on NS's private-pool `m.search()` path only.
- **[NEUTRAL]** Hardened errors: #5258 reject empty queries, #5482 invalid filters → 400, #5444 warn on hybrid→semantic degrade.

**Embeddings / LLM**
- **[NEUTRAL for graph] #5609 / #5415** — native `embed_batch` for GoogleGenAI/VertexAI/Ollama. **Does NOT resolve NS's graph-embedder batching blocker** — NS's graph path uses *graphiti's* `GeminiEmbedder`, never mem0's. graphiti's `embedder/openai.py create_batch` is **byte-identical vendored↔HEAD**. Helps only mem0's vector-side embedder.
- **[WIN, marginal] #4030** — Vertex-AI as first-class LLM provider + proper `GeminiConfig`. Applies only to mem0's extraction LLM, not graphiti.

**Reranker** — **[WIN, latent] #5636** export all 5 rerankers; **#5560** respect `top_k`; **#5635** clamp scores. NS uses graphiti reranking; latent until NS exposes mem0's `RerankerFactory`.

**Server/security** — **[WIN] #5447** httpx≥0.28 proxy repair via `utils/http.py:build_http_client`; carries a breaking rename (§4.2).

---

## 3. graphiti 0.28.2 → 0.29.2 — what matters to NS

**Drivers**
- **[NEUTRAL]** Roster unchanged (neo4j, falkordb, kuzu, neptune). `driver/driver.py` **zero diff**; the `# Legacy ... Phase 1` marker at `:97` predates 0.28.2. Neo4j driver: no diff. Neptune: no diff (still least-exercised).
- **[WIN, conditional] FalkorDB hardening** — **#1549** is *load-bearing for NS*: default group_id `'\_'`→`'_'` (old was an invalid RediSearch token) + group_id RediSearch escaping rewritten to escape **all** non-alphanumerics (`falkordb_driver.py:127,419`). NS's hyphenated multi-axis group_ids **silently failed to match on FalkorDB pre-0.29.2.** Plus #1531 NUL-byte stripping, #1125/#1536 `falkordblite` embedded. **FalkorDB only becomes viable for NS as of this version.**
- **[DO-NOT-ADOPT] #1548** — **Kuzu deprecated** (DeprecationWarning, dropped from default test matrix). Credible roster narrows to Neo4j (default) + FalkorDB (now viable) + Neptune (unproven).

**Combined extraction + custom types**
- **[WIN, available NOW, no sync] custom entity/edge ontology** — `add_episode()` already accepts `entity_types`/`edge_types`/`edge_type_map` in **both** 0.28.2 and HEAD. NS passes **none** (`graphiti_memory.py:320-328` passes only `group_id`). Projecting NS's 13-category taxonomy (`schemas.py:79-158`) is the **highest-value, lowest-risk graph win and needs no graphiti sync.**
- **[WIN, needs sync + adapter rewrite] #1432/#1443** — combined node+edge extraction (1 LLM call vs 2) via `extract_nodes_and_edges_bulk(use_combined_extraction=True)`, reachable **only from `add_episode_bulk`**, not `add_episode` (`bulk_utils.py:271` default `False`). Capturing it means migrating the adapter to the bulk path — behavioral change (bulk skips cross-episode edge-invalidation/dedup), not a flag flip.
- **[WIN, free on upgrade] #1498** — strengthened extract prompts (attribute-hallucination guards, entity-precision) + 250-char attribute cap. NS gets these automatically on its existing `add_episode` path.

**Sagas** — **[NEUTRAL]** `SagaNode` gained summary fields + `summarize_saga()` + `summarize_sagas.py`. NS never creates sagas — purely additive, no action. *But this is the dominant merge-conflict zone* (§5).

**Attribute/group_id safety** — **[WIN, protective] #1498** — attribute write changed to guarded `if k not in entity_data` (`nodes.py:567`, `edges.py:362`), so a hallucinated `group_id` attribute can't overwrite the core field. NS not exposed today (passes `group_id` as first-class kwarg). Resolves prior open question.

**Search** — **[NEUTRAL]** `search.py` +677/−323 is **entirely OpenTelemetry tracing** (`NoOpTracer` default). `search_config.py`, `search_config_recipes.py`, `search_filters.py`, `search_helpers.py`, `search_utils.py` all **byte-identical**. No recipe/RRF/MMR/reranker algorithm changed.

**LLM providers** — **[NEUTRAL] #1551** default → `gpt-5.5`; **#1537** OpenAIGenericClient json_schema default. NS not exposed (explicitly sets gemini models, builds `OpenAIClient` directly). But #1537's json_schema mode is the one lever toward gateway-routing graphiti's LLM (§6).

---

## 4. Breaking changes & required NS edits

### 4.1 LEAD: Score-normalization (#5391) — **NOT a hazard for current NS; prior analysis §8.3 is wrong.**
Qdrant is **not in #5391's file list** (verified `git log v2.0.2..HEAD -- mem0/vector_stores/qdrant.py` → only #5380); Qdrant already returns cosine similarity ~[0,1] higher=better; NS already lives in the normalized regime (vendored `main.py` defaults `threshold=0.1`, rejects thresholds outside [0,1]). **Every NS score site is already correct and stays correct** (Qdrant only):

| `memory_service.py` | Code | Verdict |
|---|---|---|
| `:827` | `"score": getattr(hit,"score",None)` (raw `query_points`) | native similarity — safe |
| `:1290` | `deduped.sort(...score..., reverse=True)` | higher=better — safe |
| `:1362/1434` | `_GRAPH_ENRICH_THRESHOLD=0.7; if score<0.7: skip` | correct direction — safe |
| `:2809/2879` | dedup `0.95`; `if hit_score<threshold: continue` | safe |
| `:2923` | `merged.sort(...score..., reverse=True)` | safe |

**The real rule:** #5391 gates **backend-pluggability, not the sync.** If NS ever selects pgvector/milvus *while pinned to 2.0.2*, the `0.95`/`0.7` thresholds rank backwards on distance-based backends. Sync mem0 ≥2.0.7 **before** shipping any non-Qdrant backend; then thresholds work unchanged everywhere. **Downgrade from "blocking" to "verify-only / pluggability-enabler."**

### 4.2 `config.http_client` semantics flip (#5447) — *most likely real mem0 break.*
`LlmFactory` now reads `config.http_client_proxies`; `config.http_client` now returns a built `httpx.Client`, not the proxies dict. **Action:** audit `config.py` mem0 config-dict build for any `config.http_client` read expecting a dict.

### 4.3 Empty-query rejection (#5258) / invalid-filter 400 (#5482).
mem0 native `search("")` now raises. **Action:** audit NS recall paths that could pass `""` to mem0 native `search()`.

### 4.4 graphiti attribute-merge guard (#1498) — non-breaking for NS, confirmed safe.

### 4.5 Episode index 0-based (#1432) — not exposed (NS sends one episode per `add_episode`).

### 4.6 Model/field additions (graphiti) — additive, non-breaking. `add_episode` gained `saga`/`custom_extraction_instructions` (keyword-defaulted) — NS's call at `:320` is forward-compatible. `MAX_SUMMARY_CHARS` 500→1000 is cosmetic (verify wiki synthesizer has no 500 assumption).

### 4.7 graphiti patches that are NOT obsoleted — **keep all three** (corrects prior optimism):
- **small_model/gpt-4.1-nano** (`graphiti_memory.py:41-51`): HEAD still hard-defaults `DEFAULT_SMALL_MODEL='gpt-4.1-nano'` (`openai_base_client.py:35`; #1551 only changed the *main* default). Gateway can't provision nano. **Keep.**
- **Reranker `config=` fix** (`:99-104`): `OpenAIRerankerClient.__init__` at HEAD still takes `config`, not `api_key=`. **Keep.**
- **Embedder-on-AI-Studio** (config.py:306-321): graphiti's `create_batch` byte-identical vendored↔HEAD — still sends multi-input that Vertex rejects with 400, no per-item fallback. **Keep.** Upgrading unlocks **zero** of these three.

---

## 5. Local-patch conflict map (the `sync-upstream.sh` risk surface)

### mem0 (3 true patches + an orphaned layer)

| Patched file | What it does | Upstream touched in range? | Severity |
|---|---|---|---|
| `mem0/graphs/` + `memory/{graph,graphiti,apache_age,kuzu,memgraph}_memory.py` | **NS-restored graph layer** (deleted upstream #4805) | **Absent upstream** | **HIGH (orphan/divergence)** — the pluggable-graph vision leans on this code; permanently NS-forked |
| `memory/graphiti_memory.py` | 3 load-bearing patches (§4.7) + scope→group_id composition (fixes Graphiti cross-user leak) | n/a (absent upstream) | **HIGH** — sole mem0↔graphiti bridge |
| `utils/factory.py` | Adds `GraphStoreFactory` (incl. `"graphiti"`) | **Yes:** #5447/#4030/#5190 touch Llm/Embedder factories, **not** GraphStoreFactory | **LOW–MEDIUM** — likely auto-mergeable |
| `embeddings/gemini.py` | Adds batched `embed_batch()` | **Yes: #5609** adds upstream's own — convergent | **MEDIUM–HIGH** — take upstream's |
| `embeddings/base.py` | Tightens `embed_batch` signature | No | **LOW** |
| `embeddings/openai.py` | `embed_batch` honors `config.embedding_batch_size` (client-side chunk size, clamped 1–100) and falls back to per-item embeds when a batched call is rejected with a single-input error. Needed because the gateway's Vertex embedding endpoint accepts only ONE input per request — with `LLM_GATEWAY_ENABLED` the batched call 400'd and conversation extraction silently stored zero facts. NS wiring: `EMBEDDER_MAX_BATCH_SIZE` in service `config.py` (gateway embedder block defaults it to 1). | Upstream has the same `embed_batch` (MAX_BATCH=100 chunking) — re-apply this diff on sync | **MEDIUM** — small self-contained diff in one method + one helper |
| `configs/embeddings/base.py` | Adds `embedding_batch_size: Optional[int] = None` to `BaseEmbedderConfig` (consumed by `embeddings/openai.py` above) | No | **LOW** — additive kwarg |
| `pyproject.toml` | `[graphiti]` extra + `[tool.uv.sources]` editable | **Yes, heavily** (2.0.7 bump, `vector_stores`→`vector-stores` #4934, CVE bumps) | **HIGH** — near-certain conflict every sync; `[tool.uv.sources]` is load-bearing |

### graphiti — Saga/reference_time/extraction cluster (the dominant conflict)

| Patched file | What it does | Upstream in range? | Severity |
|---|---|---|---|
| `prompts/extract_nodes.py` | anti-hallucination + saga summary prompt | #1498/#1432/#1422 rewrote same | **CRITICAL** |
| `prompts/extract_edges.py` | distinct-entity rules | #1498/#1432 | **CRITICAL** |
| `utils/maintenance/node_operations.py` | dedup tunables | #1498/#1432/#1422/#1361 | **CRITICAL** |
| `utils/maintenance/edge_operations.py` | `reference_time=episode.valid_at` | #1498 same fn, diff signature | **CRITICAL** |
| `graphiti.py` | full Saga lifecycle | upstream added its own | **CRITICAL** |
| `nodes.py` | `SagaNode` 4 new fields | HEAD defines same fields | **CRITICAL** |
| `prompts/summarize_sagas.py` | NS saga prompt | upstream also added this file (diff body) | **CRITICAL (add/add)** |
| `edges.py` | `reference_time` field | HEAD defines `reference_time` (`:280,355,986,1001`) | **HIGH (convergent)** |
| `models/edges/edge_db_queries.py`, `models/nodes/node_db_queries.py` | reference_time / saga Cypher | HEAD equivalent | **HIGH** |
| `driver/graph_operations/graph_operations.py` | saga query stubs | upstream added own | **HIGH** |
| `utils/maintenance/dedup_helpers.py` | gate reorder | #1361 | **HIGH** |
| `llm_client/config.py` | adds `fallback_model` | #1551 same `__init__` | **MEDIUM–HIGH** |
| `prompts/dedupe_nodes.py` | schema `duplicate_name`→`duplicate_candidate_id:int` | #1361 reshaped same | **MEDIUM–HIGH (consumer coupling)** |
| `utils/text_utils.py` | `MAX_SUMMARY_CHARS` 500→1000 | 3 commits | **MEDIUM–HIGH** |
| `prompts/{snippets,summarize_nodes,dedupe_edges,lib}.py`, `community_operations.py`, `search/search.py` | rewrites / tracer | #1361 overlap | **MEDIUM** |
| `llm_client/gemini_client.py` | `fallback_model` retry | #1498 adjacent | **MEDIUM** |
| `llm_client/gliner2_client.py` | pure ruff/black reformat | 2 real commits | **LOW — take upstream** |
| `driver/neo4j_driver.py` | real fix: `kwargs.setdefault('database_')` | **0 commits** | **LOW — the one safe re-apply** |

**Biggest reframe:** NS and upstream built the *same* saga + reference_time feature in parallel with different code across ~10 files. **Recommendation: adopt upstream's saga/reference_time wholesale and delete NS's parallel implementation, rather than line-merging.** Same for the extraction/dedup prompt rewrites — quality-critical, need a human prompt-engineering decision per file, not `git merge`.

---

## 6. What upgrading unlocks for the pluggable-backend vision

**P1 (vector swap) — mem0 makes it EASIER.** Roster unchanged, but **#5391's [0,1] higher=better contract across all backends is the single most valuable item for the vision.** Today NS's higher=better merge is correct *only because* it's on Qdrant. Post-2.0.7 it's **portable** — pgvector/milvus/redis return the same [0,1] similarity, so NS's `0.95`/`0.7` thresholds work unchanged on every backend. **This is the real reason to sync mem0: it de-risks the vector swap, not the current deployment.** Residual P1 work unchanged: `config.py:403,407` hardcoded providers + ~9 raw `QdrantClient` leaks bypassing the factory.

**P2 (graph driver swap) — graphiti makes it EASIER, but only via FalkorDB-now-viable.** #1549's group_id RediSearch fix is a **prerequisite**: NS's hyphenated group_ids only match on FalkorDB from 0.29.2. Roster: Neo4j + FalkorDB (viable) + Neptune (unproven); Kuzu out. Caveat unchanged: the real FalkorDB tax is NS's hand-written Neo4j-dialect Cypher (`list_projects` STARTS WITH, `DETACH DELETE`, `wiki_synthesizer/graph_patcher.py`), which 0.29.2 doesn't address.

**Can NS finally route graphiti's LLM+embedder through the gateway, retiring patches? Mostly NO.**
- #5609 (mem0 embed_batch): mem0 vector-side only. Embedder-on-AI-Studio patch stays.
- #4030 (mem0 Vertex LLM): mem0 vector-side only.
- #1537/#1146 (graphiti OpenAIGenericClient json_schema): the **one** lever to flip `llm_gateway_graphiti_enabled=True` — but NS builds `OpenAIClient`, not `OpenAIGenericClient`, so it needs a **new opt-in patch** (`graphiti_memory.py:39` → `OpenAIGenericClient(structured_output_mode='json_schema')`), not a removal.

**Net: the sync removes zero existing graph-routing patches.** It de-risks the *vector* half (P1) and unblocks FalkorDB for the *graph* half (P2); graphiti LLM/embedder gateway routing stays forked.

---

## 7. Recommended upgrade plan

**Sync mem0 first, graphiti second** — mem0 has 3 small true patches + a clean orphan story; graphiti is a 20-file conflict cluster best handled by adopting-upstream-wholesale, which benefits from a stable mem0 underneath.

**Step 0 — Pre-flight audits (before any checkout):**
- Grep `config.py` for `http_client` reads expecting a dict (#5447).
- Grep NS recall paths for empty-string `search("")` against mem0 native (#5258).
- Snapshot the graph re-attach shim: `memory_service.py:300-341` + `main.py:78-80` reach into `GraphStoreFactory`, `_memory.graph.graphiti/_bridge`, `MemoryConfig`. #5102 touched `main.py search()` and #4030 touched `utils/factory.py` — re-verify the import path + shim shape survive 2.0.7.

**Step 1 — mem0 2.0.2 → 2.0.7:** re-apply the 3 true patches (`graphiti_memory.py` bridge, `GraphStoreFactory` in `factory.py`, `[tool.uv.sources]`). Resolve convergent `embeddings/gemini.py` (take upstream). No score-code changes (§4.1). Tests: full unit suite + integration `test_async_pipeline.py`; assert Qdrant scores stay [0,1] descending. **Confirm the v3-pipeline `Memory` internals the shim hooks still expose `self._memory.graph.graphiti`** (highest risk, §8.1).

**Step 2 — graphiti 0.28.2 → 0.29.2:** **adopt upstream's saga + reference_time + extraction/dedup prompt rewrites wholesale; delete NS's parallel implementations.** Re-apply only what has no upstream equivalent: `neo4j_driver.py` `database_` fix (clean), re-add `fallback_model` on top of #1551's defaults if still wanted. Keep all 3 `graphiti_memory.py` patches (§4.7). Verify `MAX_SUMMARY_CHARS` final value + importers. Tests: same suite; spot-check extraction quality (should improve from #1498); confirm group_id matching on Neo4j.

**Sequencing with prior P0–P4 roadmap:**
- Do the **custom entity/edge ontology wire-in NOW**, independent of either sync — project `schemas.py:79-158` into `add_episode(entity_types=, edge_types=, edge_type_map=)` at `graphiti_memory.py:320`. Highest-value/lowest-risk graph win; works on 0.28.2.
- Gate **`explain=True`** behind a flag (private-pool only): add at `memory_service.py:1197/1203/1216`, thread `score_details` through `_merge_results:2909` (sort on `score_details.semantic_score if present else score`). **Shared-pool rows (`:811` raw `query_points`) have no `score_details` — merge must tolerate `None`.** Requires ≥2.0.5.
- **Defer** combined extraction (needs graphiti sync **and** `add_episode`→`add_episode_bulk` migration).
- P1/P2 remain post-sync work, but the sync is now their **precondition** (score-portability for P1; FalkorDB group_id for P2).

**Do NOT adopt:** Kuzu (#1548 deprecated); mem0's Vertex LLM / embed_batch for the *graph* path (wrong code path); flipping `llm_gateway_graphiti_enabled=True` without first adding the `OpenAIGenericClient(json_schema)` patch.

---

## 8. Open questions / verify-before-sync TODOs

**CONFIRMED (2026-06-22): mem0's OSS graph layer was deleted upstream, and `graphiti_memory.py` is 100% NS-authored.** Re-verified against full upstream history, not just the two tags:
- **The deletion is real and dated.** mem0 OSS shipped a self-hostable graph layer through v1.x (`mem0/graphs/{configs,tools,utils}.py` + `neptune/*`, `memory/graph_memory.py` (Neo4j), `memory/memgraph_memory.py`). Commit **`a488e190` (PR #4805), 2026-04-14** — *"feat(oss): port v3 pipeline with hybrid search, entity extraction, and additive scoring"* — removed all of it. `find mem0/ -iname '*graph*'` returns **nothing** at both 2.0.2 and HEAD 2.0.7. OSS replacement is an internal entity store (`utils/entity_extraction.py`, `ENTITY_BOOST_WEIGHT`, additive scoring in `main.py`), not a graph DB. The marketed "Graph Memory" is now a hosted-**Platform**-only feature (`docs/platform/features/graph-memory.mdx`: *"no external graph database to provision… no Neo4j, Memgraph, or other graph store to deploy"*).
- **`graphiti_memory.py` never existed upstream.** No add commit on any branch; `git log -S 'graphiti_memory'` across `--all` returns nothing. It is 444 lines of NS-original code. By contrast `graph_memory.py` / `memgraph_memory.py` / `kuzu_memory.py` / `neptune/*` are upstream v1.x files NS *restored* after the deletion (graph_memory.py ≈ identical to v1.0.10) — vestigial; only `"graphiti"` is wired in production.
- **Lineage of the vendored `mem0/`:** a hybrid — mem0 2.0.2 *core* (per `pyproject.toml`) + resurrected v1.x graph scaffolding + the NS Graphiti adapter. This is the real explanation for the stale "mem0 = v1.0.10" project memory: the *graph code* is v1.x, the *core* is 2.0.2.

**Implication:** NS is the sole maintainer of a graph layer upstream deliberately amputated. This is not a sync conflict (the files won't textually clash — they're simply absent upstream); it is permanent divergence that grows every release as the v3 pipeline evolves under the shim. The strategic option this surfaces: since NS already reaches *past* mem0 into Graphiti's `_bridge` directly, consider retiring the "impersonate a mem0 graph provider" indirection entirely and calling Graphiti through an NS-owned `GraphPort` from `MemoryService` — removing the most fragile coupling (the re-attach shim) instead of perpetually re-grafting it. Fold this into the P0/P2 decision (`13-pluggable-backend-gap-analysis.md`).

1. **Graph re-attach shim survival (highest *operational* risk).** Does `memory_service.py:300-341` reaching into 2.0.7's v3-pipeline `Memory` (post-#4805 graph-layer removal, commit `a488e190`) still find `self._memory.graph.graphiti`? The orphaned layer won't conflict textually, but the surrounding mem0 internals it fabricates a `.graph` against changed. **Must be exercised against a real 2.0.7 checkout before committing the sync.**
2. **`config.http_client` consumers (#5447).**
3. **Empty-query paths (#5258).**
4. **`MAX_SUMMARY_CHARS` final value** after graphiti merge + wiki-synthesizer downstream.
5. **`dedupe_nodes.py` schema coupling** — if adopting upstream's prompt, confirm `node_operations.py`/`dedup_helpers.py` consume the same `duplicate_candidate_id:int` (-1 sentinel) shape.
6. **`limit`→`top_k` rename** at `memory_service.py:2862/1877` — confirm stable at 2.0.7.
7. **`add_episode` forward-compat** — confirm new `saga`/`custom_extraction_instructions` defaults don't change NS's single-episode behavior.

**Three corrections to the prior gap analysis (`13-pluggable-backend-gap-analysis.md`):** (a) §8.3 score-normalization is *not* a blocking gate on the sync — Qdrant is untouched and NS already sorts descending; it's a pluggability *enabler*. (b) The explain field is `score_details` with no `temporal_boost` key (not `score_breakdown`). (c) The graph-embedder batching blocker is graphiti-side and **not** resolved by mem0's `embed_batch`; all three `graphiti_memory.py` patches remain required.
