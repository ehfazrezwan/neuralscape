# Engine-Maximization Implementation Spec (Handoff)

*Status: ready for an implementing agent. Created 2026-06-22. Verified against vendored **mem0 2.0.2** and **graphiti 0.28.2** (the trees NS ships) — every change below works on the vendored versions, **no upstream sync required**. Builds on docs 15 (parity), 16 (pluggable design — separate effort), 17 (rewire feasibility).*

## 0. How to use this document

This is a **task list for an implementing agent**. Each work item (M1–M3 for mem0, G1–G4 for graphiti) is self-contained: goal, exact file:line targets, code sketches, constraints/gotchas, acceptance checks, effort, risk, dependencies. Implement in the order given in §6 (sequencing). **Do not** attempt the items in §7 (Out of scope) — they are structurally impossible or empirically rejected and are documented so you don't rediscover them.

**Repo:** `/Users/ehfaz.rezwan/Projects/neuralscape`. Service: `neuralscape-service/`. mem0 subtree: `mem0/mem0/`. graphiti subtree: `graphiti/graphiti_core/`. NS-authored adapter: `mem0/mem0/memory/graphiti_memory.py`.

**Workflow rules (from CLAUDE.md):** never commit to `dev`/`main`; branch `feature/engine-maximization` off `dev`. Run `uv run pytest tests/ --ignore=tests/test_async_pipeline.py -v` after each item. Integration tests (`test_async_pipeline.py`) need Neo4j+Redis+Qdrant.

## 1. Goal & guiding principle

**Retain 100% of Neuralscape's current functionality**, and for every operation where mem0 or graphiti can do it natively *without losing an NS feature*, route through (or activate) the native pathway. This is the pragmatic middle path: not "rewire everything" (doc 17 proved much of that impossible), not the full pluggable abstraction (doc 16, a separate program). The wins here are **retrieval quality** (mem0 hybrid BM25 + entity boost + rerank; graphiti cross-encoder + typed graph) and **correctness/maintainability** (native delete cascade; deleting raw Cypher).

## 2. What is irreducibly NS's — DO NOT remove or "nativize" these

These were investigated (doc 17) and confirmed as structural gaps in mem0/graphiti or NS's defining value. **Leave them exactly as they are:**

- **NS's own LLM extraction + 13-category taxonomy** (`memory_service.py:434-491`, `prompts.py`, `schemas.py:79-183`). mem0 self-hosted has **no** category system (2.0.7 literally `raise`s "not supported in OSS"), and its inferred-add parser reads only `{text, attributed_to}`, dropping any per-fact category. Keep NS extraction.
- **The shared cross-writer pool** (raw `query_points`, `memory_service.py:811,1420`). mem0's read API mandates single-`user_id` namespacing; there is no writer-agnostic mode. Keep raw Qdrant here.
- **Full-collection pagination / sweeps** (raw `scroll`, `:857,968,1225,2225,2725,2780`). mem0 has no pagination cursor. Keep raw scroll for dedup cron / expiry / enumeration.
- **is-null / field-missing filter** (`:2210`). No such operator in mem0's DSL. Keep raw Qdrant.
- **`list_projects` prefix scan** (raw Cypher, `:1930`). graphiti's ORM does exact `IN` match returning hydrated nodes; NS needs `STARTS WITH` + `DISTINCT group_id`. Keep raw Cypher.
- **The scope/group_id algebra, visibility/owner model, project-id inference, junk filter.** All NS policy. Keep.

## 3. mem0 work items

> **Key correction to doc 15 carried here:** NS's *personal pool already reads via `m.search()`* (`memory_service.py:1197-1221`), which runs mem0's full hybrid pipeline. The gap is that **NS-written rows lack the fields that pipeline scores on**, so hybrid/entity/rerank silently no-op on NS data. M1–M3 close that.

### M1 — Populate `text_lemmatized` on writes so BM25 works on NS rows  *(effort: M, risk: low, highest value)*

**Goal:** NS-written memories participate in mem0's BM25 keyword leg on read, instead of being vector-only.

**How mem0 does it (reference, read-only):**
- `lemmatize_for_bm25(text) -> str` — `mem0/utils/lemmatization.py:22-50` (pure spaCy, no LLM/IO; falls back to input if spaCy missing).
- mem0's writer sets `payload["text_lemmatized"]` at **top level** (`main.py:810`, `:1599`).
- The Qdrant adapter reads top-level `payload.get("text_lemmatized")` at insert time and builds the `bm25` sparse vector (`vector_stores/qdrant.py:194-203`, encoder `_encode_bm25` `:103-118`).
- On read, `m.search` lemmatizes the query with the **same** function (`main.py:1349`) and runs `keyword_search` (`:1362`) → normalized + fused in `score_and_rank` (`utils/scoring.py:60-121`). **So once NS rows carry the sparse vector, NS's existing reads use BM25 with zero read-side changes.**

**Changes:**
1. `memory_service.py` top: `from mem0.utils.lemmatization import lemmatize_for_bm25`.
2. `_batch_store_facts` payload build (~`:1069-1099`): add **top-level** key (sibling of `data`/`hash`, NOT inside the nested `"metadata"` dict): `"text_lemmatized": lemmatize_for_bm25(content)`. For batch, precompute `texts_lemmatized = [lemmatize_for_bm25(t) for t in texts]` once before the loop (spaCy is CPU-bound). The single batch `m.vector_store.insert(...)` at `:1095-1099` is unchanged.
3. `store_raw` payload (~`:690-704`): same one-line top-level addition. `insert` unchanged.

**CRITICAL gotcha — the BM25 collection slot:** mem0 only attaches BM25 vectors if the Qdrant collection was **created with the `bm25` named sparse-vector slot**. For a pre-existing collection it sets `_has_bm25_slot=False` and silently disables BM25 (`qdrant.py:130-144`). NS's `neuralscape_memories` (`config.py:66`) predates v3 and **must be recreated** — you cannot add a sparse slot in place. See M1-backfill.

**M1-backfill (required, run once in NS venv):**
1. Verify current state: `client.get_collection("neuralscape_memories").config.params.sparse_vectors` — if it lacks `"bm25"`, migration is needed.
2. Scroll all points (payloads + **dense vectors**, `with_vectors=True`).
3. Create a fresh slotted collection (new name or delete+recreate so `create_col` provisions the `bm25` slot). Keep a Qdrant snapshot first.
4. For each row: set `payload["text_lemmatized"] = lemmatize_for_bm25(payload["data"])` (preserve every other key incl. nested `metadata`), then `m.vector_store.insert(vectors=[stored_dense], ids=[id], payloads=[payload])` — reuse the stored dense vector, **no re-embedding**. The adapter regenerates the sparse vector from `text_lemmatized`.
   - Note: `set_payload` does **not** refresh the sparse vector (`qdrant.py:480-483`) — you must full-`insert`.

**Acceptance:**
- `m.vector_store._has_bm25_slot is True`; `get_collection(...).config.params.sparse_vectors` contains `"bm25"`.
- After an NS write, `m.vector_store.get(<mid>).payload["text_lemmatized"]` is non-empty; `data` unchanged; retrieving with `with_vectors=True` shows a `bm25` sparse vector.
- A keyword-only query (rare term present verbatim in a fact but semantically distant) now surfaces that NS fact where it didn't before.
- Existing unit tests pass; NS `metadata` sub-dict byte-identical (only top-level `text_lemmatized` added).

### M2 — Populate mem0's entity store from NS's writer  *(effort: M, risk: low, fast-follow to M1)*

**Goal:** NS rows get entity-boost on personal-pool reads. Pure spaCy, no LLM coupling.

**Reference:** write helpers `extract_entities_batch(texts)` (`utils/entity_extraction.py:147-174`) and `_upsert_entity(entity_text, entity_type, memory_id, filters)` (`main.py:413-454`). Read-side boost `_compute_entity_boosts` (`main.py:1440-1499`) requires entity rows scoped by the same `user_id` NS already searches with, whose `linked_memory_ids` contain NS UUIDs. Entity collection is `f"{collection}_entities"`, lazily created.

**Change (batch path, after the existing `insert`+history loop in `_batch_store_facts`):**
```python
from mem0.utils.entity_extraction import extract_entities_batch
entity_filters = {"user_id": user_id}   # must match read-side scope
try:
    for mid, ents in zip(memory_ids, extract_entities_batch(texts)):
        for entity_type, entity_text in ents:
            m._upsert_entity(entity_text, entity_type, mid, entity_filters)
except Exception as e:
    logger.warning(f"Entity store population failed (non-critical): {e}")
```
`store_raw`: same with `extract_entities(content)` over the single fact.

**Perf note:** `_upsert_entity` does 1 embed + 1 search + 1 upsert per entity (N round-trips). For batch, prefer porting mem0's Phase-7 batched logic (`main.py:866-953`: one `embed_batch` for unique entities + `entity_store.search_batch` + one batch insert). Gate behind an env flag (e.g. `NS_ENTITY_STORE_ENABLED`).

**Scope caveat (acceptable, not a regression):** entity boost only benefits the **personal pool** (`m.search`). The shared pool bypasses `m.search` so it won't get boosts — that's fine, it never did.

**Acceptance:** `m.entity_store.list(filters={"user_id": uid}, top_k=...)` returns rows whose `linked_memory_ids` include NS UUIDs; a query with a stored proper noun boosts the linked NS fact's rank.

### M3 — Enable mem0's reranker on personal-pool reads  *(effort: S, risk: low, opt-in)*

**Goal:** cross-encoder/LLM rerank over personal-pool hybrid results. Fully supported on 2.0.2.

**Reference:** `Memory.search(..., rerank=True)` applies only when `rerank=True` **and** `self.reranker` is configured (`main.py:1230-1233`); reranker built from `config.reranker` at init (`:349-355`); `RerankerFactory` providers (`utils/factory.py:238+`): `cohere`, `sentence_transformer`, `zero_entropy`, `llm_reranker`, `huggingface`. `sentence-transformers>=5.0.0` already a dep (`mem0/pyproject.toml:73`).

**Changes:**
1. `config.py` (near the mem0 config `vector_store` block ~`:402`): add, gated behind `NS_RERANK_ENABLED`:
```python
config["reranker"] = {"provider": "sentence_transformer",
                      "config": {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}}
# alt: {"provider": "llm_reranker", "config": {"llm": {"provider": "gemini", "config": {...}}}}
```
2. Personal-pool `m.search` calls (`memory_service.py:1197,1203,1216`): add `rerank=True`.

**Caveats:** affects personal pool only (shared pool would need a manual `m.reranker.rerank(...)`). `sentence_transformer` loads a model into the worker (memory + cold start) — keep opt-in. Land **after** M1/M2.

**Acceptance:** `m.reranker is not None`; `m.search(..., rerank=True)` returns results with a `rerank_score` and reorders vs `rerank=False`.

## 4. graphiti work items

### G1 — Typed ontology (`entity_types`/`edge_types`/`edge_type_map`)  *(effort: M–L, risk: med)*

**Goal:** project NS's taxonomy into graphiti so the graph is typed instead of all-generic-`Entity`. graphiti's flagship feature, currently unused (`graphiti_memory.py:320-328` passes none).

**API & hard constraint:** `add_episode(entity_types, edge_types, edge_type_map)` (`graphiti.py:916-934`); `validate_entity_types` (`utils/ontology_utils/entity_types_utils.py:23-37`) **rejects any entity-type field name colliding with `EntityNode.model_fields`** = `{uuid, name, group_id, labels, created_at, name_embedding, summary, attributes}`. Edge attrs likewise must avoid `EntityEdge` reserved fields (`{uuid, group_id, source_node_uuid, target_node_uuid, created_at, name, fact, fact_embedding, episodes, expired_at, valid_at, invalid_at, reference_time, attributes}`) — e.g. use `decided_at`, never `valid_at`. `edge_type_map` default when omitted is `{('Entity','Entity'): list(edge_types)}`.

**Change:**
1. New module `neuralscape-service/graph_ontology.py` defining Pydantic entity/edge models + `ENTITY_TYPES`, `EDGE_TYPES`, `EDGE_TYPE_MAP`. Starter ontology (refine against `schemas.py` categories): entities `Preference, TechStack, Convention, ArchitectureComponent, Dependency, Decision, Person`; edges `Prefers, Uses, DependsOn, Decided`; map e.g. `("Person","TechStack"):["Uses"]`, `("ArchitectureComponent","Dependency"):["DependsOn"]`, plus `("Entity","Entity"):[]` catch-all. Keep attributes few (each is an LLM-populated Neo4j prop). *(Full sketch in the research transcript; reproduce and tune.)*
2. Wire through config (preferred — **no subtree edit**): in `graphiti_memory.py __init__` after ~`:207` read `self._entity_types = getattr(graph_config, "graphiti_entity_types", None)` (+ edge_types, edge_type_map). Pass them in the `add_episode(...)` call at `:320-328`. NS `config.py` populates those config fields by importing from `graph_ontology`.

**Heterogeneous-graph migration:** none required. Old untyped nodes keep only `:Entity`; new nodes get extra labels (`:Preference:Entity`) + typed attrs. Search matches `:Entity`/`:Episodic`, so retrieval is unaffected. Dedup is **self-healing**: when a new typed extraction resolves to an existing untyped node (`resolve_extracted_nodes`, `graphiti.py:1067`), graphiti merges labels/attrs onto it. Do **not** write a backfill script.

**Acceptance:** unit test asserts no entity model field collides with the reserved set + `validate_entity_types(ENTITY_TYPES) is True`; integration: ingest "I prefer tabs; the project uses Postgres 16" → a node carries label `Preference`, another `TechStack`, and `TechStack.attributes["version"]=="16"` round-trips via `EntityNode.get_by_uuid`; re-ingesting a pre-existing untyped "Postgres" node gains the `TechStack` label (no duplicate).

### G2 — Combined cross-encoder search recipe  *(effort: S, risk: med — benchmark first)*

**Goal:** activate the cross-encoder (already built at `graphiti_memory.py:176-195`, never invoked) and populate the `nodes/episodes/communities` result layers that are empty today.

**Change:** swap `EDGE_HYBRID_SEARCH_RRF` → `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` (`from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_CROSS_ENCODER`, defined `search_config_recipes.py:81-108`) at the **search** sites: `memory_service.py:~1551` (`_do_graph_search`, both the `else` and exception-fallback assignments) and `:~1689` (`search_graph`), and `main.py:~552`.

**MUST-FIX while here (pre-existing singleton-mutation race):** recipes are module-level singletons and NS does `config.limit = limit`. Deep-copy before mutating:
```python
config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
config.limit = limit
```
**Leave `_expire_graph_edges_for_memory` (`:~2606`) on `EDGE_HYBRID_SEARCH_RRF`** — invalidation only needs edge matches; cross-encoder there is wasted latency.

**Result handling needs no change:** `_do_graph_search` already maps `results.{edges,nodes,episodes,communities}` (`memory_service.py:1567-1581`) — they just populate now. (BFS legs in the combined recipe stay inert without a `center_node_uuid`; harmless.)

**Benchmark before committing (no prior NS measurement exists):** add `tests/bench_search_recipes.py` comparing RRF vs cross-encoder on a hand-labeled ~20–30 query set: report **MRR**, **recall@10**, **p50/p95 latency** (harness sketch in research transcript). Expectation: better MRR, higher p95. If latency regresses unacceptably, config-gate the recipe behind a flag rather than hardcoding.

**Acceptance:** graph search returns non-empty `nodes`/`episodes` where it returned `[]`; concurrent searches with different `limit` don't bleed (proves the `model_copy` fix); MRR(xenc) ≥ MRR(rrf) on the labeled set with recorded p95 delta.

### G3 — `remove_episode` instead of raw `DETACH DELETE`  *(effort: S, risk: low — strict win)*

**Goal:** correct cascade on episode delete. NS's raw `MATCH (e:Episodic {uuid}) DETACH DELETE e` (`memory_service.py:~2390`) leaves orphaned entity nodes/edges and already caused 2 bugs (#23).

**Change:** replace with `self._run_on_bridge(g.remove_episode(episode_uuid))`. Signature `remove_episode(episode_uuid: str)` (`graphiti.py:1694-1722`): deletes only edges whose **creating** episode is this one (`episodes[0]==uuid`) and entity nodes mentioned **solely** by this episode (`episode_count==1`) — shared entities survive.

**Error parity:** `remove_episode` raises `NodeNotFoundError` (graphiti.errors) where the old Cypher silently no-op'd. Wrap to preserve NS's return contract:
```python
from graphiti_core.errors import NodeNotFoundError
try:
    self._run_on_bridge(g.remove_episode(episode_uuid))
    return {"message": f"Episode {episode_uuid} deleted"}
except NodeNotFoundError:
    return {"message": f"Episode {episode_uuid} not found (already deleted)"}
except Exception as e:
    logger.error(...); return {"error": str(e)}
```
Audit the `delete_junk_episodes` loop (~`:2411-2485`) — the not-found catch preserves its tolerate-missing behavior.

**Composes with NS soft-expiry:** `_expire_graph_edges_for_memory` sets `expired_at` (soft); `remove_episode` hard-deletes by `episodes[0]`; the two are independent and `expired_at` is untouched by removal. No expiry change needed.

**Acceptance:** removing an episode that created a unique entity+edge deletes them (old Cypher fails this); a shared entity ("Postgres") mentioned by two episodes survives removing one; bogus UUID returns the not-found message, not a 500.

### G4 — Wiki back-refs via `EntityNode.attributes`; delete the enrich Cypher  *(effort: M, risk: med)*

**Goal:** remove the raw-Cypher read round-trip (`_enrich_graph_results`, `memory_service.py:~1596-1649`). NS's custom props (`memory_id`, `wiki_path`, `ns_visibility`, `ns_owner`, `wiki_synthesized_at`) are written as top-level Neo4j props and **already round-trip through `EntityNode.attributes`** (return query `properties(n) AS attributes`, `node_db_queries.py:283-291`; `get_entity_node_from_record` pops only the 7 reserved keys, `nodes.py:1033-1062`). So the read-side Cypher re-fetches data graphiti already loaded.

**Change (read side — the primary win):** in `_do_graph_search` node mapping (`memory_service.py:1571-1581`), read attrs off the ORM object:
```python
nodes = [{"uuid": n.uuid, "name": n.name, "summary": n.summary,
          "memory_id": n.attributes.get("memory_id"),
          "wiki_path": n.attributes.get("wiki_path"),
          "ns_visibility": n.attributes.get("ns_visibility")}
         for n in results.nodes]
```
Then **delete `_enrich_graph_results` and its call (~`:1585`)**. Synergistic with G2: pre-G2 `results.nodes` was empty; post-G2 nodes populate and these attrs ride along free.

**Write side — what can/can't move to ORM:**
- **`patch_wiki_path` (UUID-keyed)**: CAN move. `add_episode` returns `result.nodes` (full `EntityNode`s, `graphiti.py:1151-1155`). Stamp `node.attributes["wiki_path"]=...` then `await node.save(driver)` (`nodes.py:539-573`).
- **`attach_memory_id` (120s time-window)**: CANNOT move as-is — the adapter's `add` discards UUIDs (`graphiti_memory.py:330-347` returns names only), and its `coalesce` first-writer-wins semantics have no ORM equivalent. *Optional best-fix:* make `MemoryGraph.add` also return `node_uuids=[n.uuid for n in result.nodes]`, then NS stamps `memory_id` by UUID via the ORM, deleting the time-window heuristic and its race. If you don't want to touch the subtree return shape, leave `attach_memory_id` as raw Cypher (acceptable).
- **`patch_wiki_path_by_memory_ids`**: CANNOT move cleanly — no `get_by_attribute` in 0.28.2; stays raw Cypher.

**FOOTGUN — embedding loss on ORM re-save:** `EntityNode.save` does `SET n = $entity_data` (rewrites the whole node incl. `name_embedding`). A node loaded via `get_by_uuids` has `name_embedding=None` (return query omits it). **Before any `.save()` of a loaded node, call `await node.load_name_embedding(driver)`** (`nodes.py:510-537`) or you'll null the embedding. The time-window Cypher avoids this by `SET`ting only specific props.

**Constraints:** keep `wiki_synthesized_at` an **ISO string** (matches commit `8f9fde5` #68) — native datetime in `.attributes` comes back as a different type. Communities have no `attributes` field — don't try to stamp them (no patcher does).

**Acceptance:** after deleting `_enrich_graph_results`, graph search still returns `memory_id`/`wiki_path` on nodes that have them; an ORM stamp+save followed by `get_by_uuid` shows the prop **and** a still-populated `name_embedding` (proves the guard); `wiki_synthesized_at` stays an ISO string.

## 5. Effort & risk summary

| Item | What | Effort | Risk | Native capability gained |
|---|---|---|---|---|
| **M1** | `text_lemmatized` on writes + collection backfill | M (~2–4d) | low | mem0 BM25 hybrid on NS rows |
| **M2** | entity-store population | M (~2–4d) | low | mem0 entity-boost (personal pool) |
| **M3** | reranker on reads (opt-in) | S (~½d) | low | mem0 rerank |
| **G1** | typed ontology | M–L (~1wk) | med | graphiti typed graph |
| **G2** | cross-encoder combined recipe + singleton fix + benchmark | S (~1d) | med | graphiti cross-encoder + multi-layer results |
| **G3** | `remove_episode` | S (~½d) | low | correct delete cascade |
| **G4** | ORM attributes; delete enrich Cypher | M (~2–4d) | med | less raw Cypher; ORM round-trip |

Total ≈ **3–4 weeks** for all seven. The high-value/low-risk core (M1, G2, G3) is ≈ 1 week.

## 6. Sequencing

1. **G3** (isolated, strict win, warm-up).
2. **M1 + M1-backfill** (highest value; coordinate the collection recreation with a maintenance window).
3. **M2** (depends on M1's collection; fast-follow).
4. **G2** (benchmark first; also a prerequisite for G4 since it populates `results.nodes`).
5. **G4** (read-side after G2; decide on the optional adapter `node_uuids` change).
6. **M3** (opt-in, after M1/M2 so rerank operates on the enriched result set).
7. **G1** (largest; can proceed in parallel after the ontology module is designed — independent of M-items).

Each item: branch work, run unit suite, then integration suite for graph/vector items.

## 7. Out of scope — do NOT attempt (documented so you don't rediscover)

- **Routing writes through `Memory.add`** — flattens metadata to top-level, breaking NS's `metadata.*` read filters; requires re-keying all reads + reindex + a perf regression. Augment NS's writer (M1/M2) instead.
- **Retiring NS extraction for mem0's** — no OSS categories; parser drops per-fact fields (§2).
- **Shared pool / full-sweep / is-null via native API** — mem0 structural gaps (§2).
- **`list_projects` via ORM** — needs prefix + DISTINCT; ORM can't (§2).
- **`build_communities`** — NS benchmarked it "an order of magnitude too slow" (`synthesizer.py:17-21`); keep category-bucketing.
- **`add_episode_bulk`** — skips edge invalidation + date extraction (its own docstring `graphiti.py:1226`); breaks NS temporal correctness.

## 8. Global no-regression checklist (run after all items)

- `uv run pytest tests/ --ignore=tests/test_async_pipeline.py -v` green.
- Integration `test_async_pipeline.py` green against live Neo4j/Redis/Qdrant.
- NS payload `metadata` sub-dict unchanged except intended additions; 13-category taxonomy, scope algebra, visibility, project inference all behave identically.
- Shared-pool reads still return rows (untouched).
- A full write→poll→recall round-trip via REST and MCP returns expected memories.
- `./scripts/sync-upstream.sh` still applies cleanly (G1 used config-wiring, not subtree edits; if the optional G4 `node_uuids` adapter change was taken, note it as a new tracked local patch in doc 14 §5).
