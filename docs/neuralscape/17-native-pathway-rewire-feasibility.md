# Decoupling Neuralscape onto the Engines' "Best Pathways": Feasibility & Effort

*Investigation 2026-06-22, with `git blame`/`git log` on every bypass site to recover original rationale. Question asked: "If we rewired all of NS to go through the best-performing pathways mem0 and graphiti provide, what would that look like and how much work?" Sources: vendored mem0 2.0.2, graphiti 0.28.2; upstream mem0 2.0.7, graphiti 0.29.2; NS service git history.*

---

## TL;DR — the reframe

**`memory_service.py`'s bypasses are mostly deliberate or structurally forced, not accidental debt.** When we traced each one to its origin commit, ~70% turned out to be NS routing *around a real, named limitation* of mem0/graphiti — not a shortcut. So "rewire everything through the native best pathways" is the wrong target: doing it fully would mean either **re-forking the engines** or **deleting NS's defining features** (the shared team pool, the 13-category taxonomy, multi-user visibility, full-collection dedup). 

The honest answer splits into three buckets:

| Bucket | Count | What to do |
|---|---|---|
| **Structurally impossible** to route natively (engine can't express it) | ~5 | Keep the bypass; it's load-bearing |
| **Deliberately/empirically rejected** native pathway | 2 | Leave alone (already measured worse) |
| **Genuine opportunity** — native pathway is better or equal | ~4 | Worth doing; scoped below |

And one correction to doc 15 that changes the premise:

> **Doc 15 overstated "NS reuses nothing."** On **reads**, NS's *personal pool already goes through `m.search()`** (`memory_service.py:1197-1221`) — i.e. it already gets mem0's hybrid BM25 + vector + entity-boost + optional rerank. The reuse gap is almost entirely on the **write** path and the **shared** pool, not reads.

---

## 1. Vector side (mem0)

### Structurally impossible to route through the public API — KEEP these bypasses
- **Shared cross-writer pool** (`memory_service.py:811`, `:1420`). Origin commit `f86766a` ("multi-user support") chose raw `query_points` deliberately: *"bypasses mem0's user_id namespacing because shared memories span multiple writers."* Verified blocker: every mem0 read entry point (`search`, `get_all`) **raises unless `filters` contains user_id/agent_id/run_id** (`main.py:1193-1197`) and compiles it into a Qdrant `must` AND (`qdrant.py:361-363`). mem0's entire model is single-entity namespaced retrieval; there is **no writer-agnostic mode**. NS's team-shared pool is inexpressible in mem0's filter DSL. **This is the deepest, most load-bearing bypass.**
- **Full-collection pagination / sweep** (`:857,968,1225,2225,2725,2780`). `get_all`→`list` issues **one bounded `scroll` with no offset cursor** (`main.py:1079`, `qdrant.py:531`). The dedup cron, `expire_old_memories`, and user-id enumeration need *every* point. mem0 cannot deliver that. Raw scroll is required.
- **is-null / field-missing filter** (`:2210`, `_list_null_category_memories`). mem0's DSL has `eq/ne/gt/in/contains/AND/OR/NOT` but **no is-null operator**; its `"*"` wildcard means match-*all*, the opposite. Not expressible.

### Genuine opportunities — WORTH doing
- **Write path doesn't populate mem0's hybrid index (the real prize).** NS's batch writer (`:1069-1084`) and `store_raw` (`:690-698`) write neither `text_lemmatized` (BM25) nor entity-store entries. So even though reads call `m.search()`, **NS-written rows are degraded for hybrid/entity-boosted retrieval** — they only match on the vector leg. **Highest value-to-effort fix in the whole report:** enrich the existing batch writer to call mem0's `lemmatize_for_bm25` + populate the entity store, *keeping* the single-batch upsert. Captures mem0's best retrieval pathway without surrendering the taxonomy or reintroducing per-fact LLM calls. **Effort: M.**
- **Per-id delete → `Memory.delete(id)`** (`:2746,2094,2179`). Drop-in; mem0's `delete` also records history + graph cleanup. Keep NS's own edge-expiry as a wrapper. **Effort: S.**
- **Out-of-band `db.add_history`** (`:707,1104`). Pure compensation for the direct insert; mem0's `_create_memory` records history automatically (`main.py:1606`). Free to drop once writes route through the writer. **Effort: S.**

### The trap — do NOT naively "go through `Memory.add`"
Routing writes through `Memory.add(infer=False, metadata=…)` looks clean but `_create_memory` **flattens metadata to the top level** of the payload (`main.py:1593-1599`), while NS reads filter on nested `metadata.category`/`metadata.visibility` keys *everywhere*. Adopting `add` for writes forces re-keying **every NS read filter + reindexing the existing collection** — and still loses the batch-embed perf win (`add(infer=False)` embeds per-message, `main.py:685`). This is why the recommended write fix *augments* NS's batch writer rather than replacing it with `add`. **Effort if done naively: L + regression risk.**

---

## 2. Graph side (graphiti)

### Genuine opportunities — WORTH doing
- **Episode delete → `remove_episode`** (`:2390`). NS's raw `DETACH DELETE e` was already the source of two bugs (#23: wrong label, `parameters_` conflict) and is *more aggressive* (orphans entities). `Graphiti.remove_episode` (`graphiti.py:1765`) cascades correctly. **Strict improvement. Effort: S.**
- **Activate the cross-encoder via `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`** (`:1551,1689`; `main.py:552`). The cross-encoder client is *already built and attached* (`graphiti_memory.py:176`) but the hardwired edge-only RRF recipe never invokes it. Switching the recipe is the entire change and also populates the `nodes/episodes/communities` result layers that are currently empty. **BUT (Q1 verdict): no NS benchmark ever compared RRF vs cross-encoder** — the edge-only recipe was the initial default (`136a339`) carried forward, not a measured decision. Low-risk to try; quality benefit unproven; known cost is extra rerank calls/latency per query. **Effort: S; validate before committing.**
- **Typed ontology (`entity_types`/`edge_types` on `add_episode`)** (`graphiti_memory.py:320-328`). Fully supported in both versions; never attempted. The blocker is *design effort* (map NS's 13 categories → a graph ontology) + a heterogeneous-graph period until re-ingested, not API support. **Effort: M–L.**

### Structurally forced — KEEP raw Cypher
- **`list_projects`** (`:1930`). Needs a `STARTS WITH` prefix scan returning *only* `DISTINCT group_id` across `:Entity/:Episodic/:Community`. graphiti's `EntityNode.get_by_group_ids` does exact `IN` matching and returns **fully hydrated nodes** (`nodes.py:415`) — it cannot express prefix predicates or projection-only returns. Raw Cypher is required.

### Deliberately/empirically rejected — DO NOT revisit
- **Community rebuild.** The one bypass NS actually benchmarked. `synthesizer.py:17-21` (verbatim): *"the LLM tournament summarization there was an order of magnitude too slow to run on real data, so we replaced it with category-based bucketing."* Pivot: commit `53a858c`. Settled.
- **`add_episode_bulk`.** Its own docstring (`graphiti.py:1226`): *"does **not** perform edge invalidation or date extraction."* NS depends on temporal edge invalidation (`0225453`). Bulk would silently break temporal correctness. Off the table.

---

## 3. Extraction & taxonomy — irreducibly NS's

This is the make-or-break for "go native," and the verdict is **clear: NS's extraction cannot be retired.**

- **mem0 self-hosted has no category system** — README's original rationale (`README.md:679`) verified true. 2.0.2 has nothing; **2.0.7 adds `Memory.project.update(custom_categories=…)` that explicitly `raise`s `"not supported in OSS"`** (`main.py:376,390`). Categories are platform/hosted-only.
- **Deeper structural blocker:** mem0 2.x's inferred-add pipeline parses LLM output to `{text, attributed_to}` only and copies caller `metadata` *uniformly onto every fact* (`main.py:795,808-816`). There is **no code path that reads a per-fact `category`/`scope`/`domain` from the LLM JSON** — any structured field the model emits is silently dropped. So `Memory.add(infer=True, prompt=<NS prompt>)` (option a) would mean *re-forking the very pipeline you're trying to adopt*. Net negative.

**Irreducibly NS (must stay in the service layer regardless):** the 13-category taxonomy + per-fact assignment (`schemas.py:79-122`, `prompts.py`), scope algebra + project-id inference (`:1047-1066`), the v2 vocabularies (`schemas.py:165-183`), multi-user visibility/owner + `group_id` encoding (`:206-236`), and the junk/anti-hallucination filter (`:77-82`).

---

## 4. The root cause of the Neo4j hard-binding: the wiki-synthesizer

A standout structural finding. **Almost every raw-Cypher / NS-custom-property site exists to serve the wiki-synthesizer's back-reference feature**, not graphiti's design:
- `memory_id`, `ns_visibility`, `ns_owner` (write-time `attach_memory_id`) and `wiki_path`, `wiki_synthesized_at` (post-synthesis patchers) are **NS-invented node properties graphiti's ORM doesn't model** — so they need raw Cypher to write and a raw-Cypher round-trip (`_enrich_graph_results`, `:1596`) to read back.
- The deepest binding — the 120s `created_at` time-window `attach_memory_id` (`graph_patcher.py`) — is forced because NS's *one-`add_episode`→N-memories* shape leaves **no clean node UUID to key on at write time**.

**Implication:** remove (or re-architect) the wiki-synthesizer and roughly **all of `graph_patcher.py` + the enrich round-trip disappear**, leaving only `list_projects` as residual Neo4j coupling. Conversely, the wiki feature — not graphiti — is what keeps NS hard-bound to Neo4j. Any serious graph-portability effort must address the synthesizer first.

*Partial mitigation without removing the feature (M):* `add_episode` returns `result.nodes` (UUIDs); the UUID-keyed patchers and the search-side enrich could move to graphiti's `EntityNode.attributes` (which round-trips arbitrary props, `nodes.py:563,1033`) by reading full `EntityNode` objects on search. This retires the search-side Cypher and the UUID-keyed patchers — but **not** the time-window `attach_memory_id`, which has no UUIDs at write time.

---

## 5. Consolidated effort estimate

| Work item | Bucket | Effort | Risk | Notes |
|---|---|---|---|---|
| Populate `text_lemmatized` + entity store in NS batch writer | opportunity (vector) | **M** (~2–4d) | Low | **Highest value:** unlocks mem0 hybrid/entity-boost for NS-written rows; needs a backfill pass |
| `remove_episode` instead of raw `DETACH DELETE` | opportunity (graph) | **S** (~½d) | Low | Strict correctness win |
| Per-id `Memory.delete` + drop out-of-band history | opportunity (vector) | **S** (~½d) | Low | Wrap with NS edge-expiry |
| Switch to `COMBINED…CROSS_ENCODER` recipe | opportunity (graph) | **S** (~½d) | Med | **Unproven benefit — benchmark first**; adds latency/cost |
| Typed ontology on `add_episode` | opportunity (graph) | **M–L** (~1wk+) | Med | Ontology design + heterogeneous-graph migration |
| Route writes through `Memory.add` (nested→flat payload + reindex) | trap | **L** (~1–2wk) | High | **Not recommended** — re-key all reads + reindex + perf regression |
| Move wiki back-refs to `EntityNode.attributes` | graph-decoupling | **M** (~2–4d) | Med | Retires search-side Cypher + UUID patchers, not time-window patcher |
| Shared pool / full-sweep / is-null → native | impossible | — | — | **Cannot** — mem0 structural gaps |
| Retire NS extraction for mem0's | impossible | — | — | **Cannot** — no OSS categories + parser drops per-fact fields |

**Realistic "go native where it actually helps" program:** the four S/M opportunity items ≈ **1.5–2 weeks** and capture the genuine retrieval-quality and correctness wins. Typed ontology adds ~1 week. Everything else is either structurally impossible or a measured/known regression.

---

## 6. Recommendation

1. **Stop framing it as "rewire everything."** The bypasses encode hard-won knowledge; doc 15's "reuses nothing" is true for *writes* but false for *reads*.
2. **Do the high-value, low-risk subset** (the four S/M items, ~1.5–2 weeks): write-path BM25/entity indexing is the single best return; `remove_episode` and `Memory.delete` are clean correctness wins; the cross-encoder recipe is worth a *benchmarked* trial.
3. **Treat the shared pool, full-collection sweep, is-null, and custom extraction as permanent NS responsibilities** — they're mem0/graphiti structural gaps, and they're precisely the features that make NS more than a thin wrapper.
4. **Recognize the wiki-synthesizer as the graph-portability blocker.** Any move toward the pluggable-graph vision (doc 16 P2/P3) must decide the synthesizer's fate first — it, not graphiti, is what binds NS to Neo4j.
5. **This analysis strengthens doc 16's port design:** the ports must expose the *richer-than-mem0* primitives (cross-writer query, cursor pagination, is-null) as first-class methods, because those are exactly the operations no upstream public API provides — confirming the `VectorPort` must be a superset of mem0's surface, not a proxy of it.
