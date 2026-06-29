# Neuralscape Feature-Parity Gap Analysis: What the Engines Can Do vs. What NS Exposes

*Verified offline on 2026-06-22 against the vendored subtrees (mem0 **2.0.2**, graphiti **0.28.2**) and freshly cloned upstream (mem0 **2.0.7** @ `871a1de7`, graphiti **0.29.2**). Every row is cited to actual source. This is the dimension docs 13 (coupling) and 14 (version-drift) did not cover: **capability parity** — for each thing mem0/graphiti can do, does NS surface it, use it, reimplement it, or ignore it?*

> Provenance note carried from docs 14/README: the vendored `mem0/` is **2.0.2 core (the v3 pipeline: entity store + BM25 + additive scoring) + an NS-restored v1.x graph layer + the NS-authored `graphiti_memory.py` adapter.** Stock upstream 2.0.7 has **zero** graph files. So "mem0 can do X" below means the vendored 2.0.2 surface NS actually ships against.

---

## 1. The headline: NS uses both engines as dependency-injection containers, not via their public APIs

This is the finding that reframes the whole parity question. NS does **not** drive mem0 and graphiti through their public interfaces and then "fall short" on a few features. It bypasses the public interfaces almost entirely and reaches into private internals:

- **mem0:** NS instantiates `Memory.from_config(...)` (`memory_service.py:295-298`) purely as a container, then reaches past the public API into three internals — `Memory.embedding_model`, `Memory.vector_store.client` (the raw `QdrantClient`), and `Memory.graph` (→ `.graphiti`, `._bridge`). **It calls none of `add` / `search` / `get` / `get_all` / `update` / `delete` / `history` / `reset`.** All extraction, search, dedup, storage, update, delete are reimplemented in `memory_service.py` against those internals plus NS's own `google.genai` client.
- **graphiti:** NS uses a thin slice of the public surface (`add_episode`, `search_`, `retrieve_episodes`, the ORM `get_by_group_ids` classmethods) but **bypasses the driver abstraction** with ~7 sites of hand-written Neo4j-dialect Cypher, and **collapses search to a single edge-only recipe**, leaving most of graphiti's retrieval surface structurally present but empty.

**Consequence for parity:** the gap is not "NS exposes 80% of mem0." It's that NS re-implements the *vector orchestration layer itself* and uses only graphiti's *write+dedup+search-core*. Most of what each engine markets as its value-add is unreachable by NS users. The pluggable-backend vision (doc 16) is therefore also an opportunity to **stop reimplementing** and route through a real port that can carry these capabilities.

---

## 2. mem0 capability parity

Status legend: **USES** (called internally) · **EXPOSES** (surfaced via REST/MCP) · **BYPASSES** (NS reimplements the same thing itself) · **IGNORES** (never touched).

| Capability | mem0 module | NS status | NS cite | Parity note |
|---|---|---|---|---|
| `Memory.add(infer=True)` full write pipeline | `memory/main.py:573` | **BYPASSES** | own extraction `memory_service.py:434-491`; raw upsert `:1017-1125` | NS mirrors the v3 pipeline with its own taxonomy |
| `Memory.add(infer=False)` raw store | `memory/main.py:662` | **BYPASSES** | `memory_service.py:1017-1125` | — |
| `Memory.search()` hybrid (semantic+BM25+entity-boost+rerank) | `memory/main.py:1126` | **BYPASSES** | raw `client.query_points` `:811,1420` after `embedding_model.embed` `:788` | **NS does plain vector search; forgoes BM25 fusion, entity boost, rerank entirely** |
| Advanced metadata filter DSL (`gt/lt/in/contains/AND/OR/NOT/*`) | `memory/main.py:1239-1341` | **IGNORES** | hand-built Qdrant `Filter` objects | Users get no general operator surface |
| `Memory.get(id)` | `memory/main.py:973` | **BYPASSES** | Qdrant `scroll`/`retrieve` in service | — |
| `Memory.get_all(filters)` | `memory/main.py:1016` | **BYPASSES** | `client.scroll` `:857,968,2225` | `list_memories` reimplements |
| `Memory.update(id)` (re-embed + in-place + history + entity relink) | `memory/main.py:1501` | **BYPASSES/IGNORES** | delete-only `:2179,2746` | **No true update path; edit = delete+recreate, identity & history lost** |
| `Memory.delete(id)` (+ history row + entity cleanup) | `memory/main.py:1524` | **BYPASSES** | `vector_store.delete` `:2179,2746` | Skips history + entity-store cleanup |
| `Memory.delete_all(scope)` | `memory/main.py:1540` | **BYPASSES** | `:2716-2790` | — |
| `Memory.history(id)` change log (SQLite) | `memory/main.py:1573` | **IGNORES** | — | **No per-fact audit/version trail at all** |
| `Memory.reset()` | `memory/main.py:1752` | **IGNORES** | — | — |
| Procedural-memory summarization | `memory/main.py:1618` | **IGNORES** | — | NS's "procedural" category is its own taxonomy, unrelated |
| Custom fact-extraction prompt hooks | `memory/main.py:583,729` | **BYPASSES** | NS owns extraction (`prompts.py`) | — |
| Built-in entity store + entity-boost scoring | `memory/main.py:389-532`; `utils/scoring.py` | **IGNORES** | Graphiti is NS's graph instead | Redundant graph subsystem sits dead in the subtree |
| BM25 hybrid keyword search | `memory/main.py:1362`; `utils/lemmatization.py` | **IGNORES** | pure vector | **Retrieval-quality gap, esp. rare proper nouns** |
| Reranker subsystem (cohere/sentence-transformer/zero-entropy/llm/hf) | `utils/factory.py:238-289`; `reranker/*` | **IGNORES** (vector path) | only graphiti reranker configured `config.py:303,320` | `search(rerank=True)` + 5 rerankers unreachable |
| Embedder roster (11 providers) | `utils/factory.py:139-164` | **USES (pinned)** | `config.py:363-396` pins gemini/openai | abstraction used, 9 providers unreachable |
| LLM roster (17 providers) | `utils/factory.py:30-56` | **BYPASSES** | configured but never called; extraction via own `genai.Client` `:442` | **The configured mem0 LLM is dead weight** |
| Vector-store roster (23 providers) | `utils/factory.py:167-204` | **USES (pinned qdrant) + leaks client** | `config.py:402`; raw client `:787,848,960,1387,2205,2716,2774` | **Qdrant-specific calls make the choice non-swappable** |
| Graph-store roster | `utils/factory.py:212-234` | **USES (graphiti) + leaks internals** | `.graph.graphiti`/`._bridge` `:340-341` | semi-public `.add`, then private driver |
| `AsyncMemory` | `memory/main.py:1795` | **USES (container only)** | `main.py:99-105`; wraps sync with `to_thread` | async public methods not called |
| Telemetry / score explainability | `memory/telemetry.py`; `utils/scoring.py` | **IGNORES** | returns Qdrant score only | — |
| `from_config(dict)` | `memory/main.py:534` | **USES** | `memory_service.py:298` | **The one public entry point NS genuinely uses** |

---

## 3. graphiti capability parity

| Capability | graphiti module | NS status | NS cite | Parity note |
|---|---|---|---|---|
| `add_episode` (single) | `graphiti.py:980` | **USES** (sole write path) | `graphiti_memory.py:320` | always single-episode |
| `add_episode_bulk` | `graphiti.py:1230` | **IGNORES** | — | one ARQ task per memory → never batches |
| Custom `entity_types` | `graphiti.py:990` | **IGNORES** | `graphiti_memory.py:320-328` passes none | **headline feature unused; every node is generic `Entity`** |
| Custom `edge_types` / `edge_type_map` | `graphiti.py:993-994` | **IGNORES** | — | falls to empty default map |
| `excluded_entity_types` / `custom_extraction_instructions` | `graphiti.py:991,995` | **IGNORES** | — | — |
| Temporal invalidation (`valid_at`/`invalid_at`/`expired_at`) | edge_operations | **USES (implicit)** | inherent in `add_episode`; sets `reference_time` `:317` | **but never reads/exposes validity on results** — edges flattened to `{uuid,name,fact}` `:1567` |
| Node + edge dedup | node/edge_operations | **USES (implicit)** | inherent in `add_episode` | the adapter's stated reason to exist `:148-150` |
| `search` (basic, edges) | `graphiti.py:1526` | **USES** (adapter) | `graphiti_memory.py:370` | `EDGE_HYBRID_SEARCH_RRF` |
| `search_` (advanced multi-layer) | `graphiti.py:1602` | **USES + EXPOSES** | `:1565,1695`; REST `main.py:572`; MCP `:275` | NS's primary graph read |
| Search recipes (16: RRF/MMR/cross-encoder × edge/node/community/combined) | `search_config_recipes.py` | **BYPASSES default** | hardwires `EDGE_HYBRID_SEARCH_RRF` `:1560,1689` | **`search_`'s own default is `COMBINED_…CROSS_ENCODER`; NS overrides to edge-only** |
| MMR reranking | recipes | **IGNORES** | — | reachable only via raw config override |
| Cross-encoder reranking | `cross_encoder/*` | **CONFIGURED, NEVER INVOKED** | client built `graphiti_memory.py:176`; config `:303,320` | **pays to construct a reranker no recipe ever triggers** |
| Node / episode / community search layers | search_config | **EXPOSES shape only** | `.nodes/.episodes/.communities` mapped `:1571-1581` | **structurally present but empty under edge-only recipe** |
| `retrieve_episodes` | `graphiti.py:926` | **USES + EXPOSES** | `:2330`; REST `main.py:482` | — |
| `build_communities` | `graphiti.py:1489` | **IGNORES (removed)** | replaced by category-bucketing `synthesizer.py:17-19` | communities only read, never built |
| `update_communities` | `graphiti.py:1184` | **IGNORES (off)** | `config.py:60` `=False` | — |
| Sagas (ordered episode chains + incremental summaries) | `graphiti.py:346-568,1410` | **IGNORES** | no `saga=` anywhere | entire subsystem unused |
| `add_triplet` (manual fact insertion) | `graphiti.py:1645` | **IGNORES** | — | no manual fact path |
| `remove_episode` (+ orphan cleanup) | `graphiti.py:1765` | **BYPASSES** | raw `DETACH DELETE` `:2390` | **weaker than upstream — leaves orphaned nodes/edges** |
| Driver roster (neo4j/falkordb/kuzu/neptune) | `driver/*` | **IGNORES all but Neo4j** | adapter builds `Neo4jDriver` only `graphiti_memory.py:185` | abstraction defeated by raw Cypher (§5) |
| `graph_driver=` injection | `graphiti.py:147` | **USES** | `graphiti_memory.py:192-197` | injects Neo4j only |
| LLM / embedder / cross-encoder rosters | `llm_client/*`, `embedder/*`, `cross_encoder/*` | **USES subset** | `graphiti_memory.py:33-115` | gemini/openai-gateway; Azure/GLiNER2 unused |
| OTel tracing | `graphiti.py:149` | **IGNORES** | own structlog | — |

---

## 4. Ranked parity gaps (combined, user-visible first)

1. **No per-fact history / audit / true update (mem0 `history`+`update`).** The most user-visible gap for something called a "memory layer." Facts can only be deleted, not versioned or edited in place. *(mem0 §2)*
2. **Typed graph ontology unused (graphiti `entity_types`/`edge_types`).** NS's 13-category taxonomy never reaches the graph; every node is a generic `Entity`. This is graphiti's flagship feature and the **highest-value, lowest-risk win — it needs no upstream sync** (`add_episode` already accepts the kwargs at both versions). *(graphiti §3)*
3. **Hybrid retrieval left on the table (mem0 BM25 + entity-boost; graphiti cross-encoder).** NS does plain vector search and forces edge-RRF, so two independent reranking/keyword-fusion subsystems it already ships are inert. Measurable recall/precision loss, especially rare proper nouns. *(mem0 §2, graphiti §3)*
4. **Multi-layer graph search collapsed to edges-only.** `nodes`/`episodes`/`communities` are plumbed through the API but empty under the default recipe — the surface *looks* richer than it behaves. *(graphiti §3)*
5. **Advanced metadata filter DSL unreachable (mem0).** No general `gt/lt/in/contains/AND/OR/NOT` operators for queries like "concepts contains X AND created_at > Y." *(mem0 §2)*
6. **Backend portability already forfeited.** Raw `QdrantClient` calls and Neo4j-dialect Cypher hard-bind NS to Qdrant+Neo4j today (§5) — the pluggable vision starts from negative, not zero. *(both)*
7. **Sagas, `add_triplet`, `add_episode_bulk`, communities — entirely unused graphiti subsystems** that NS's session/conversation and batch-ingest models could map onto. *(graphiti §3)*
8. **`remove_episode` bypass leaves graph bloat.** NS's raw `DETACH DELETE` skips orphan-node/edge cleanup. *(graphiti §3)*

---

## 5. Portability blockers (what hard-binds NS to Qdrant + Neo4j today)

**Vector → Qdrant.** NS reaches past mem0's wrapper to the raw `QdrantClient` for everything the wrapper can't do — cross-writer shared-pool search, pagination, is-null, OR-filters, projected scrolls: `memory_service.py:787,811,848,960,1387,1420,2205,2716,2774` + Qdrant-typed filter imports `:780-785`. A `VectorPort` must therefore expose **richer query/scroll primitives than mem0's wrapper**, not just proxy it.

**Graph → Neo4j (three independent bindings).** (a) adapter only ever builds `Neo4jDriver` (`graphiti_memory.py:185`); (b) ~7 raw-Cypher sites use Neo4j-dialect constructs (`datetime()`, `STARTS WITH`, label-less `MATCH (n)`, `coalesce` on `SET`); (c) those sites depend on Neo4j range indexes and NS-custom node props graphiti's ORM doesn't model:

| Site | Query gist | Portability |
|---|---|---|
| `memory_service.py:1621` enrich | `MATCH (n) WHERE n.uuid IN $u RETURN n.memory_id,n.wiki_path` | label-less scan unsafe on Kuzu/Neptune |
| `memory_service.py:1930` list_projects | 3× `MATCH (:Label) WHERE group_id STARTS WITH … UNION` | Neo4j-only |
| `memory_service.py:2390` delete episode | `MATCH (e:Episodic{uuid}) DETACH DELETE e` | Kuzu differs; weaker than `remove_episode` |
| `graph_patcher.py:66` attach_memory_id | `datetime()` + `coalesce` on `SET` | Neo4j-only |
| `graph_patcher.py:130,203` patch_wiki_path | label-less `MATCH (n)` + custom props | unsafe on Kuzu/Neptune |
| `synthesizer.py:321` shared groups | `STARTS WITH` + label-less scan | Neo4j-only |
| `scripts/migrate_graph_groups.py:77` | label-interpolated `STARTS WITH` | Neo4j-only (tooling) |

---

## 6. "Stop forking / start delegating" candidates

Because NS reimplements the vector layer, several NS subsystems duplicate code the engines already ship:

- **Get / list / delete-all** over raw `scroll` (`memory_service.py:857-968,2716-2790`) duplicate mem0 `get_all`/`delete_all` — **lowest-risk methods to route back through a port.**
- **Batch embed + upsert** (`:1017-1125`) re-implements mem0's batch insert; NS already uses `embed_batch` (`:1092`), so switching to `vector_store.insert` would drop hand-written plumbing **and** regain store portability.
- **Dedup** (MD5 + semantic) overlaps mem0's hash dedup (`main.py:784-803`).
- **Entity graph**: mem0's entity store vs graphiti — pick one; the other is dead weight in the subtree.
- **Extraction**: NS's `genai` extraction duplicates the *mechanics* of mem0's v3 extractor (existing-memory retrieval, anti-hallucination UUID mapping, batch embed, hash dedup) — only the taxonomy differs.

---

## 7. What this means for the pluggable design (doc 16)

The parity findings define the **minimum surface a port must carry** so NS can keep doing everything it does today while becoming backend-agnostic — and the **capability flags** a port must advertise, because no two backends support the same feature set (this is why §4's gaps can't simply be "turned on" uniformly). Doc 16 derives `VectorPort` / `GraphPort` method sets directly from the real call sites enumerated here, plus a **capability-negotiation** layer so a backend that lacks (say) BM25 or cross-encoder degrades gracefully instead of breaking.
