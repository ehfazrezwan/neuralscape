# Neuralscape: Deep Code Analysis, Gap Analysis & Evolution Architecture

*Lead architect synthesis of six specialist deep-reads. Evidence is `file:line` into `/Users/ehfaz.rezwan/Projects/neuralscape/`. Where the source findings disagreed, the contradiction is named and resolved inline.*

---

## 1. Executive Summary

Neuralscape is already 80% of the way to the owner's new vision, and almost nobody on the team has noticed — because the unifying layer they want to build *already exists as the policy/product layer*, and the thing actually blocking it is a few dozen lines of raw-client leakage buried inside one 2,924-line file.

The central thesis, and the single most important decision in this document: **the swappable unit must be the DATABASE, not the ENGINE.** You keep mem0-the-orchestrator and Graphiti-the-engine intact — they *are* the frontier engineering (LLM episode extraction, bi-temporal edge invalidation, hybrid RRF search) the vision explicitly says to reuse — and you make their *backends* pluggable via config. mem0 already abstracts 24 vector databases behind `VectorStoreFactory`; Graphiti already abstracts 4 temporal-graph databases behind `GraphDriver`. Neither needs reinventing. "Abstract them away" (a pure NS-owned port layer that drops mem0/Graphiti) is the wrong path: it forfeits exactly the engineering you want to keep and would have NS reimplement temporal contradiction detection from scratch.

What blocks "swap the backend by config today" is narrow and concrete: (1) two hardcoded provider literals (`config.py:403,407`), and (2) roughly nine raw-`QdrantClient` escape hatches plus a cluster of hand-written Neo4j Cypher strings in `memory_service.py` and the wiki synthesizer that reach *around* the abstractions NS already inherited. The fix is not to build an abstraction — it is to **stop leaking around the ones you already have**, behind two narrow NS-owned ports (`VectorPort`, `GraphPort`) that delegate to the upstream factories and isolate only the operations the upstream interfaces genuinely don't cover. This is a deliberate, defensible evolution of the recorded "opinionated dual-backend" philosophy — held as **opinionated defaults, pluggable internals**.

---

## 2. What Neuralscape Is Today

Neuralscape is a production-grade agentic memory layer that wraps **mem0** (vector-store orchestration) + **Graphiti** (temporal knowledge graph) and exposes REST (FastAPI), an MCP server (8 tools), and a Claude Code/Cowork plugin. Writes are async (enqueued to Redis/ARQ, return 202); reads are synchronous (return 200). Backends today: Qdrant (vectors), Neo4j (graph via Graphiti), Redis/ARQ (queue), Gemini (LLM extraction + embeddings).

**Stale doc claims, corrected (all six findings agree):**

| Claim | Reality | Evidence |
|---|---|---|
| mem0 "upstream v1.0.10" (stored team memory) | **mem0 vendored = 2.0.2** | `mem0/pyproject.toml` → `version = "2.0.2"`; NS code pins to it at `memory_service.py:281` ("upstream c9e8482a, merged in #38") |
| graphiti version | **0.28.2** (accurate) | `graphiti/pyproject.toml` |
| `memory_service.py` is "~1200 LOC" (CLAUDE.md) | **2,924 lines** | direct count |
| MCP "7 tools" (CLAUDE.md) | **8 tools** | `mcp_server.py` |

**Resolved contradiction:** the stored-memory "v1.0.10" is *stale*, not a mislabel. Every finding independently verified `2.0.2` from the subtree, and the service code repeatedly works around 2.0.2-specific behavior (the `limit`→`top_k` rename at `memory_service.py:1877`; the removed graph auto-wiring re-attached at `:300-341`). **Trust the subtree, not the stored memory.** Note also the red herring: `config.py:418` emits `"version": "v1.1"` — that is mem0's *memory-schema* version, unrelated to the package version.

A defining architectural fact that drives much of the analysis: **mem0 2.0.2 removed graph auto-initialization from `Memory`.** NS compensates by manually re-attaching the Graphiti adapter via a `SimpleNamespace` shim and reaching into private internals — `from mem0.utils.factory import GraphStoreFactory`, then `self._memory.graph.graphiti` / `self._memory.graph._bridge` (`memory_service.py:300-341`; duplicated in `main.py:78-80`). This hack is simultaneously NS's cleverest glue and its most fragile coupling — a live signal that leaning on mem0's *internal wiring* (rather than its *public factories*) as your abstraction is a maintenance liability.

---

## 3. Where Neuralscape Shines

The crucial strategic insight: **NS's identity-defining value splits cleanly into a portable policy/product layer and an entangled I/O layer.** The policy layer is *already* the backend-agnostic unifying layer the owner wants — it is expressed against `MemoryResponse` objects and `MemoryService` method signatures, not against Qdrant/Neo4j.

**Fully backend-agnostic crown jewels (survive any swap untouched):**

- **13-category taxonomy + scope/visibility/vault defaults** (`schemas.py:79-158`). A controlled vocabulary across 5 memory types, with three orthogonal default-resolution tables: category→scope, category→visibility, category→vault path. Pure Pydantic, zero backend references. The most portable asset NS owns. Neither mem0 (free-form categories) nor Graphiti (no category concept) provides this *policy*.
- **Memory-model v2 controlled vocabularies** (`schemas.py:165-335`): `domain`, `observation_type`, `concepts[]`, `source_type`, `confidence`, `expires_at`, each validated against a vocab with **symmetric write-path and filter-path validation** (typos rejected as 422 on both). Additive — omitting them reproduces v1.
- **`group_id` composition algebra** (`memory_service.py:206-259`): folds three access axes (owner, team-shared, project) into one namespace string, with read-set expansion yielding "your private ∪ team shared." The docstring (`:220-223`) is explicit that this **fixes a real cross-user leak in Graphiti**. Genuine multi-tenant access-control engineering.
- **The agent-facing product layer**: 8 MCP tools, REST mirror at `/v1/*`, the self-loop-guarded plugin capture/inject lifecycle (`post-tool-use.ts`, `session-start.ts`), and an embedded **OAuth 2.1 Authorization Server** purpose-built for the Cowork connector flow (`oauth.py` — RFC 8414/9728, PKCE, HMAC `typ`-tagged tokens, fail-closed email allowlist in `allowlist.py:32-57`). This entire layer touches no Qdrant/Neo4j code — it calls `MemoryService` methods. **It survives a backend swap untouched.**
- `worker.py` and `mcp_server.py` are **clean delegators** — no direct engine calls. They are not coupling blockers.

**Crown jewels that are accidentally coupled to Qdrant/Neo4j today** (the design is portable; the *implementation* is welded):

- **Dual-backend merge / interleave / source-authority** (`search()`, `memory_service.py:1133-1357`; `_deduplicate_responses:2558`; `_merge_results:2909`). The orchestration operates on `MemoryResponse` objects (portable), but the *fetch* underneath is hard-bound: shared-pool search uses raw `qdrant_client.models` (`:780`), graph search imports `EDGE_HYBRID_SEARCH_RRF` and calls `g.search_()` over the private `_bridge` (`:1550-1565`). **This is the single most entangled crown jewel** — the merge *idea* is the project's identity, but its call path assumes both a Qdrant payload shape and Graphiti's search-recipe API simultaneously.
- **Graph→vector v2 enrichment** (`_enrich_graph_with_v2`, `:1364-1470`): a bespoke cross-store join — embed each graph edge, top-1 Qdrant search above a 0.7 threshold, copy v2 metadata onto the graph row. The *pattern* ("enrich engine-A rows from engine-B by vector similarity") is exactly what a unifying layer should own; the implementation reads Graphiti edges *and* issues raw `client.query_points` (`:1384,:1420`).
- **Category-aware Gemini extraction + batch upsert** (`prompts.py:25-66`; `_batch_store_facts:1008-1125`): prompt-as-classifier + single-embed/single-upsert (a real efficiency win over mem0's per-fact add). Prompt/parse is agnostic; the batch-store tail is Qdrant-coupled and Gemini-coupled.
- **Wiki synthesizer** (`extensions/wiki_synthesizer/`): walks Graphiti communities→source memories using `memory_id`/`wiki_path` provenance NS back-stamps onto graph nodes via raw Cypher (`graph_patcher.py:66-81,130-159`). Neo4j-coupled.

---

## 4. Gap Analysis

| Capability | mem0 / graphiti offers | NS uses today | Gap | Impact |
|---|---|---|---|---|
| **Vector backends** | mem0 `VectorStoreFactory`: 24 providers (qdrant, pgvector, pinecone, milvus, weaviate, redis, supabase, elasticsearch, faiss, …) | **qdrant only** (hardcoded `config.py:403`) | 23 backends unreachable; provider is not a config knob | **Blocks the "any vector DB" half of the vision.** Trivial config fix + remove raw-client leaks |
| **Graph drivers** | graphiti `GraphDriver`: Neo4j, FalkorDB, Kuzu, Neptune | **Neo4j only** (mem0 adapter hardcodes `Neo4jDriver`, `graphiti_memory.py:185`) | 3 drivers unreachable; NS fishes the instance out of mem0 instead of constructing `Graphiti(graph_driver=...)` | **Blocks the "any temporal graph DB" half.** Note: **Kuzu deprecated upstream 0.29.2** — do not target it |
| **Graph search recipes** | 16 prebuilt recipes (combined/edge/node/community × rrf/mmr/node_distance/episode_mentions/cross_encoder) | **1** (`EDGE_HYBRID_SEARCH_RRF`, and only on advanced/legacy paths; the hot mem0-adapter path uses bare default) | 15 recipes unused; MMR/node_distance/episode_mentions rerankers untouched | Missed retrieval quality; the merge layer could exploit richer recipes |
| **Rerankers** | mem0: 5 (cohere, llm_reranker, …); graphiti cross-encoder: bge/gemini/openai | **none** explicitly requested | Cross-encoder reranking never invoked | Quality left on the table |
| **Custom entity/edge types (ontology)** | graphiti `add_episode(entity_types=, edge_types=, edge_type_map=)` | **none** — adapter passes no types | NS's 13-category taxonomy is *not* projected into the graph ontology | The graph is generically typed; NS's rich category model doesn't shape graph structure |
| **Communities** | `build_communities()` + community search | **abandoned** ("order of magnitude too slow," `synthesizer.py:17`) — replaced with category bucketing | NS hand-rolls community-like grouping | Acceptable for now; revisit if upstream perf improves |
| **Bulk ingest** | `add_episode_bulk()` | **none** | Per-episode ingest only | Throughput ceiling on large imports |
| **Retrieval explainability** | mem0 2.0.5 `search(explain=True)` → `score_breakdown` (semantic/keyword/entity_boost/temporal_boost) | **N/A** (not in vendored 2.0.2) | NS's merge can't use sub-scores | **Strategically valuable for the merge layer** — pull on sync |
| **Score normalization [0,1]** | mem0 2.0.5 across 11 adapters | **N/A** (2.0.2) | NS merge may compare raw distances | **Breaking on sync**: merge code assuming raw distance ranks backwards. Audit before upgrading |
| **Combined node+edge extraction** | graphiti 0.29.0 (`use_combined_extraction`) — one LLM call | **N/A** (0.28.2) | Higher extraction cost | Cost win; opt-in, low-risk to adopt |
| **Provenance (memory_id↔node), categories, multi-user isolation, content-hash dedup, TTL expiry** | **Upstream will never provide** | NS owns all of it | These are NS's permanent moat | Must stay NS-owned; the port layer must preserve them across any backend |

**Net upstream-delta posture:** mem0 is 5 patch releases behind (2.0.2 → 2.0.7), graphiti is 1 minor + 2 patches behind (0.28.2 → 0.29.2). Both deltas are small and low-risk; the big backend rosters were already vendored. The two changes that *matter* on sync are mem0's [0,1] score normalization (breaks naive distance-based merge) and graphiti's safer attribute merging that no longer lets attribute dicts overwrite `group_id` (verify NS never sets `group_id` via an attribute dict — it doesn't appear to, but confirm).

---

## 5. The Coupling That Blocks the Vision

**Blocking a VECTOR-DB swap** (all in `memory_service.py` unless noted):

1. **Hardcoded provider literal** — `config.py:403` emits `"provider": "qdrant"`. Not a setting.
2. **~9 raw `QdrantClient` escape hatches** reaching past mem0's `VectorStoreBase` into `m.vector_store.client`:
   - `client.query_points(...)` — `:811, :1420` (shared-pool OR-query and enrichment; mem0's filters are AND-only equality, so the "(user_id=caller) OR (visibility=shared)" query *cannot* be expressed through the abstraction).
   - `client.scroll(..., next_offset)` — `:857, :968, :2225, :2725, :2780` (content-hash dedup, scroll-all pagination, recent-scan-for-dedup). `VectorStoreBase.list(filters, top_k)` has **no cursor contract** — it cannot deterministically page a whole collection.
   - `IsNullCondition` null-category listing — `:2196`. Not expressible in mem0's flat filter dict.
   - `qdrant_client.models.Filter/FieldCondition/MatchValue/MatchAny` imports — `:780, :847, :1384, :2197, :2714`.
3. **Qdrant payload schema as the de-facto data contract** — dotted keys `metadata.scope/category/project_id/domain/observation_type/concepts/visibility` + top-level `user_id/hash` (`:791-807, :1399-1412`). Both reads and filters assume this nested layout.
4. **768-dim embedding baked into config + collection** — `config.py:334,369,394`. Any embedder with different dims forces collection recreation.

**Resolved contradiction (does `VectorStoreBase` suffice?):** one finding says "adopt `VectorStoreFactory`, NS already runs on ~80% of it"; another says "`VectorStoreBase` leaks badly." Both are right at different altitudes. The *factory* is the correct construction mechanism (use it). The *base interface* under-covers four NS operations (hash-dedup, scroll-all, null-filter, boolean-OR) that live below it bound to the concrete Qdrant client. **Resolution: delegate construction to the factory; own a thin port for the four leaking operations.**

**Blocking a GRAPH-DB swap** (harder than the vector side):

1. **The mem0-internal graph re-attach hack** — `memory_service.py:300-341`. NS fabricates a shim of mem0's removed `MemoryConfig.graph_store` and grabs private `_memory.graph.graphiti` / `_bridge`. To swap drivers, NS must instead construct `Graphiti(graph_driver=...)` directly (a config-selected `GraphProvider`) rather than fishing the instance out of mem0.
2. **Raw, Neo4j-dialect Cypher** — `:1621-1632` (enrichment), `:1930-1949` (`list_projects`, which further relies on **Neo4j-specific per-label range indexes** for `STARTS WITH` performance), `:2390-2393` (episode `DETACH DELETE`), plus `graph_patcher.py:66-81,130-159`. FalkorDB/Kuzu/Neptune drivers expose `session()` but with dialect differences. **This is the real graph portability tax** — not the driver interface, but NS's hand-written Cypher.
3. **Direct `graphiti_core` ORM/recipe imports** — `EntityNode/EntityEdge/CommunityNode.get_by_group_ids`, `EDGE_HYBRID_SEARCH_RRF`, `edge.expired_at=...; edge.save(driver)` (`:2258-2366, :2606, :2635-2698`). Temporal invalidation is bound to Graphiti's edge model — but this is *Graphiti-the-engine* coupling, which we **intend to keep**; it is portable across all 4 drivers for free.
4. **The `_bridge` async event-loop coupling** — every direct graph call routes through Graphiti's private loop runner (`:341, :363, :1564`).

**Cross-cutting:** the graph coupling is **duplicated in `main.py` legacy endpoints** (`:64-95, :392-572`) — any swap must be done twice, or the legacy paths removed first. And the **standalone Gemini extractor** (`memory_service.py:16, :442, :481-491`) is a third axis, independent of both engines, only partially parameterized via the gateway.

---

## 6. Evolution Architecture

### The governing principle: opinionated DEFAULTS, pluggable INTERNALS

This is how to hold the recorded "store twice, query both, merge" philosophy *and* the new pluggable vision without contradiction. The recorded philosophy is correct as a **default posture and product opinion** — NS ships dual-backend, ships Qdrant+Neo4j, ships the merge-with-graph-authority rule. What evolves is that the *engine identity below the default becomes a config knob*. You are not abandoning the opinion; you are refusing to *hardcode* it. The team's memory that says "no plan to abstract the backends" should be updated: the dual-backend *merge* is the permanent opinion; the *specific databases* are now an implementation detail.

### What NS owns vs. delegates

**Delegate to upstream (do not re-wrap):**
- **Engine/provider construction:** mem0 `VectorStoreFactory.create(provider, cfg)` and `Graphiti(graph_driver=<provider>)`. NS config picks the provider string; the factories build it.
- **The intelligence:** mem0's extraction-orchestration where used, Graphiti's `add_episode`, bi-temporal edge invalidation, hybrid RRF search. This is the frontier work — keep it.

**NS owns two narrow Protocol ports** (~6-8 methods each), defined precisely by what NS already does *around* the upstreams:

```
VectorPort (Protocol)
  upsert(records)                      # → mem0 vector_store.insert
  search(vector, filters, top_k)       # → mem0 vector_store.search
  get(id) / delete(id)                 # → mem0 wrapper
  scroll_all(filters) -> iterator      # cursor abstraction  (covers :857,:968,:2725)
  find_exact(field_eq: dict)           # hash-dedup + null-category (covers :833,:2196)
  search_or(filter_groups)             # shared-pool OR query  (covers :811,:1420)

GraphPort (Protocol)               # wraps Graphiti, NOT the DB
  add_episode(text, group_id, ...)     # → graphiti.add_episode (keep ontology hook open)
  search(query, group_ids, recipe)     # → g.search_ with selectable recipe
  list_entities/edges/communities(...) # → *.get_by_group_ids
  invalidate_edge(edge)                # → edge.expired_at; edge.save(driver)
  run_read_cypher(query, params)       # THE DIALECT SEAM
  delete_group(group_id) / delete_episode(uuid)
```

`run_read_cypher` is the load-bearing seam: isolate **every** hand-written Cypher string (`list_projects`, enrichment, synthesizer patchers) behind it, so a non-Neo4j driver needs only that one method re-dialected — not the whole service.

### How the crown-jewel layers become backend-agnostic

The policy layers (§3) are *already* agnostic — they only need their I/O re-pointed at the ports:
- **Merge/dedup/source-authority** (`_deduplicate_responses`, `_merge_results`) already operate on `MemoryResponse` — leave them; just have `search()` fetch via `VectorPort.search_or` + `GraphPort.search` instead of raw clients.
- **Graph→vector enrichment** becomes `GraphPort.list_edges()` + `VectorPort.search()` — the cross-store-join *pattern* NS should own, now expressed against ports.
- **group_id algebra, taxonomy, v2 vocabularies, OAuth/MCP/plugin** — untouched; they live above the ports.
- **Provenance back-stamping** moves behind `GraphPort.run_read_cypher`.

### Target architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  AGENT-FACING SURFACE  (backend-agnostic, survives any swap)      │
│  MCP (8 tools) · REST /v1 · Plugin hooks · OAuth 2.1 · allowlist  │
└───────────────────────────────┬─────────────────────────────────┘
                                 │ MemoryService method calls
┌───────────────────────────────▼─────────────────────────────────┐
│  POLICY / CONTRACT LAYER  (NS's moat — pure, portable)            │
│  13-cat taxonomy · v2 vocab+validation · group_id algebra ·       │
│  merge/interleave/source-authority · dedup/TTL · provenance       │
│  Gemini category-aware extraction (LLM axis, gateway-routable)    │
└──────────────┬───────────────────────────────┬──────────────────┘
               │ VectorPort                     │ GraphPort
┌──────────────▼──────────────┐  ┌──────────────▼──────────────────┐
│  mem0 ORCHESTRATOR (keep)   │  │  GRAPHITI ENGINE (keep)          │
│  extraction · dedup-merge   │  │  add_episode · bi-temporal       │
│  VectorStoreFactory ────────┼──┼─ invalidation · RRF/recipes      │
└──────────────┬──────────────┘  │  GraphDriver ────────────┐      │
   provider=cfg │                 └──────────────────────────┼──────┘
┌──────────────▼──────────────┐         provider=cfg         │
│ Qdrant│pgvector│pinecone│... │  ┌──────────────────────────▼──────┐
│   (24 via factory)          │  │ Neo4j│FalkorDB│Neptune  (Kuzu ✗) │
└─────────────────────────────┘  └─────────────────────────────────┘
   DEFAULT: Qdrant                   DEFAULT: Neo4j
   = opinionated default,            = opinionated default,
     pluggable internal                pluggable internal
```

---

## 7. Phased Roadmap

Ordered to deliver value early and de-risk. Reconciled with NS's recorded roadmap (data-layer connectors, LLM-gateway routing, OAuth, wiki) — note that **gateway routing and OAuth are already shipped** (commits bad3406, the auth branch in flight), so this roadmap layers the backend-pluggability work on top of that base.

**P0 — De-couple & seam (no behavior change).** *Goal:* introduce `VectorPort`/`GraphPort` Protocols with a single Qdrant+Neo4j default impl; route all `MemoryService` I/O through them; collapse the duplicated graph coupling in `main.py` legacy endpoints into the service (or delete legacy paths). Isolate every raw Cypher string behind `GraphPort.run_read_cypher`. *Files:* new `ports/` package; `memory_service.py` (all engine call-sites), `main.py:64-95,392-572`, `extensions/wiki_synthesizer/graph_patcher.py`. *Risk:* medium — large mechanical refactor of a 2,924-line file; mitigate with the existing test suite + a characterization test on `search()` merge order.

**P1 — Config-driven vector backend swap via mem0 factory.** *Goal:* make `vector_store.provider` + its config dict env-driven; replace the ~9 raw-client leaks with `VectorPort` methods (`scroll_all`, `find_exact`, `search_or`) whose default impl still uses the Qdrant client but is now swappable; add per-provider dim/`on_disk` conditionals. Prove it by standing up pgvector or pinecone in CI. *Files:* `config.py:336-403`, `memory_service.py:780-996,1384-1467,2196-2780`. *Risk:* medium — the OR-query and scroll-cursor semantics differ per backend; some backends lack `keyword_search` (mem0 2.0.5 warns about this — relevant once synced).

**P2 — Config-driven graph driver swap via Graphiti.** *Goal:* stop fishing `self._memory.graph.graphiti` out of mem0; construct `Graphiti(graph_driver=<GraphProvider>)` directly with a config-selected provider; provide a FalkorDB or Neptune dialect impl of `run_read_cypher`. *Files:* `memory_service.py:300-341` (replace the shim), `config.py:407` (provider knob + driver conn settings). *Risk:* high — the re-attach hack is the most fragile glue; the dialect tax on hand-written Cypher is real. **Do not target Kuzu** (deprecated upstream 0.29.2). Mitigate by keeping Neo4j the default and gating new drivers behind feature flags.

**P3 — Adopt unused upstream richness.** *Goal:* project NS's 13-category taxonomy into Graphiti `entity_types`/`edge_types` ontology; expose selectable search recipes/rerankers through `GraphPort.search(recipe=)`; evaluate combined extraction (graphiti 0.29.0) for cost. Sync upstreams (mem0→2.0.7, graphiti→0.29.2) **after auditing the merge code for the [0,1] score-normalization break** and the 0-based episode index change. *Files:* adapter `add_episode` call, `search_config` plumbing (`memory_service.py:1550-1565`), `prompts.py`. *Risk:* low-medium — opt-in features; the score-normalization audit is the one sharp edge.

**P4 — Broader engine-agnostic posture + connectors.** *Goal:* fold mem0 2.0.5 `score_breakdown` into the merge ranking; formalize the cross-store-join enrichment as a first-class `VectorPort`+`GraphPort` capability; wire the recorded data-layer connectors as additional write sources feeding the same ports. *Risk:* low — additive, built on P0-P3 seams.

---

## 8. Risks, Unknowns & Verification TODOs

1. **`get_project_context` returns empty for project "neuralscape" — likely root cause located.** `memory_service.py:1765-1768` filters project memories with `filters={"user_id": user_id, "metadata.project_id": project_id}` and `top_k=200`. This path **does not use the shared-pool bypass** (`_search_shared_pool`) that exists precisely because mem0's `user_id` namespacing cannot express "any writer, shared visibility." So any project memory written with `visibility=shared` by a *different* user_id — or stored under a project_id-key shape that doesn't match `metadata.project_id` — is invisible here. **Three concrete suspects to verify:** (a) the caller's `user_id` resolved by the new auth/identity slugging (`identity.py`) differs from the `user_id` the memories were stored under (very likely given the in-flight auth branch changes the user_id derivation); (b) the memories carry `metadata.project_id` under a different nesting (`_mem_to_response` unwraps `{metadata:{metadata:{}}}` at read but the *filter* at `:1766` assumes one level); (c) `top_k=200` silently caps. **TODO:** dump `m.get_all(filters={"user_id": <resolved id>})` raw for the neuralscape project and compare the stored `user_id`/`metadata.project_id` against what the MCP caller resolves to. The `mcp_server.py:261` default page size of 25 is *not* the cause — that only pages an already-populated result.

2. **mem0 version ambiguity — resolved, but update the stored memory.** Vendored is 2.0.2 (verified four ways). The stored "v1.0.10" must be corrected in the team's own memory, since it has already misled at least one analysis input.

3. **Score-normalization break on mem0 sync (2.0.5).** Before upgrading, audit `_deduplicate_responses`/`_merge_results` and `search()` for any assumption that lower score = better. If present, the merge ranks backwards post-upgrade. **Blocking gate on P3 sync.**

4. **Graphiti `group_id` attribute-merge safety (0.29.0).** Confirm NS never sets `group_id` through an attribute dict (it appears to set it as a first-class field via the adapter — verify in `graphiti_memory.py` call). Low risk but cheap to confirm.

5. **Graphiti driver layer is mid-refactor** ("Phase 1 / legacy interface" markers in `driver/driver.py`). FalkorDB/Neptune dialect behavior under `run_read_cypher` is not yet validated by NS. Pin 0.28.2→0.29.2 deliberately and track the operations/search-interface refactor before committing P2 to a non-Neo4j default.

6. **Legacy `main.py` endpoint duplication** means P0/P1/P2 must touch two code paths. Confirm whether the legacy root endpoints are still in use (vs. only `/v1/*`); if dead, delete them first to halve the refactor surface.

7. **Embedding-dim portability (768).** Any backend/embedder swap producing non-768 vectors requires collection recreation + re-embedding. There is no migration path today — P1 must define one before advertising embedder pluggability.

8. **The standalone Gemini extractor is a third, separate swap axis** (`memory_service.py:16,442`), only partially covered by the gateway. The "any LLM" axis is real but out of scope for the vector/graph pluggability vision — name it explicitly so it isn't conflated.
