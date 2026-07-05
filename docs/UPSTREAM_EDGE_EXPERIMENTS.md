# Upstream Edge Experiments — unused mem0/graphiti capability worth mining

Status: **planned, not started** (2026-07-05). This is the follow-up plan from
the subtree audit: NS uses ~4% of the vendored code by volume, and — more
interestingly — **1 of graphiti's 16 search recipes and none of its reranking
machinery**. Everything below is additive, flag-gated, requires **no upstream
upgrade**, and changes nothing while its flag is off. Each experiment carries
an adopt/kill criterion measured on the preserved bench stores
(`~/nsbench-backups/2026-07-05/`, DMR mini-50 for iteration → full-500 for the
verdict; current baseline **65.2%** full-500 clean).

## E-1. Cross-encoder reranking of the fused top-k (highest leverage)

- **What**: graphiti ships `graphiti_core/cross_encoder/` with a
  `gemini_reranker_client` (same model family we already run — no new vendor).
  The client is usable **standalone**: rerank NS's own fused vector+graph
  top-k list right before response assembly, without touching either search
  leg.
- **Where**: after `_deduplicate_responses` in the search path; flag
  `RERANK_ENABLED` (+ `RERANK_TOP_K`, default e.g. 20 → rerank → return 10).
- **Cost**: one extra model call per flagged search (~0.3–0.8 s). Consider
  enabling only for `ask_memory` evidence passes and explicit "accuracy mode"
  first.
- **Adopt if**: full-500 DMR beats 65.2% by ≥2 pts with p50 search latency
  under an agreed ceiling; kill otherwise.

## E-2. Graph-proximity personalization (node-distance / BFS-origin)

- **What**: `EDGE_HYBRID_SEARCH_NODE_DISTANCE` recipe and
  `bfs_origin_node_uuids` filters rerank/scope results by graph distance from
  a focal node — the active project's entity or the user. Retrieval no
  pure-vector system can express: "relevant AND near what you're working on."
- **Where**: project-scoped recall + `get_project_context`; focal node =
  project entity resolved once per call. Flag `GRAPH_PROXIMITY_RERANK`.
- **Adopt if**: project-scoped queries improve on a project-tagged DMR slice
  without degrading global recall.

## E-3. Communities (GraphRAG-style thematic clusters)

- **What**: `add_episode(update_communities=True)` /
  `Graphiti.build_communities()` produce entity communities with LLM
  summaries. NS already exposes `get_graph_communities` — **the tables are
  empty because nothing ever builds them**.
- **Where**: build inside the dreaming sweep (off hot path), per pool, flag
  `DREAMING_COMMUNITIES`. Consumers: identity card substrate (thematic
  summaries instead of raw staged rows), `get_project_context` overviews,
  dreaming reflection input.
- **Cost**: LLM calls during sweeps only.
- **Adopt if**: card/context quality visibly improves (manual eval) at
  acceptable sweep cost; independent of DMR.

## E-4. `add_episode_bulk` for ingest

- **What**: book/document ingests currently write graph episodes one at a
  time via `graph_jobs`; graphiti has a bulk path.
- **Caution**: bulk skips parts of the incremental resolution/dedup pipeline —
  verify semantics parity on a sample corpus before adopting. Measure ingest
  wall-clock on a book-sized corpus.
- **Adopt if**: ≥2× ingest speedup with equivalent graph quality (node/edge
  counts + spot-check).

## E-5. (Noted, low priority) mem0 `reranker/` package

`mem0/mem0/reranker/` (cohere, huggingface, llm_reranker, zero_entropy)
overlaps with E-1; graphiti's Gemini reranker fits the stack better. Revisit
only if E-1 wins and we want a rerank option on the vector leg in isolation.

## Explicitly not pursued

- `EDGE_HYBRID_SEARCH_EPISODE_MENTIONS` — overlaps with NS's own
  reinforcement/salience mechanism (`times_derived`); redundant.
- mem0's 20+ alternative providers — optionality, not edge; NS is committed
  to Qdrant + Gemini.
- Absorbing mem0 entirely (the CBM-style decision) — a separate strategic
  conversation, not an experiment.

## Sequencing

E-1 first (biggest expected payoff, cheapest to measure). E-2 second if E-1
lands. E-3 rides the next dreaming-mode work. E-4 whenever ingest volume makes
it worth it. Prerequisite for all: none — the subtree prune keeps
`graphiti_core/` and `mem0/mem0/` whole.
