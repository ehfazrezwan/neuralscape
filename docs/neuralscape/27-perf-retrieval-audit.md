# 27 — Performance & Retrieval-Accuracy Audit (post-sprint, dev @ 3e42a89)

Six parallel code audits over every feature merged in the 2026-07-03/04 sprint (PRs #97–#118),
hunting for pathways that degrade latency, ranking/recall, extraction quality, or fail
silently. ~55 findings, deduplicated and ranked below. Companion to `25-…` (chronicle) and
`26-…` (benchmark findings). Fixes land as one worktree/PR per cluster, each defect
reproduced by a failing test before the fix.

Legend: **[A]** retrieval accuracy · **[L]** latency/throughput · **[S]** silent failure/data loss.

---

## Cluster 1 — Search recall (`fix/retrieval-recall`) — highest benchmark impact

1. **[A] BM25 lexical leg silently amputated mid-sprint.** Commit `7024a81` ("embed once")
   replaced mem0-v3 hybrid search (dense + lemmatized BM25 fusion + entity boosts + 4×
   over-fetch) with dense-only exactly-k `query_points` (`memory_service.py:1283-1289,
   1391-1397`). Sparse vectors are still *written* on every insert (`mem0/…/qdrant.py:196-201`)
   but never queried. Proper-noun recall (all benchmarks) takes the direct hit.
2. **[A] The vector/graph weave evicts scored hits.** Graph edges enter with `score=None` and
   are positionally interleaved 1:1 with ranked vector hits, then `combined[:limit]`
   (`memory_service.py:2053-2063, 4451-4462`) — at k=10, up to 5 ranked vector results are
   displaced by unranked relation strings.
3. **[A] Graph search returns bi-temporally invalidated edges.** `_do_graph_search` passes no
   `SearchFilters`; dream INVALIDATE/PRUNE/MERGE stamps `invalid_at` but nothing reads it —
   stale facts surface as graph rows and consume top-k slots (`memory_service.py:2275-2302`;
   `graph_patcher.py:374-381`).
4. **[A] `times_derived` boost: always-on, unbounded, no kill switch.** `1 + 0.05·ln(1+td)`
   re-sorts after vector scoring; at td=30 a 0.80-cosine wrong hit beats a 0.90 correct hit;
   scores exceed 1.0 (`memory_service.py:96, 1302, 1411, 1987-1988`).
5. **[S→A] Tombstone dedup black hole.** `_find_by_content_hash` has no `dream_tombstoned`
   must_not — re-storing a fact dreaming tombstoned is silently swallowed and stays
   unrecallable forever (`memory_service.py:1019-1031, 1417-1488`).
6. **[S] Stale hash after dream rewrite.** `_rewrite_content` never recomputes `payload["hash"]`
   — corrupts write-path dedup and the exact-dedup cron (which hard-deletes by hash group)
   (`consolidate.py:439-459`; `memory_service.py:4697-4711`).

## Cluster 2 — Search latency (`fix/search-latency`)

7. **[L] N+1 per-edge Gemini embed on every search.** `_enrich_graph_with_v2` embeds + queries
   Qdrant once per graph edge, sequentially, even when no v2 filter/field is requested —
   3–12s of the measured ~13s hybrid latency (`memory_service.py:2031-2036, 2106-2147`).
8. **[L] Graphiti edge similarity is an unindexed full Cypher cosine scan** over the read-set,
   serialized after the vector pass behind a silent 30s timeout; the query is embedded twice
   (`graphiti_core/search/search_utils.py:415-441`; `memory_service.py:570-592, 1998`).
9. **[L] Nothing is gathered**: embed → vector pools → graph → enrichment run strictly
   sequentially; ask high tier multiplies this ×6 (~65-80s/question).
10. **[L→A] Shared mutable recipe singleton.** `EDGE_HYBRID_SEARCH_RRF.limit` is mutated
    per-call from multiple threads; a concurrent delete clamps a live search's graph fan-out
    to 5 (`memory_service.py:2296-2298, 2426-2428, 4470-4473`).
11. **[L] Savings meter awaited inline on every read** (REST + MCP; Redis 2s timeouts on the
    response path; a meter exception converts a successful search into an error); SSE
    `publish_event` does sync Redis I/O on the API event loop (`main.py:1553, 1740-1748`;
    `savings_meter.py:296-316`; `event_stream.py:131-151`).
12. **[L] Every raw write pays a full hybrid search** (graph pass + edge enrichment included)
    as its idempotency check, and those internal searches pollute the dreaming recall traces
    (`worker.py:163-168`; `memory_service.py:2066-2072`).
13. **[L] `log_recall` fires ~5N+4 Redis writes per search even with dreaming disabled**;
    unbounded executor queue; `dreaming:dyn` hash grows forever (`memory_service.py:2069-2071`;
    `traces.py:39, 73-142`).
14. **[L] Unindexed payload filters** (`dream_tombstoned`, `visibility`, `scope`) on every
    hot-path query; `list_memories` doesn't exclude tombstones at all
    (`memory_service.py:1264-1267, 3152-3181`; `mem0/…/qdrant.py:158-176`).

## Cluster 3 — ask.py evidence quality (`fix/ask-evidence`)

15. **[A] The 120-row evidence budget drops the NEWEST memories** (sorted ascending, truncated
    from the front); timestamp-less graph rows sort first and eat the budget → stale answers
    and false abstentions (`ask.py:169-177`).
16. **[A] 500-char clip cuts passages mid-fact** — ingest passages default to 1500 chars, so
    two-thirds of any passage row is invisible to the answerer (`ask.py:109, 183-185`;
    `ingest/chunking.py:22`).
17. **[A] Cross-source duplicates waste rows** (graph-uuid vs vector-id twins across passes,
    deduped by id only) (`ask.py:323-326`).
18. **[A] Keyword pass: ANY-term substring match in scan order, capped at 5,000 points, rows
    promoted to the front and endorsed as exact matches** (`memory_service.py:3131-3150`;
    `ask.py:173-176, 226-229`).
19. **[A] Forced update-language pass dilutes evidence** with off-topic update-flavored rows on
    every low+ ask, at the cost of one full hybrid search (`ask.py:86, 364-365`).

## Cluster 4 — Write path (`fix/write-path`)

20. **[L] Gateway path: one HTTP embed round trip per fact, strictly serial** — a 40-fact
    conversation ≈ 4-16s serial embed inside one of 10 fast-worker slots; 25-item checkpoints
    identical (`config.py:628`; `mem0/…/openai.py:104-110`; `memory_service.py:1525-1580, 1742`).
21. **[S→A] Conversation path has no storage-level idempotency.** `_batch_store_facts` skips
    content-hash dedup; graph episodes have no idempotency key — any ARQ re-run/resubmission
    duplicates facts AND episodes (the 4,999 double-episode mechanism), and duplicates eat
    top-k (`memory_service.py:1645-1783`; `graphiti_memory.py:364-374`).
22. **[A] Whole conversation extracted in ONE LLM call** — no windowing/token guard; long
    sessions drop facts; post-#118 an extraction failure zeroes the session; the same
    unbounded text becomes one giant Graphiti episode (`prompts.py:162-186`;
    `memory_service.py:719-728, 786-799`).
23. **[A] mem0's UPDATE/DELETE resolution never runs** (both write paths bypass `m.add`) —
    contradictory facts accumulate as peers; the dedup cron then hard-deletes the *older* row,
    destroying temporal history (`memory_service.py:1033, 1657-1660, 4733-4811`).
24. **[L] Session summarizers run on the fast queue**, competing with latency-sensitive writes
    for the 10 slots (`worker.py:32-76, 1253-1261`).

## Cluster 5 — Dreaming safety (`fix/dreaming-safety`)

25. **[S] A hallucinated `contains_secret: true` hard-deletes at ANY confidence** — bypasses
    the gate; the one irreversible primitive (`consolidate.py:284-285, 383-388`).
26. **[S→A] REWRITE/REFRAME/MERGE-survivor rewrites destroy original text irreversibly** at any
    confidence; the merge prompt caps survivors at ~60 words — guaranteed fact/keyword loss
    (`consolidate.py:272-290, 408-459`; `extensions/dreaming/prompts.py:26-29`).
27. **[S] Silent no-op sweeps**: an LLM-exhausted call returns `""` → 0 actions → status
    "dreamt" + gate stamped; meanwhile dreaming-enabled permanently disables semantic dedup —
    broken LLM = consolidation silently off forever (`sweep.py:107-130, 438-440`;
    `worker.py:1014-1021`).
28. **[A] Node-scoped graph invalidation**: tombstoning one memory invalidates every edge on
    its entity nodes, including edges asserted by live memories (`graph_patcher.py:374-381`).
29. **[L] Full-collection scroll runs before any gate** — every pool's rows materialized in
    worker RAM even when all pools are quiet (`sweep.py:163`; `consolidate.py:91-151`).
30. **[S] Read-modify-write metadata patches race** (tombstone vs times_derived bump — whole
    `metadata` dict replacement) (`consolidate.py:462-536`; `memory_service.py:1504-1521`).

## Cluster 6 — Plugin/MCP surface (`fix/plugin-mcp`)

31. **[L] File Read Gate: synchronous node spawn on every Read** + a 500-row full-payload NS
    fetch per first-read of any file >1.5KB (`hooks.json:5-16`; `read-gate.ts:41-48`).
32. **[A] Read Gate basename-only matching denies legitimate reads** and substitutes stale
    memory titles for real file content (`read-gate.ts:110-129, 201-236`).
33. **[A] Index-first steering**: "NEVER fetch full details without filtering first" — filtering
    happens on lossy 10-word titles; session-start injection flipped to titles-only with
    "never expand speculatively" (`mcp_server.py:109-113, 188-189`; `disclosure.ts:240`).
34. **[S] Redaction fail-closed truncates to end-of-string** after any stray literal
    `<private>` (`utils.ts:259-264`); **stop-hook flush advances the transcript offset even
    when every POST failed** — permanent conversation loss (`session-end.ts:52-57`).
35. **[L] Two `MemoryService` instances per API process**; MCP's cold-inits on first tool call
    (`main.py:85`; `mcp_server.py:40`); per-call Redis clients in `get_card`/`schedule_dream`.
36. **[A] Adapter categories leak into every MCP enum (13→30) and default to global scope**;
    a queued job whose adapter failed to register silently falls back to the default taxonomy
    (`adapters/__init__.py:30-48`; `schemas.py:141-147`; `adapters/base.py:149-158`).

## Exonerated (audited clean — don't re-litigate)

- Extraction prompt: still the core-13 menu; adapter taxonomy growth never reached the
  classifier. Custom instructions are a byte-for-byte no-op when unset.
- `index_only`: boundary-only mapping after the identical full search, REST and MCP.
- Salience tie-break at k=0 default: byte-identical path, no Redis I/O; reads never write
  Qdrant; settling/gates are cron-only; surprisal is cron-only and bounded.
- Vector-side tombstone exclusion: proper index-time must_not (top-k safe); raw-path
  content-hash dedup; graph group_id scoping (native, no cross-user leak).
- Token meter encoder: cached singleton, threaded, kill-switch honored. SSE route itself,
  auth middleware, checkpoint dedup semantics, webhook placement: clean.
- graphify import (~1ms, heavy modules lazy); OKF fully off the hot path.
