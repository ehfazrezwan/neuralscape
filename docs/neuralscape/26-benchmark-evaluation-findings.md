# 26 — Benchmark Evaluation Findings (2026-07-04 battery, first pass)

Everything learned from the first attempt to score feature-frozen Neuralscape against the
competitive benchmark union. Companion to `24-competitive-roadmap.md` (§E5) and the session
chronicle (`25-…`). Raw results and manifests live on the `bench/battery-2026-07` branch
(`neuralscape-bench/results/`, incl. `landed-store-verification.md`).

**Status: the battery is PAUSED at a maintainer checkpoint.** Scores below are labeled by
run condition; several are floors or void, not measurements. Nothing here is a publishable
number yet — that is the point of this document.

---

## 1. The suites and where they stand

Config for all runs: `reasoning_level=high`, k=10, judge = Gemini (temp 0), seed 42,
dreaming OFF (baseline), isolated stack.

| Suite | Status | Score | Competitor context (self-reported, their configs) |
|---|---|---|---|
| DMR (500q) | run-1 contended 57.0% → **quiet fully-landed 68.2%** | 68.2% is the working clean baseline (double-enriched-graph confound noted) | MemGPT 93.4, Zep 94.8 |
| BEAM (400q) | baseline complete | 27.8% (R@10 column void — attribution incompatibility) | Honcho publishes per-tier results |
| ConvoMem (1047q) | **VOID** — qa-id collision defect | 13.9→14.4 (A/B control only) | MemPalace repo reproductions |
| LoCoMo | interrupted mid-run (battery stop) | — | mem0 66.9 / mem0ᵍ 68.4, Honcho 89.9 |
| MemBench | interrupted mid-ingest | — | MemPalace repo reproductions |
| LongMemEval_S | **HELD** (the $89 suite) pending fix-wave approval | — | Honcho 90.4, Zep ~71–79, MemPalace 96.6 (R@5 retrieval only) |

## 2. The landed-store verification (the day's most important measurement)

The maintainer hypothesized the low scores were partly "answers ran before the async store
finished landing." A controlled test — answer+judge re-runs with **no re-ingest** against the
verified fully-landed, quiet store — split the hypothesis cleanly:

- **Store-state-at-answer-time: CONFIRMED major driver.** DMR 57.0% → **68.2%** (+11.2pp,
  net +56 corrected answers out of 500) with identical config. About a third of the gap to
  Zep/MemGPT was a measurement artifact of racing the async write path.
- **Concurrent load alone: ruled out as major.** ConvoMem was the natural A/B (same store
  both runs; its run-1 answers executed during peak cross-suite contention) and moved only
  +0.5pp.
- Evidence quality: zero LLM backoff events in the quiet runs; quiet-run latency was *higher*
  (p50 9.0s → 11.3s), so gains are not a speed artifact. Ingestion completeness verified for
  every benchmark user (DMR min/p50/max = 34/47/66 memories per conversation; zero
  under-ingested users).

## 3. Defects found in the harness/store (fix wave, proposed)

1. **ConvoMem qa-id collision** — ids collide across categories; resume/judging corrupted.
   Fix: category-namespaced ids. Until then the ConvoMem absolute score is void.
2. **BEAM retrieval attribution incompatibility** — R@10 = 0.0% across 400 questions is a
   flat-zero plumbing signature: the lexical session-attribution approach does not fit BEAM's
   document format. The accuracy column is real; the retrieval column must be marked
   incompatible or reworked.
3. **Targeted deletes must be graph-aware** — deleting run-1's DMR vectors left Neo4j holding
   both ingest waves (4,999 episodes → the re-run answered against a double-enriched graph).
   Confound is favorable-direction-unknown; a fresh-stack DMR run resolves it (~$4).
4. **Ingest→answer drain barrier** — the harness now must gate the answer phase on full task
   completion + queue drain (`queue_status.caught_up` — NS's own C4 tool). Partially landed
   during verification (resume keys on `(qa_id, qtype)`, backoff telemetry).
5. **One suite at a time, idle machine.** Cross-suite phase concurrency plus host swap
   pressure SIGKILLed the runner twice mid-run (7/8 GB swap). Scoring is a measurement
   activity: sequential per suite, nothing else on the host, Docling stopped.

## 4. Product findings (the real value of the battery so far)

1. **`occurred_at` is the missing envelope field.** NS stamps `created_at` at write time and
   exposes no event-time override — so bulk-ingested histories all "happened today" and
   BEAM's event_ordering (0%), contradiction_resolution (0%), and temporal_reasoning (12.5%)
   are structurally unanswerable, exactly as pre-registered. Graphiti models event time
   internally; exposing it on the write path unlocks temporal question types across every
   suite *and* is a genuine product feature for any historical ingestion. Highest-value
   single change to come out of the battery. (Breaks feature freeze; maintainer decision.)
2. **The evidence clip hides retrieved facts.** `ask.py` clips each evidence row at
   `_EVIDENCE_CONTENT_CLIP = 500` chars (cap 120 rows) — facts past the boundary are
   retrieved but invisible to the answerer, masquerading as "reasoning misses" in failure
   taxonomies.
3. **Abstention works under fire.** BEAM abstention: 90%. The strict abstention contract
   (zero evidence → "I don't know" with zero LLM calls; citations validated against retrieved
   evidence) is a real differentiator — most systems bluff.
4. **Summarization-type questions punish distillation-based memory** (BEAM summarization
   2.5%): whole-corpus summaries want full transcripts, which NS deliberately does not keep
   as its primary representation. Either route such questions to session summaries
   (E3 slots) or accept and document the trade.
5. **Structured file-path metadata is missing from the envelope** (found during the File
   Read Gate build, confirmed relevant here): memories carry file references only in prose,
   which forces lexical attribution — the same weakness behind finding #2 in §3.
6. **Hybrid graph search latency**: `POST /v1/search` measured ~13s live (Graphiti pass) vs
   ~70ms for list reads. Fine for chat recall; disqualifying for hooks and a drag on
   benchmark answering loops. Optimization candidate.
7. **`index_only` is exonerated**: it maps results to compact rows strictly at the API
   boundary after the identical vector+graph search; benchmarks used full-content evidence.
   The salience tie-breaker (k=0) is byte-identical off; the reinforcement boost is inert at
   `times_derived=1`. Retrieval economics did not degrade retrieval.

## 5. The proposed path (pending maintainer approval)

1. Fix wave: §3 items + optionally §4.1 (`occurred_at`) and §4.2 (raise the clip).
2. Fresh-stack clean DMR baseline; re-run cheap suites through the fixed harness.
3. Release LongMemEval_S ($89) through the proven pipeline.
4. The differentiator experiment: dream the benchmark store, re-run answer+judge, and report
   pre/post-consolidation deltas — the column none of the six suites' publishers can produce.

## 6. Honest framing for whenever numbers are published

Competitor figures are self-reported on their own configurations (answer model, retrieval
depth, judge all differ). NS's own first numbers were depressed by measurement artifacts we
found, documented, and fixed — the methodology (drain barriers, controlled A/Bs, labeled
run conditions, signed confounds) is itself the credibility asset. Publish the harness, the
manifests, and the caveats with the scores.
