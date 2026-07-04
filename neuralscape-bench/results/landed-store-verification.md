# Landed-store verification: were battery scores depressed by contention / late-landing memories?

**Question.** Run-1 answer phases executed immediately after (or during) heavy concurrent
ingest, with async writes still in flight and the host under swap pressure. How many
points of the DMR and ConvoMem gaps were timing/contention artifacts vs genuine
pipeline quality?

**Method.** Battery stopped; bench stack (`nsbench-accuracy`, :8398) verified quiet
(all three ARQ queues at 0 before and at every poll during the runs; no other runner
processes). Answer+judge re-run only — **no re-ingest** — with config identical to
run-1: k=10, reasoning_level=high, judge gemini-2.5-flash (temp 0, thinkingBudget 0),
seed 42, full question sets, answering concurrency 4.

## Ingestion-completeness sanity (pre-run)

Qdrant facet over `neuralscape_memories`:

| suite | bench users | memories/user min / p50 / max | users < 5 memories |
|---|---|---|---|
| bench-dmr | 500/500 | 34 / 47 / 66 | 0 |
| bench-convomem | 200/200 | 13 / 50 / 82 | 0 |

Every sampled DMR user had all 5 sessions (`run_id` s1–s5) present. The DMR answer
phase targets **run-2's store**: run-2 ingest completed (2499/2499 manifest sessions;
one dataset session of 2500 failed in both runs), and run-1's DMR vectors no longer
exist — they were deleted before the run-2 re-ingest (BEAM/ConvoMem vectors from the
same collection survive, so this was a targeted delete, not a collection wipe).

## Results

### DMR (500 questions, LLM-judge)

| | run-1 (contended) | quiet re-run | Δ |
|---|---|---|---|
| overall accuracy | **57.0%** (285/500) | **68.2%** (341/500) | **+11.2 pp** |

Paired per-question: 98 flipped incorrect→correct, 42 correct→incorrect (net +56).
Median memories_considered 27 → 30. Single question type (dmr), so no per-type split.

### ConvoMem (1047 questions, LLM-judge)

| type | run-1 (contended) | quiet re-run | Δ |
|---|---|---|---|
| overall | **13.94%** | **14.42%** | **+0.5 pp** |
| abstention_evidence | 9.0% | 9.0% | 0.0 |
| assistant_facts_evidence | 4.0% | 5.0% | +1.0 |
| changing_evidence | 0.5% | 1.0% | +0.5 |
| implicit_connection_evidence | 54.1% | 58.1% | +4.1 |
| preference_evidence | 41.8% | 41.2% | −0.5 |
| user_evidence | 1.6% | 1.6% | 0.0 |
| retrieval R@10 | 0.191 | 0.191 | 0.0 |

Paired flips: 16 up, 11 down — symmetric judge/answer noise. The known
id-collision defect (cross-category stem collisions; see the run-VOID note on
`accuracy-convomem-20260704T111741Z.json`) was deliberately **held constant** so the
re-run isolates the contention variable only. The absolute ConvoMem number remains
invalid until the category-namespaced-id fix lands.

## Run hygiene / residual noise

- Answer-phase backoff/retry events: **0/500** (DMR) and **0/1047** (ConvoMem) — far
  under the 1% noise threshold. Judge unparseable: 0 in both.
- Queues stayed at 0 at every poll; nothing else ran against the stack.
- Median ask latency was *higher* in the quiet re-runs (DMR 9.0s→11.3s,
  ConvoMem 20.5s→21.3s), so the score gains are not a latency/timeout artifact.
- The answer runner was OOM/SIGKILLed twice mid-ConvoMem by host swap pressure
  (7 GB/8 GB swap in use); runs were resumed losslessly. Resume for both answer and
  judge phases now keys on `(qa_id, qtype)` — plain `qa_id` is non-unique under the
  ConvoMem collision and would have silently skipped items.

## Forensic timeline notes

- ConvoMem is the clean A/B: its store is **identical** in both runs (vectors from
  09:03–10:12Z untouched), and its run-1 answer window (10:12–11:14Z) overlapped the
  entire DMR run-2 ingest (until 11:02Z) plus LoCoMo ingest+answering — peak
  contention. Quiet re-run moved it +0.5 pp. **Pure answer-time contention ≈ noise.**
- DMR is confounded: run-1 answered against run-1's (since-deleted) vector store; the
  re-run answered against run-2's fresh ingest *plus* a double-enriched graph
  (4,999 DMR `Episodic` nodes — run-1's 2,500 were never deleted from Neo4j, so run-2
  added a second enrichment pass). The +11.2 pp therefore measures *store state*, not
  answer-time load.

## Verdict

**The "answers ran while the stack was contended" half of the hypothesis is ruled out
as a major driver** — ConvoMem, the controlled comparison, moved +0.5 pp under quiet
conditions. **The "store state at answer time" half is confirmed for DMR**: against a
verified fully-landed store the same questions score +11.2 pp (57.0% → 68.2%), i.e.
roughly a third of the gap to Zep/MemGPT (93–95%) was store-state artifact and the
remaining ~25–27 pp is genuine pipeline quality. Caveats: the DMR gain cannot be fully
split between (a) run-1's deficient/incomplete vector store and (b) the re-run's
double-enriched graph; a clean single-ingest fresh-stack DMR run would resolve that.
ConvoMem's 13.9→14.4% stays defect-dominated (id collisions) — re-score only after
the fix wave.

*Config: identical to run-1 (k=10, reasoning=high, judge gemini-2.5-flash, seed 42,
full sets). Result JSONs: `accuracy-dmr-20260704T121157Z.json`,
`accuracy-convomem-20260704T143724Z.json`.*
