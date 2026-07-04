# DMR run-3 phase 1: fresh-stack rebuild + ingest + drain verification

Clean single-ingest fresh-stack DMR run to resolve the run-2 confound
(deficient run-1 vector store vs double-enriched graph — see
`landed-store-verification.md`). **Phase 1 only** — answer/judge deliberately
deferred to a quiet machine.

## Stack

- Compose project `nsbench-accuracy` on **:8398** (livestack untouched),
  compose files: bench-battery worktree `docker-compose.yml` +
  `neuralscape-bench/docker-compose.bench.yml`.
- Prior stack destroyed **with volumes** at 2026-07-04 ~21:47Z
  (`down -v`; all `nsbench-accuracy_*` volumes confirmed gone) — no run-1/run-2
  vectors, no doubled graph enrichment.
- Image rebuilt from the merged tree: **bench branch merge commit `5753803`**
  (merges `origin/dev` @ `6c23bab`, i.e. the PR #120–#124 fix wave, under the
  unchanged harness). Image `sha256:ed5e4ac3c005…`; in-container marker checks:
  `grep -c "def content_hash" memory_service.py` → 1, `telemetry.py` present.
- Config = doc-26 baseline: `DREAMING_ENABLED=false` (verified in API + graph
  worker env), `DOCLING_ENABLED=false` (docling container stopped — unused for
  DMR), `AUTH_PROVIDER=token`, models `gemini-3.1-flash-lite` (fallback
  `gemini-2.5-flash`), embedder `gemini-embedding-001`. `/health`: redis /
  vector_store / graph_store all ok.

## Ingest (2026-07-04 21:51:40Z → 23:16:56Z)

Same invocation family as runs 1–2: `--suite dmr --phase ingest
--target http://localhost:8398 --concurrency 6 --seed 42` (full 500-conv /
2,500-session MSC-Self-Instruct dataset; prior manifest archived as
`state/ingest-dmr-http-localhost-8398.run2.json`).

- Main pass: **2498/2500 sessions in 4,692 s wall** (runs 1–2 took
  6,855–? s for the same pass — the fix-wave write path is ~1.5× faster).
  Sum of per-session elapsed 27,967 s; est. 953k tokens.
- 2 sessions (`valid_231/s5`, `valid_589/s5`) failed on a transient Gemini
  **503 UNAVAILABLE** during batch store. Retry note: the deterministic
  job-id dedup (`enqueue_store`) makes an identical resubmit coalesce onto
  the *failed* ARQ result while `arq:result:<job_id>` lives, so resume passes
  no-op with an instant "failed" until that key expires (~1h keep_result).
  Cleared the two stale result keys in the bench Redis and re-ran the resume
  pass (`--concurrency 2 --poll-timeout 600`): both stored.
- **Final: 2500/2500 sessions stored, 0 failed** — first run to land the full
  dataset (runs 1–2 both stalled at 2499; the historical `valid_401/s3`
  failure did not recur).

## Drain barrier

`/v1/queue/status`: `{"main": 0, "graph": 0, "ingest": 0}, caught_up: true`
across 3 polls 20 s apart after the resume pass; `arq:in-progress:*` = 0;
graph worker idle after finishing the retried sessions' summaries at 23:14Z.

## Landed-store verification (Qdrant facet + Neo4j)

| metric | value |
|---|---|
| bench-dmr users | **500/500** |
| memories/user min / p50 / max | **14 / 27 / 45** |
| users < 5 memories | **0** |
| total DMR memories | 13,708 |
| non-DMR users in collection | 0 (clean stack) |
| Neo4j `Episodic` nodes | **2,500** (exactly 1/session — single enrichment pass; run-2 had 4,999) |
| Neo4j `Entity` nodes | 6,444 |

Per-user counts are ~40% lower than run-2's store (34/47/66): expected — the
merged write path adds content-hash dedup, so near-duplicate distilled facts
no longer land twice. Distribution shape is healthy (no zero/starved user).

## Anomalies

1. Graph worker crashed once at container start (fresh-stack race: both
   workers raced `create_collection` on `neuralscape_memories`); restarted by
   compose at 21:50:31Z, clean since. Benign, pre-ingest.
2. The two Gemini 503s + the stale-failed-result retry trap above (harness
   resume interacts poorly with deterministic-job-id dedup after a *failed*
   job; operational workaround only — no service code touched).
3. Worker error-level log lines during ingest: 2 (the two 503s). Graph
   worker: 0 after restart.

## Phase 2 (answer + judge) — command for the orchestrator

Run on a quiet machine, from `neuralscape-bench/` in the bench-battery
worktree. **Archive the raw outputs first or resume will skip all 500
questions**:

```bash
mv results/raw/answers-dmr.jsonl results/raw/answers-dmr.run2-quiet.jsonl
mv results/raw/judged-dmr.jsonl  results/raw/judged-dmr.run2-quiet.jsonl

BENCH_TOKEN=<NEURALSCAPE_API_KEY from ../.env> \
GOOGLE_API_KEY=<from ../.env> \
uv run python -m neuralscape_bench.accuracy.run --suite dmr \
  --phase answer --phase judge --phase report \
  --target http://localhost:8398 --concurrency 4 --k 10 \
  --reasoning-level high --judge-model gemini-2.5-flash --seed 42
```

(Identical config to runs 1–2: k=10, reasoning high, judge gemini-2.5-flash
temp 0, seed 42, answering concurrency 4.)
