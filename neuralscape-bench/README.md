# neuralscape-bench

End-to-end performance benchmark + dashboard for the Neuralscape memory service.

Drives the REST API and measures four metric families, with a one-shot A/B that
compares the untuned `dev` baseline against a tuned candidate branch.

> **Status: benchmark runs are paused.** The harnesses, their tests and CI are
> maintained, but no new measurements are being taken or published. Everything
> under `results/` is a historical record from older Neuralscape commits (see
> [`results/README.md`](results/README.md)); none of it is a publishable number.
> Do not quote a figure from this directory without that context.

## Module map

All paths are relative to `neuralscape-bench/`.

| Path | What it is |
|---|---|
| `neuralscape_bench/runner.py` | Single-run latency/throughput engine (`python -m neuralscape_bench.runner`). |
| `neuralscape_bench/stress.py` | Multi-user concurrency stress test. |
| `neuralscape_bench/orchestrator.py` | One-shot A/B of a baseline stack against a candidate branch. |
| `neuralscape_bench/dashboard.py`, `static/` | FastAPI dashboard over `results/*.json`. |
| `neuralscape_bench/client.py`, `models.py` | Async REST client; pure data models and statistics. |
| `neuralscape_bench/accuracy/` | Memory-accuracy battery: `run.py` (CLI), `stack.py` (isolated stack up/down), `ingest.py`, `answer.py`, `judge.py`, `metrics.py`, `report.py`, `costs.py`, `download.py`, `sampling.py`, `schema.py`, `manifest.py` (resume state). |
| `neuralscape_bench/accuracy/suites/` | One loader per suite: `locomo`, `longmemeval`, `dmr`, `beam`, `convomem`, `membench`. |
| `neuralscape_bench/service_baseline/` | Mechanics-only service baseline harness (fake provider, closed-loop driver, run recorder). Usage and safety rules: [`scripts/service_baseline/README.md`](scripts/service_baseline/README.md). |
| `neuralscape_bench/run_manifest.py` | Deterministic run-record models (evidence about a run; not an authorization). |
| `icebench/` | Code-intelligence benchmark: `run.py` (CLI: `corpora`, `index`, `query`, `score`, `report`), `corpora.py`, `corpus_filters.py`, `rail.py` (memory cap + timeout), `schema.py`, `adapters/` (Neuralscape engines), `systems/` (standalone competitor adapters; see [`icebench/systems/CAPABILITIES.md`](icebench/systems/CAPABILITIES.md)), `trackq/` (query generation, oracle, scoring), `tokensave/` (token-savings navigation benchmark), `report/` (Markdown + HTML report). |
| `trackb/` | Track B harnesses that follow a competitor's published method: `longmemeval_ref/`, `mem0_locomo/`, and `mem0_tracka/` (a mem0 control run under the Track A settings). |
| `scripts/` | CI helpers (`ci_run_suite.sh`, `ci_skip_gate.py`, `ci_wheel_smoke.py`) and `service_baseline/` scripts. |
| `ci/expected_skips.json` | Declared skips for the skip gate. |
| `tests/`, `icebench/**/tests`, `trackb/**/tests` | The test suites. |

## Metrics

- **Write latency** — `POST /v1/memories/raw`: enqueue (202) latency *and*
  end-to-end (enqueue → task `completed`). The headline: on the untuned stack a
  write completes only after an inline ~2-min Graphiti `graph.add`; the tuned
  stack returns after the fast vector insert and defers graph work.
- **Read latency** — `search` / `graph_search` / `list` / `context`, p50/p95/p99.
- **Read-under-write contention** — read p95 with no writers vs under sustained
  concurrent write load.
- **Throughput + error rate** — reads/sec and writes/sec at a target concurrency.

> Scope note: this harness drives the **REST** API, so it does *not* exercise the
> MCP event-loop fix (REST routes already offload to a thread pool). It measures
> the write-path + worker-isolation improvements. Each run records this in its notes.

## Install

```bash
cd neuralscape-bench
uv sync --extra dev
```

## Run a single benchmark

```bash
# Against the live local stack (no auth needed in dev):
uv run python -m neuralscape_bench.runner \
  --target http://localhost:8199 --label dev-live --profile light --live-baseline
```

Results are written to `results/<label>-<ts>.json`. `--profile full` uses larger
samples + a long poll timeout (captures true end-to-end write latency; slow on the
untuned stack and uses Gemini quota). Bench data is written under a throwaway
`bench-<runid>` user and bulk-deleted afterward.

## One-shot A/B (untuned vs tuned)

Builds an isolated stack per branch on offset ports (only the API port is
published; backing stores stay internal per compose project), benchmarks both,
and writes `results/compare-<ts>.json`.

```bash
# Use the already-running stack as the live baseline; build only the candidate:
uv run python -m neuralscape_bench.orchestrator \
  --candidate-branch perf/write-read-isolation \
  --baseline-url http://localhost:8199 --profile full

# Or build BOTH baseline (dev) and candidate fresh, with identical seed data:
uv run python -m neuralscape_bench.orchestrator \
  --candidate-branch perf/write-read-isolation --profile full --env-file ../.env
```

Requires Docker + a `.env` with `GOOGLE_API_KEY` / `NEO4J_PASSWORD` (built stacks
run the real graph path). `--no-teardown` leaves stacks up for inspection.

## Multi-user stress test

Simulates **N distinct users** hitting the service concurrently with a read/write
mix for a fixed duration — the "can it handle concurrent users" scenario. Reports
aggregate throughput + error rate, overall read/write latency percentiles, and
crucially **per-user p95 + a fairness measure** (does one user's load starve
another?). Writes are enqueue-only (202 latency under load), not polled to
completion.

```bash
uv run python -m neuralscape_bench.stress \
  --target http://localhost:8199 --label dev-stress \
  --users 20 --duration 30 --concurrency 3 --profile full
```

Each simulated user gets its own `stress-<runid>-uN` namespace (seeded + cleaned
up). `fairness.read_p95_spread` = max/min per-user read p95 (1.0 = perfectly
fair; large = some users starved); `read_p95_cv` is the coefficient of variation.
Results render in the dashboard with a per-user p95 bar chart.

> Auth note: distinct users are simulated via distinct `user_id` (local dev needs
> no auth). Under per-user-token auth you'd issue a token per user — a future knob.

## Dashboard

```bash
uv run python -m neuralscape_bench.dashboard   # http://localhost:9100
```

Pick a run to see its charts; select a second run to compare; A/B (`compare-*.json`)
files show baseline-vs-candidate deltas. Buttons trigger a single run or an A/B.

## Memory-accuracy battery (roadmap E5)

`neuralscape_bench/accuracy/` runs the published competitor suites against an
**isolated** NS stack and produces LLM-judged accuracy + retrieval R@k per
suite and per question type, with a markdown battery table comparing against
the competitors' *self-reported* figures.

| Suite | Source (canonical) | Size (as configured) | Status |
|---|---|---|---|
| LoCoMo | `github.com/snap-research/locomo` (`data/locomo10.json`, ~2.8 MB) | 10 convs / 272 sessions / 1,986 QA | fetch verified |
| LongMemEval_S | HF `xiaowu0162/longmemeval-cleaned` (`longmemeval_s_cleaned.json`, ~277 MB) | 500 questions / ~23.9k sessions | fetch verified |
| LongMemEval_M | HF `xiaowu0162/longmemeval-cleaned` (`longmemeval_m_cleaned.json`, **2.7 GB**) | 500 questions / ~250k sessions | fetcher implemented (same host/schema as S; not downloaded by default) |
| DMR | HF `MemGPT/MSC-Self-Instruct` (`msc_self_instruct.jsonl`, ~8.5 MB, Apache-2.0) | 500 convs / 2,500 sessions / 500 QA | fetch verified |
| BEAM | `github.com/vectorize-io/agent-memory-benchmark` (`data/beam/<tier>/`) | 100k tier: 20 users / 170 docs / 400 QA (tiers to 10M) | fetch verified |
| ConvoMem | HF `Salesforce/ConvoMem` (`core_benchmark/evidence_questions/`) | default subset: 2 files × 6 categories → ~1,047 QA (full corpus ~75k QA) | fetch verified |
| MemBench | `github.com/import-myself/Membench` (`MemData/FirstAgent/`) | default 3 categories → 4,500 sessions / 3,500 QA (`--membench-categories all` for the rest) | fetch verified |

Datasets download at run time into `datasets/` (**gitignored — never
committed**). Raw per-question outputs (contain conversation text) live under
`results/raw/` (gitignored). Only aggregate `results/accuracy-*.json` +
battery markdown are committed, and those are historical (see
[`results/README.md`](results/README.md)).

Suite notes that follow from the current code:

- **BEAM retrieval recall is reported as incompatible**, not as a number. Its
  gold ids are user-level while its documents are whole multi-turn
  conversations, so session attribution does not apply; reports print
  `n/a (incompatible: ...)` for that column (`retrieval_incompatible` in
  `accuracy/suites/beam.py`). BEAM answer accuracy is still judged.
- **ConvoMem question ids are namespaced per category** so ids from different
  categories cannot collide (this fixed a resume/judging defect).
- **DMR:** speaker identity is no longer encoded twice in turn content, and
  every session gets a deterministic `occurred_at` anchored to a fixed time.
- **Run manifest (pending #542):** an opt-in `--manifest` option on the accuracy
  runner that records a `run_manifest` for the run is in review and not yet on `dev`.

> BEAM note: Honcho publishes BEAM results (github.com/plastic-labs/honcho-benchmarks
> holds their result JSONs), but that repo contains no dataset/harness — the
> runnable dataset is the vectorize-io repo above (HF mirror `Mohammadta/BEAM`).

### Running the full battery

```bash
# 1. offline + free: fetch datasets, verify shapes, estimate token costs
uv run python -m neuralscape_bench.accuracy.run --suite all --phase fetch
uv run python -m neuralscape_bench.accuracy.run --suite all --phase estimate

# 2. bring up an ISOLATED stack (own compose project + volumes, offset port —
#    never the livestack; Docker ops run under the /tmp/ns-e2e.lock protocol)
uv run python -m neuralscape_bench.accuracy.run --stack up          # :8398

# 3. per suite: ingest → answer → judge → report (each resumable; interrupt-safe)
BENCH_TOKEN=<stack NEURALSCAPE_API_KEY, if auth is enabled> \
uv run python -m neuralscape_bench.accuracy.run --suite locomo \
  --phase ingest --phase answer --phase judge --phase report \
  --target http://localhost:8398 --concurrency 2

# stratified sample instead of the full suite (seed recorded in provenance):
#   --sample 100 --seed 42
# retrieval depth / answering tier:  --k 10 --reasoning-level high

uv run python -m neuralscape_bench.accuracy.run --stack down
```

Ingestion uses the conversation-extraction path (`POST /v1/memories`,
session-sized batches, `run_id=<session_id>`, per-conversation users
`bench-<suite>-<conv_id>`), polling task completion — never re-storing.
Answers come from `POST /v1/ask` (default `reasoning_level=high`); the
retrieval probe is `POST /v1/search` top-k with lexical session attribution
(see `metrics.py` — NS returns distilled facts, not turn ids, so R@k is
attribution-based and labeled as such). Judging is Gemini
(`--judge-model`, temperature 0) with the standard question/gold/answer →
correct/incorrect protocol; abstention questions grade "no such info" as
correct. 429/5xx everywhere get exponential backoff — throttle, don't fail.

**Dreaming/consolidation is OFF for the baseline run** (pure
write→recall→ask). A post-dream re-run is the NS-differentiator experiment
and is deliberately left as future work.

### Estimated full-run token costs (analytic estimates)

These are model-based estimates, not measured spend. They predate the
historical runs in `results/`, and a real run's cost can differ.

From `--phase estimate` over the real datasets (assumptions: 4 chars/token;
ingest = 3.5 LLM input passes per session token (NS extraction + Graphiti
enrichment) + 600 prompt tokens/call, output 12% of input; ask(high) ≈ 12k
in / 500 out per question; judge ≈ 450/40; Gemini 2.5 Flash list pricing
$0.30/M in, $2.50/M out — all overridable in `costs.py`):

| Suite | Input tokens | Output tokens | Est. cost |
|---|---|---|---|
| LoCoMo (full) | ~25.6 M | ~1.1 M | ~$10.4 |
| LongMemEval_S (full 500) | ~234.6 M | ~7.6 M | ~$89.4 |
| DMR (full 500) | ~10.9 M | ~0.4 M | ~$4.2 |
| BEAM (100k tier) | ~15.8 M | ~0.6 M | ~$6.2 |
| ConvoMem (default subset) | ~18.8 M | ~0.7 M | ~$7.5 |
| MemBench (default 3 categories) | ~62.8 M | ~2.5 M | ~$25.0 |
| **Battery total** | **~368 M** | **~12.9 M** | **~$143** |

LongMemEval_S dominates because every one of its 500 questions carries its
own ~115k-token haystack that must be ingested under a separate user.
`--sample N` scales ingest + answering roughly linearly for the per-question-
haystack suites (LME, DMR, ConvoMem, MemBench). LongMemEval_M is ~10× S on
ingest — run it only after S numbers justify it. Wall-clock is dominated by
extraction/enrichment throughput of the single-stack worker, not tokens.

## ICEBench (code intelligence)

`icebench/run.py` drives the `corpora -> index -> query -> score -> report`
flow. By default the indexed corpus copy has **docstrings stripped**, so the
`nl_locate` queries (which are docstrings) are not present verbatim in what the
systems index. `--no-strip-docstrings` (on `index` and `query`, which must
agree) indexes the corpus unmodified; the run is then labelled as a leaked
upper bound and the report carries that protocol label. See
`docs/ice-benchmark/ice-fable-review.md` section 3.1 for why this exists.

Competitor adapters live in `icebench/systems/`; what each one calls is in
[`icebench/systems/CAPABILITIES.md`](icebench/systems/CAPABILITIES.md).

## Tests

```bash
uv run pytest -q   # pure stats, client request/poll (mocked transport), dashboard API,
                   # accuracy suite parsers/metrics/judge/manifests (fixture-based)
```

### CI coverage (`.github/workflows/bench-ci.yml`)

The `bench-ci` workflow runs on pull requests and pushes that touch
`neuralscape-bench/**` (documentation-only changes included). It builds the
wheel and runs the suites against the **installed wheel**, not the source tree,
on the host and again in a container with no network:

- copied suites: `tests/`, `icebench/tests`, `icebench/trackq/tests`,
  `icebench/tokensave/tests`;
- suites run from the installed package (`--pyargs`): `icebench/report`, the
  `cbm_standalone` and `graphify_standalone` adapter tests, and the `trackb`
  suites (`longmemeval_ref` except its loader test, which needs the downloaded
  dataset; `mem0_locomo`; `mem0_tracka`).

`scripts/ci_run_suite.sh` is the single list of suites; `ci_wheel_smoke.py`
asserts that every suite ran.

**Skip gate.** `scripts/ci_skip_gate.py` reads the JUnit report and fails the
run on any failure or error, on a skip that is not declared in
`ci/expected_skips.json`, on a declared skip with a different reason, and on a
declared skip that no longer happens. If you add a test that must skip in CI
(for example one needing a binary the image does not carry), add an entry with
a reason and a `why`; do not weaken the gate. Live-binary smokes for the
competitor adapters are expected skips for this reason.

## Third-party files

`static/chart.umd.min.js` is a vendored copy of Chart.js 4.4.1 (MIT licence;
the file header carries the notice, and it bundles `@kurkle/color` 0.3.2, also
MIT). It is not our code; update it by replacing the file, not by editing it.
