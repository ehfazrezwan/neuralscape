# neuralscape-bench

End-to-end performance benchmark + dashboard for the Neuralscape memory service.

Drives the REST API and measures four metric families, with a one-shot A/B that
compares the untuned `dev` baseline against a tuned candidate branch.

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

## Tests

```bash
uv run pytest -q   # pure stats, client request/poll (mocked transport), dashboard API
```
