# Service baseline harness

Mechanics-only measurements of the memory service in an isolated compose project: a fake
provider with fixed latency, a closed-loop driver inside the network, container resource
sampling and a `RunManifest` per run. It measures service mechanics, not provider-bound
capacity, and proposes no budgets.

## Recorded data
- `driver.json`: per-cell summaries (counts, errors, percentiles only where the sample count
  supports them: p50 from 5, p95 from 20, p99 from 100 samples).
- `samples-<workload>-c<N>.jsonl.gz`: **every request**, warmup included and flagged
  (`warmup`), with start and end offsets in seconds, accept and completion times and outcome.
  Percentiles can be recomputed, other warmup rules applied and drift inside a cell inspected.
- `record.json`: image facts, host facts (load before and after, other containers before,
  after and during the run), resource peaks, cold start.

## Rules the numbers follow
- **Warmup:** the first `max(warmup, concurrency)` dispatched operations of a cell are
  excluded, because the first `concurrency` operations start together.
- **Throughput:** measured operations divided by the window from the first measured start to
  the last measured end (not wall time including warmup).
- **Accept series:** only accepted requests (HTTP 202) enter accept statistics; refusals are
  counted as errors and as failures in the manifest.
- **Run selection:** a run is *clean* when no container outside the run's compose project was
  running at its start, at its end or at any sample during it. The rule looks at the
  environment only, never at results; `python -m neuralscape_bench.service_baseline.report
  RUN_DIR... --mode clean|all` prints both views so the effect of the rule is visible. Host
  load is recorded but not used to select. Arms are interleaved and the order is reversed
  between repetitions to spread host drift.
- **Closed loop:** concurrency is fixed, which understates tail latency under overload. The
  driver records its own CPU use per cell; a value near one core means it is the limit.
