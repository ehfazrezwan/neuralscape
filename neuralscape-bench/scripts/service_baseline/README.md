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

## Safety rules the harness enforces
- **Disposable projects only.** Every compose command, `down -v` and sentinel-volume removal
  goes through `journey_safety.require_disposable_project`: the project name must start with
  `ns-bench-`. The production project name is refused in any letter case, and so is an empty
  name. The check runs before any docker command is built, so a refusal makes no docker call.
- **Loopback only.** The stack files publish no host ports. Before a run the runner (and the
  journey) scan the compose file and any override and refuse anything that publishes beyond
  `127.0.0.1`/`[::1]`, uses host networking, runs privileged or mounts the Docker socket.
- **Egress-free option.** `--compose-override scripts/service_baseline/compose.no-egress.yml`
  (or `NS_BENCH_COMPOSE_OVERRIDE`) makes the stack network internal; the runner then checks
  on Docker that the network really is internal and records it. Telemetry of mem0, Graphiti
  and Qdrant is switched off in the stack either way.
- **Older images.** The in-container scripts import the governed-service modules lazily and
  fall back (token-only seeding, API without an enforcer) with a notice when an image does
  not carry them.

## Command lines (usage only, nothing starts)
- `python -m neuralscape_bench.service_baseline.runner --help`
- `python -m neuralscape_bench.service_baseline.driver --help`
- `python -m neuralscape_bench.service_baseline.report --help`
- `python scripts/service_baseline/journey.py --help`
