"""Benchmark engine: drives a Neuralscape target and measures four metric families.

REST-visible improvements this captures (vs the untuned baseline):
- **Write e2e latency / throughput** — the big one: on the untuned stack a raw
  write completes only after an inline ~2-min Graphiti graph.add; the tuned
  stack returns after the fast vector insert and defers graph work.
- **Read latency** (search/graph-search/list/context) and **read-under-write
  contention** — secondary effects of isolating slow graph work.

NOTE: the MCP event-loop fix (#1) is not exercised here — this harness drives
the REST API, whose routes already offload to a thread pool. That's recorded in
the run notes so the comparison isn't over-claimed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from neuralscape_bench.client import NeuralscapeClient, TaskTimeout
from neuralscape_bench.models import BenchConfig, RunResult, Target, summarize

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_PROFILES = {
    # Fast iteration: tiny e2e sample + short timeout (untuned writes won't finish).
    "light": dict(iterations=20, concurrency=6, write_load_writers=4, contention_reads=30,
                  throughput_duration_s=6.0, e2e_cap=5, poll_timeout_s=25.0, seed_count=20),
    # Faithful comparison: more samples + long timeout to capture true e2e write latency.
    "full": dict(iterations=50, concurrency=8, write_load_writers=8, contention_reads=50,
                 throughput_duration_s=12.0, e2e_cap=10, poll_timeout_s=180.0, seed_count=30),
}

_QUERIES = [
    "deployment workflow", "database schema", "authentication", "caching strategy",
    "error handling", "api design", "testing approach", "performance tuning",
]


def config_for_profile(profile: str, **overrides) -> BenchConfig:
    base = dict(_PROFILES.get(profile, _PROFILES["light"]))
    base.update({k: v for k, v in overrides.items() if v is not None})
    return BenchConfig(**base)


def _content(i: int) -> str:
    topic = _QUERIES[i % len(_QUERIES)]
    return f"Benchmark note {i}: the team's {topic} convention is documented and applied consistently across services."


async def _timed(coro):
    t0 = time.perf_counter()
    await coro
    return (time.perf_counter() - t0) * 1000.0


async def _gather_bounded(factories, concurrency):
    """Run async thunks with bounded concurrency; return results in order."""
    sem = asyncio.Semaphore(concurrency)

    async def _run(thunk):
        async with sem:
            return await thunk()

    return await asyncio.gather(*(_run(f) for f in factories), return_exceptions=True)


# ── suites ─────────────────────────────────────────────────────────


async def _seed(client: NeuralscapeClient, cfg: BenchConfig, user: str, project: str) -> int:
    """Enqueue a synthetic corpus and wait until it's searchable (vector side)."""
    factories = [
        (lambda i=i: client.raw_write(_content(i), user_id=user, category="convention",
                                      scope="project", project_id=project))
        for i in range(cfg.seed_count)
    ]
    await _gather_bounded(factories, cfg.concurrency)
    # Wait for vectors to appear (don't wait for the slow graph side).
    deadline = time.perf_counter() + cfg.seed_wait_s
    while time.perf_counter() < deadline:
        res = await client.search(_QUERIES[0], user_id=user, project_id=project, limit=5)
        if res.get("results"):
            return len(res["results"])
        await asyncio.sleep(1.0)
    return 0


async def measure_write_latency(client, cfg, user, project) -> dict:
    # Enqueue phase — concurrent, capture 202 latency + start time per write.
    async def enqueue(i):
        t0 = time.perf_counter()
        resp = await client.raw_write(_content(1000 + i), user_id=user, category="convention",
                                      scope="project", project_id=project)
        enq = (time.perf_counter() - t0) * 1000.0
        return (resp.get("task_id"), t0, enq)

    results = await _gather_bounded([lambda i=i: enqueue(i) for i in range(cfg.iterations)], cfg.concurrency)
    ok = [r for r in results if isinstance(r, tuple)]
    enqueue_samples = [enq for _, _, enq in ok]
    enqueue_errors = len(results) - len(ok)

    # E2E phase — poll a capped subset to completion; e2e measured from enqueue start.
    e2e_samples: list[float] = []
    completed = timeouts = failed = 0

    async def await_one(task_id, t0):
        nonlocal completed, timeouts, failed
        try:
            status = await client.wait_for_task(task_id, timeout_s=cfg.poll_timeout_s,
                                                interval_s=cfg.poll_interval_s)
            if status.get("status") == "completed":
                e2e_samples.append((time.perf_counter() - t0) * 1000.0)
                completed += 1
            else:
                failed += 1
        except TaskTimeout:
            timeouts += 1

    subset = [(tid, t0) for tid, t0, _ in ok if tid][: cfg.e2e_cap]
    await asyncio.gather(*(await_one(tid, t0) for tid, t0 in subset))

    return {
        "enqueue_ms": summarize(enqueue_samples),
        "e2e_ms": summarize(e2e_samples),
        "e2e_attempts": len(subset),
        "e2e_completed": completed,
        "e2e_timeouts": timeouts,
        "e2e_failed": failed,
        "enqueue_errors": enqueue_errors,
    }


async def measure_read_latency(client, cfg, user, project) -> dict:
    ops = {
        "search": lambda i: client.search(_QUERIES[i % len(_QUERIES)], user_id=user, project_id=project),
        "graph_search": lambda i: client.graph_search(_QUERIES[i % len(_QUERIES)], user_id=user, project_id=project),
        "list": lambda i: client.list_memories(user_id=user, limit=50),
        "context": lambda i: client.context_global(user_id=user),
    }
    out: dict = {}
    for name, op in ops.items():
        async def one(i, op=op):
            return await _timed(op(i))
        res = await _gather_bounded([lambda i=i: one(i) for i in range(cfg.iterations)], cfg.concurrency)
        samples = [r for r in res if isinstance(r, (int, float))]
        out[name] = summarize(samples)
        out[name]["errors"] = len(res) - len(samples)
    return out


async def measure_contention(client, cfg, user, project) -> dict:
    """Read p95 with no writers vs read p95 under sustained concurrent write load."""
    async def read_burst(n):
        async def one(i):
            return await _timed(client.search(_QUERIES[i % len(_QUERIES)], user_id=user, project_id=project))
        res = await _gather_bounded([lambda i=i: one(i) for i in range(n)], cfg.concurrency)
        return [r for r in res if isinstance(r, (int, float))]

    unloaded = summarize(await read_burst(cfg.contention_reads))

    stop = asyncio.Event()
    write_attempts = write_errors = 0

    async def writer(wid):
        # The load generator tolerates write failures, but must COUNT them:
        # if writers are silently erroring, the "loaded" read latency looks
        # healthy while no real write load was ever applied (false negative).
        nonlocal write_attempts, write_errors
        i = 0
        while not stop.is_set():
            write_attempts += 1
            try:
                await client.raw_write(_content(5000 + wid * 1000 + i), user_id=user,
                                       category="convention", scope="project", project_id=project)
            except Exception:  # noqa: BLE001 — tolerated but counted (see above)
                write_errors += 1
            i += 1

    writers = [asyncio.create_task(writer(w)) for w in range(cfg.write_load_writers)]
    try:
        loaded = summarize(await read_burst(cfg.contention_reads))
    finally:
        stop.set()
        await asyncio.gather(*writers, return_exceptions=True)

    ratio = round(loaded["p95"] / unloaded["p95"], 2) if unloaded["p95"] else None
    return {
        "unloaded": unloaded, "loaded": loaded, "loaded_over_unloaded_p95": ratio,
        # Surfaced so a failed write-load (which invalidates the contention
        # result) is visible instead of silently passing.
        "write_load_attempts": write_attempts, "write_load_errors": write_errors,
    }


async def measure_throughput(client, cfg, user, project) -> dict:
    async def drive(make_call):
        count = errors = 0
        stop_at = time.perf_counter() + cfg.throughput_duration_s

        async def worker():
            nonlocal count, errors
            while time.perf_counter() < stop_at:
                try:
                    await make_call()
                    count += 1
                except Exception:  # noqa: BLE001
                    errors += 1

        await asyncio.gather(*(worker() for _ in range(cfg.concurrency)))
        return count, errors

    r_count, r_err = await drive(lambda: client.search(_QUERIES[0], user_id=user, project_id=project))
    w_count, w_err = await drive(lambda: client.raw_write(_content(9000), user_id=user,
                                                          category="convention", scope="project", project_id=project))
    dur = cfg.throughput_duration_s
    return {
        "reads_per_sec": round(r_count / dur, 1), "read_errors": r_err,
        "writes_per_sec": round(w_count / dur, 1), "write_errors": w_err,
    }


# ── top-level run ──────────────────────────────────────────────────


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


async def run_benchmark(target: Target, cfg: BenchConfig, *, now_iso: str) -> RunResult:
    run_id = now_iso.replace(":", "").replace("-", "").replace(".", "")[:15]
    user = f"bench-{run_id}"
    project = f"bench-{run_id}"
    client = NeuralscapeClient(target.base, token=target.token)
    notes = [
        "REST-driven; MCP event-loop fix (#1) is not exercised by this harness.",
        f"profile={target.profile}",
    ]
    if target.live_baseline:
        notes.append("live baseline: pre-existing corpus → read-latency is indicative, not corpus-controlled.")
    try:
        await client.health()
        seeded = await _seed(client, cfg, user, project)
        notes.append(f"seeded {seeded} searchable memories")
        metrics = {
            "write": await measure_write_latency(client, cfg, user, project),
            "read": await measure_read_latency(client, cfg, user, project),
            "contention": await measure_contention(client, cfg, user, project),
            "throughput": await measure_throughput(client, cfg, user, project),
        }
    finally:
        await client.delete_bench_data(user_id=user)
        await client.aclose()

    return RunResult(
        label=target.label, target_url=target.base, profile=target.profile,
        timestamp=now_iso, git_commit=_git_commit(), config=cfg.to_dict(),
        metrics=metrics, notes=notes,
    )


def save_result(result: RunResult) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    safe = result.label.replace("/", "-").replace(" ", "_")
    path = RESULTS_DIR / f"{safe}-{result.timestamp.replace(':', '').replace('-', '')[:15]}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2))
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _main_async(args) -> None:
    cfg = config_for_profile(args.profile, iterations=args.iterations, concurrency=args.concurrency)
    target = Target(base_url=args.target, label=args.label, token=args.token,
                    profile=args.profile, live_baseline=args.live_baseline)
    result = await run_benchmark(target, cfg, now_iso=_now_iso())
    path = save_result(result)
    print(json.dumps(result.metrics, indent=2))
    print(f"\nSaved → {path}")


def _positive_int(value: str) -> int:
    """argparse type: reject 0/negative overrides before they reach scheduling."""
    iv = int(value)
    if iv < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return iv


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark a Neuralscape target.")
    ap.add_argument("--target", required=True, help="Base URL, e.g. http://localhost:8199")
    ap.add_argument("--label", required=True, help="Run label, e.g. dev@da54615")
    ap.add_argument("--profile", default="light", choices=["light", "full"])
    ap.add_argument("--token", default=None, help="Optional bearer token")
    ap.add_argument("--iterations", type=_positive_int, default=None)
    ap.add_argument("--concurrency", type=_positive_int, default=None)
    ap.add_argument("--live-baseline", action="store_true",
                    help="Mark this target as a pre-existing populated stack (read-latency caveat).")
    asyncio.run(_main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
