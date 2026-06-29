"""Multi-user concurrency stress test.

Simulates N distinct users hammering Neuralscape at once with a read/write mix
for a fixed duration, then reports aggregate throughput + error rate, overall
read/write latency percentiles, AND per-user p95 + a fairness measure (does one
user's load starve another?). This is the "can it handle concurrent users"
scenario, distinct from the single-stream latency suites in runner.py.

Writes are enqueue-only (we measure the 202 latency + system saturation under
concurrency, not end-to-end graph completion — that's runner.measure_write_latency).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from neuralscape_bench.client import NeuralscapeClient
from neuralscape_bench.models import RunResult, Target, percentile, summarize
from neuralscape_bench.runner import _QUERIES, _content, _git_commit, save_result

_STRESS_PROFILES = {
    "light": dict(users=8, duration_s=15.0, per_user_concurrency=2, read_ratio=0.8,
                  shared_write_ratio=0.2, seed_per_user=4),
    "full": dict(users=25, duration_s=45.0, per_user_concurrency=3, read_ratio=0.8,
                 shared_write_ratio=0.2, seed_per_user=6),
}


@dataclass
class StressConfig:
    users: int = 8
    duration_s: float = 15.0
    per_user_concurrency: int = 2
    read_ratio: float = 0.8            # fraction of ops that are reads (search)
    shared_write_ratio: float = 0.2    # fraction of writes that target the shared pool
    seed_per_user: int = 4
    seed_wait_s: float = 30.0

    def to_dict(self) -> dict:
        return asdict(self)


def stress_config(profile: str, **overrides) -> StressConfig:
    base = dict(_STRESS_PROFILES.get(profile, _STRESS_PROFILES["light"]))
    base.update({k: v for k, v in overrides.items() if v is not None})
    return StressConfig(**base)


# ── pure aggregation (unit-tested) ────────────────────────────────


def aggregate_stress(per_user: dict[str, dict[str, list[float]]], *,
                     duration_s: float, errors: int) -> dict:
    """Aggregate per-user latency samples into stress metrics.

    ``per_user`` maps user_id → {"read": [ms...], "write": [ms...]}.
    Fairness is the spread of per-user read p95: ``spread`` = max/min and
    ``cv`` = stdev/mean (0 = perfectly fair; larger = some users starved).
    """
    read_all: list[float] = []
    write_all: list[float] = []
    per_user_read_p95: dict[str, float] = {}
    per_user_write_p95: dict[str, float] = {}
    for user, ops in per_user.items():
        reads = ops.get("read", [])
        writes = ops.get("write", [])
        read_all.extend(reads)
        write_all.extend(writes)
        if reads:
            per_user_read_p95[user] = round(percentile(sorted(reads), 95), 2)
        if writes:
            per_user_write_p95[user] = round(percentile(sorted(writes), 95), 2)

    total_ops = len(read_all) + len(write_all) + errors
    p95s = list(per_user_read_p95.values())
    fairness = {
        "read_p95_max": round(max(p95s), 2) if p95s else 0.0,
        "read_p95_min": round(min(p95s), 2) if p95s else 0.0,
        "read_p95_spread": round(max(p95s) / min(p95s), 2) if p95s and min(p95s) else None,
        "read_p95_cv": round(statistics.pstdev(p95s) / statistics.mean(p95s), 3)
                       if len(p95s) > 1 and statistics.mean(p95s) else 0.0,
    }
    return {
        "users": len(per_user),
        "duration_s": duration_s,
        "total_ops": total_ops,
        "ops_per_sec": round(total_ops / duration_s, 1) if duration_s else 0.0,
        "errors": errors,
        "error_rate": round(errors / total_ops, 4) if total_ops else 0.0,
        "read": summarize(read_all),
        "write": summarize(write_all),
        "per_user_read_p95": per_user_read_p95,
        "per_user_write_p95": per_user_write_p95,
        "fairness": fairness,
    }


# ── load generation ───────────────────────────────────────────────


async def _seed_user(client: NeuralscapeClient, user: str, project: str, n: int) -> None:
    for i in range(n):
        try:
            await client.raw_write(_content(i), user_id=user, category="convention",
                                   scope="project", project_id=project)
        except Exception:  # noqa: BLE001
            pass


async def run_stress(target: Target, cfg: StressConfig, *, now_iso: str) -> RunResult:
    run_id = now_iso.replace(":", "").replace("-", "").replace(".", "")[:15]
    users = [f"stress-{run_id}-u{i}" for i in range(cfg.users)]
    projects = {u: u for u in users}  # one project namespace per user
    max_conn = max(100, cfg.users * cfg.per_user_concurrency * 2)
    client = NeuralscapeClient(target.base, token=target.token, max_connections=max_conn)

    per_user: dict[str, dict[str, list[float]]] = {u: {"read": [], "write": []} for u in users}
    errors = 0
    notes = [f"profile={target.profile}", f"{cfg.users} users × {cfg.per_user_concurrency} workers",
             "writes are enqueue-only (202 latency), not polled to completion"]

    try:
        await client.health()
        # Seed each user's corpus concurrently, then wait until searchable.
        await asyncio.gather(*(_seed_user(client, u, projects[u], cfg.seed_per_user) for u in users))
        deadline_seed = time.perf_counter() + cfg.seed_wait_s
        while time.perf_counter() < deadline_seed:
            probe = await client.search(_QUERIES[0], user_id=users[0], project_id=projects[users[0]], limit=3)
            if probe.get("results"):
                break
            await asyncio.sleep(1.0)

        # Sustained load.
        deadline = time.perf_counter() + cfg.duration_s
        write_counter = {"n": 0}

        async def worker(user: str, wid: int):
            nonlocal errors
            i = 0
            # Deterministic read/write interleave by a per-worker counter
            # against read_ratio (no RNG → reproducible mixes).
            while time.perf_counter() < deadline:
                is_read = (i % 100) < int(cfg.read_ratio * 100)
                i += 1
                try:
                    if is_read:
                        t0 = time.perf_counter()
                        await client.search(_QUERIES[(wid + i) % len(_QUERIES)],
                                            user_id=user, project_id=projects[user], limit=10)
                        per_user[user]["read"].append((time.perf_counter() - t0) * 1000.0)
                    else:
                        n = write_counter["n"]
                        write_counter["n"] += 1
                        shared = (n % 100) < int(cfg.shared_write_ratio * 100)
                        t0 = time.perf_counter()
                        await client.raw_write(_content(10000 + n), user_id=user, category="convention",
                                               scope="project", project_id=projects[user],
                                               visibility="shared" if shared else "private")
                        per_user[user]["write"].append((time.perf_counter() - t0) * 1000.0)
                except Exception:  # noqa: BLE001
                    errors += 1

        tasks = [asyncio.create_task(worker(u, w))
                 for u in users for w in range(cfg.per_user_concurrency)]
        await asyncio.gather(*tasks, return_exceptions=True)
        metrics = {"stress": aggregate_stress(per_user, duration_s=cfg.duration_s, errors=errors)}
    finally:
        await asyncio.gather(*(client.delete_bench_data(user_id=u) for u in users),
                             return_exceptions=True)
        await client.aclose()

    return RunResult(label=target.label, target_url=target.base, profile=target.profile,
                     timestamp=now_iso, git_commit=_git_commit(), config=cfg.to_dict(),
                     metrics=metrics, notes=notes)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _main_async(args) -> None:
    cfg = stress_config(args.profile, users=args.users, duration_s=args.duration,
                        per_user_concurrency=args.concurrency, read_ratio=args.read_ratio)
    target = Target(base_url=args.target, label=args.label, token=args.token, profile=args.profile)
    result = await run_stress(target, cfg, now_iso=_now_iso())
    path = save_result(result)
    s = result.metrics["stress"]
    print(json.dumps(s, indent=2))
    print(f"\n{s['users']} users · {s['ops_per_sec']} ops/sec · err {s['error_rate']*100:.1f}% · "
          f"read p95 {s['read']['p95']}ms · fairness spread ×{s['fairness']['read_p95_spread']}")
    print(f"Saved → {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-user concurrency stress test for Neuralscape.")
    ap.add_argument("--target", required=True, help="Base URL, e.g. http://localhost:8199")
    ap.add_argument("--label", required=True, help="Run label, e.g. dev-stress")
    ap.add_argument("--profile", default="light", choices=["light", "full"])
    ap.add_argument("--token", default=None)
    ap.add_argument("--users", type=int, default=None)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=None, help="Workers per user")
    ap.add_argument("--read-ratio", type=float, default=None, dest="read_ratio")
    asyncio.run(_main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
