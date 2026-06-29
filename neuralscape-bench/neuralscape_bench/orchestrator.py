"""One-shot A/B: benchmark an untuned baseline against a tuned candidate.

For each side it brings up an isolated Neuralscape stack (its own
``docker compose -p`` project + the bench port override, built from that
branch's git worktree), waits for health, runs the same benchmark suite, then
tears the stack down. Emits ``results/compare-<ts>.json`` with both runs + a
per-metric comparison.

Escape hatches:
  --baseline-url URL   benchmark an already-running stack as the baseline
                       (e.g. the live untuned :8199) instead of building one
  --no-teardown        leave stacks up for inspection
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from neuralscape_bench.models import RunResult, Target, compare_metrics
from neuralscape_bench.runner import RESULTS_DIR, config_for_profile, run_benchmark, save_result

BENCH_DIR = Path(__file__).resolve().parent.parent
STACKS_DIR = BENCH_DIR / ".stacks"
OVERRIDE = BENCH_DIR / "docker-compose.bench.yml"


def _log(msg: str) -> None:
    print(f"[ab] {msg}", flush=True)


def _run(cmd: list[str], env: dict | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    _log("$ " + " ".join(cmd))
    return subprocess.run(cmd, env=env, timeout=timeout, check=True)


def _repo_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def _slug(branch: str) -> str:
    # The slug names BOTH the git worktree dir and the docker compose project,
    # so distinct branches must produce distinct slugs. Plain char-substitution
    # collides (`a/b` and `a+b` both → `a-b`), reusing the wrong worktree and
    # invalidating A/B attribution. Append a short hash of the exact branch name.
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", branch).strip("-") or "branch"
    return f"{safe}-{hashlib.sha1(branch.encode()).hexdigest()[:8]}"


def _branch_commit(worktree: Path) -> str:
    out = subprocess.run(["git", "-C", str(worktree), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def ensure_worktree(repo_root: Path, branch: str) -> Path:
    """Create (or reuse) a git worktree for ``branch`` under .stacks/<slug>."""
    STACKS_DIR.mkdir(exist_ok=True)
    path = STACKS_DIR / _slug(branch)
    if (path / ".git").exists():
        _log(f"reusing worktree {path}")
        return path
    _log(f"creating worktree for {branch} → {path}")
    # Prefer an existing local branch; fall back to origin/<branch>.
    try:
        _run(["git", "-C", str(repo_root), "worktree", "add", str(path), branch])
    except subprocess.CalledProcessError:
        _run(["git", "-C", str(repo_root), "worktree", "add", str(path), f"origin/{branch}"])
    return path


def _compose_base(project: str, worktree: Path) -> list[str]:
    return ["docker", "compose", "-p", project,
            "-f", str(worktree / "docker-compose.yml"), "-f", str(OVERRIDE)]


def stack_up(project: str, worktree: Path, api_port: int, env_file: Path) -> None:
    # The branch's compose uses `env_file: .env` relative to its own dir; make
    # sure that file exists in the worktree (copy the provided one in).
    target_env = worktree / ".env"
    if not target_env.exists():
        if not env_file.exists():
            raise FileNotFoundError(
                f"No .env at {target_env} and --env-file {env_file} not found. "
                "A stack needs GOOGLE_API_KEY / NEO4J_PASSWORD etc. to run."
            )
        shutil.copyfile(env_file, target_env)
        _log(f"copied {env_file} → {target_env}")
    import os
    env = {**os.environ, "BENCH_API_PORT": str(api_port)}
    _run(_compose_base(project, worktree) + ["up", "-d", "--build"], env=env, timeout=1800)


def stack_down(project: str, worktree: Path) -> None:
    try:
        _run(_compose_base(project, worktree) + ["down", "-v"], timeout=300)
    except Exception as e:  # noqa: BLE001
        _log(f"WARNING: teardown failed for {project}: {e} (run `docker compose -p {project} down -v` manually)")


def wait_health(api_port: int, timeout_s: float = 180.0) -> None:
    url = f"http://localhost:{api_port}/health"
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 — localhost
                if r.status == 200:
                    _log(f"healthy: {url}")
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    raise TimeoutError(f"{url} not healthy within {timeout_s}s")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _bench_target(label: str, url: str, profile: str, cfg, *, live: bool) -> RunResult:
    target = Target(base_url=url, label=label, profile=profile, live_baseline=live)
    result = await run_benchmark(target, cfg, now_iso=_now_iso())
    save_result(result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B benchmark: untuned baseline vs tuned candidate.")
    ap.add_argument("--candidate-branch", required=True, help="e.g. perf/write-read-isolation")
    ap.add_argument("--baseline-branch", default="dev")
    ap.add_argument("--baseline-url", default=None,
                    help="Benchmark this running stack as the baseline instead of building one.")
    ap.add_argument("--profile", default="light", choices=["light", "full"])
    ap.add_argument("--baseline-port", type=int, default=8298)
    ap.add_argument("--candidate-port", type=int, default=8299)
    ap.add_argument("--env-file", default=None, help="Path to the .env used for built stacks.")
    ap.add_argument("--no-teardown", action="store_true")
    args = ap.parse_args()

    repo_root = _repo_root()
    env_file = Path(args.env_file) if args.env_file else (repo_root / ".env")
    cfg = config_for_profile(args.profile)
    brought_up: list[tuple[str, Path]] = []  # (project, worktree) to tear down

    try:
        # ── Baseline ──
        if args.baseline_url:
            _log(f"baseline = live stack {args.baseline_url}")
            baseline = asyncio.run(_bench_target(
                f"{args.baseline_branch}-live", args.baseline_url, args.profile, cfg, live=True))
        else:
            wt = ensure_worktree(repo_root, args.baseline_branch)
            proj = f"nsbench-{_slug(args.baseline_branch)}"
            # Track BEFORE bring-up: if stack_up fails after creating some
            # containers, the finally block must still tear them down.
            brought_up.append((proj, wt))
            stack_up(proj, wt, args.baseline_port, env_file)
            wait_health(args.baseline_port)
            label = f"{args.baseline_branch}@{_branch_commit(wt)}"
            baseline = asyncio.run(_bench_target(
                label, f"http://localhost:{args.baseline_port}", args.profile, cfg, live=False))

        # ── Candidate ──
        wt = ensure_worktree(repo_root, args.candidate_branch)
        proj = f"nsbench-{_slug(args.candidate_branch)}"
        brought_up.append((proj, wt))  # track before bring-up (see baseline note)
        stack_up(proj, wt, args.candidate_port, env_file)
        wait_health(args.candidate_port)
        label = f"{args.candidate_branch}@{_branch_commit(wt)}"
        candidate = asyncio.run(_bench_target(
            label, f"http://localhost:{args.candidate_port}", args.profile, cfg, live=False))

        # ── Compare ──
        comparison = compare_metrics(baseline.metrics, candidate.metrics)
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"compare-{_now_iso().replace(':', '').replace('-', '')[:15]}.json"
        out.write_text(json.dumps({
            "baseline": baseline.to_dict(),
            "candidate": candidate.to_dict(),
            "comparison": comparison,
        }, indent=2))
        _log(f"comparison → {out}")
        _print_headline(comparison)
    finally:
        if args.no_teardown:
            _log("--no-teardown: leaving stacks up: " + ", ".join(p for p, _ in brought_up))
        else:
            for proj, wt in brought_up:
                stack_down(proj, wt)


def _print_headline(cmp: dict) -> None:
    keys = [
        ("write.e2e_ms.p95", "write e2e p95 (ms)"),
        ("write.enqueue_ms.p95", "write enqueue p95 (ms)"),
        ("throughput.writes_per_sec", "writes/sec"),
        ("contention.loaded_over_unloaded_p95", "read p95 loaded/unloaded"),
        ("read.search.p95", "search p95 (ms)"),
    ]
    print("\n=== headline deltas (candidate vs baseline) ===")
    for path, label in keys:
        if path in cmp:
            d = cmp[path]
            arrow = "✓" if d["improved"] else "✗"
            print(f"  {arrow} {label}: {d['baseline']} → {d['candidate']} ({d['pct_change']:+}%)")


if __name__ == "__main__":
    main()
