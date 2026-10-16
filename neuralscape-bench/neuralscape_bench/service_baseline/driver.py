"""Closed-loop load driver for the service baseline (standard library only).

Runs inside the benchmark network so host noise does not shape the driver.
Writes one JSON result file; it records its own CPU use per cell so driver
saturation can be checked independently of the service.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import resource
import threading
import time
import urllib.parse
from typing import Any, Callable

from neuralscape_bench.service_baseline.stats import summarize

USER_ID = "bench-user"
PROJECT_ID = "bench-project"
TERMINAL = {"completed", "failed"}
# The service's three isolated ARQ queues (fast, graph, ingest).
QUEUES = ("neuralscape:queue", "neuralscape:graph", "neuralscape:ingest", "neuralscape:queue:ledger")


class Client:
    """One keep-alive connection per thread."""

    def __init__(self, base_url: str, timeout_s: float, token: str | None = None):
        self.token = token
        parsed = urllib.parse.urlparse(base_url)
        self.host, self.port, self.timeout_s = parsed.hostname, parsed.port or 80, timeout_s
        self.local = threading.local()

    def _conn(self) -> http.client.HTTPConnection:
        conn = getattr(self.local, "conn", None)
        if conn is None:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
            self.local.conn = conn
        return conn

    def request(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        payload = None if body is None else json.dumps(body)
        headers = {"Content-Type": "application/json"} if payload else {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        for attempt in (0, 1):
            conn = self._conn()
            try:
                conn.request(method, path, body=payload, headers=headers)
                response = conn.getresponse()
                data = response.read()
                try:
                    return response.status, json.loads(data) if data else None
                except json.JSONDecodeError:
                    return response.status, None
            except (http.client.HTTPException, OSError):
                conn.close()
                self.local.conn = None
                if attempt:
                    raise
        raise RuntimeError("unreachable")


def _write_body(nonce: str, index: int) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Benchmark note {nonce}-{index}: unique detail {hashlib.sha256(f'{nonce}{index}'.encode()).hexdigest()[:16]}",
            }
        ],
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
    }


def _raw_body(nonce: str, index: int) -> dict[str, Any]:
    return {
        "content": f"Benchmark fact {nonce}-{index}: {hashlib.sha256(f'{nonce}{index}'.encode()).hexdigest()[:24]}",
        "category": "preference",
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "visibility": "private",
    }


def op_write(client: Client, nonce: str, index: int, poll_s: float, timeout_s: float, *, raw: bool = False,
             poll: bool = True) -> dict[str, Any]:
    start = time.monotonic()
    path, body_for = ("/v1/memories/raw", _raw_body) if raw else ("/v1/memories", _write_body)
    status, body = client.request("POST", path, body_for(nonce, index))
    accepted = time.monotonic()
    if status != 202 or not body or "task_id" not in body:
        return {"ok": False, "accepted": False, "error": f"http_{status}", "accept_ms": (accepted - start) * 1000}
    if not poll:
        return {"ok": True, "accepted": True, "error": None, "accept_ms": (accepted - start) * 1000}
    task_id = body["task_id"]
    while True:
        now = time.monotonic()
        if now - start > timeout_s:
            return {"ok": False, "accepted": True, "error": "timeout", "accept_ms": (accepted - start) * 1000}
        code, state = client.request("GET", f"/v1/memories/status/{task_id}")
        if code == 200 and state and state.get("status") in TERMINAL:
            done = time.monotonic()
            return {
                "ok": state["status"] == "completed",
                "accepted": True,
                "error": None if state["status"] == "completed" else "task_failed",
                "accept_ms": (accepted - start) * 1000,
                "complete_ms": (done - start) * 1000,
            }
        time.sleep(poll_s)


def op_read(method: str, path: str, body: Any = None) -> Callable[..., dict[str, Any]]:
    def run(client: Client, nonce: str, index: int, poll_s: float, timeout_s: float) -> dict[str, Any]:
        start = time.monotonic()
        try:
            status, _ = client.request(method, path.format(i=index), body(index) if callable(body) else body)
        except (http.client.HTTPException, OSError) as error:
            return {"ok": False, "error": type(error).__name__}
        elapsed = (time.monotonic() - start) * 1000
        return {"ok": status == 200, "error": None if status == 200 else f"http_{status}", "latency_ms": elapsed}

    return run


def _raw(poll: bool) -> Callable[..., dict[str, Any]]:
    def run(client: Client, nonce: str, index: int, poll_s: float, timeout_s: float) -> dict[str, Any]:
        return op_write(client, nonce, index, poll_s, timeout_s, raw=True, poll=poll)

    return run


WORKLOADS: dict[str, Callable[..., dict[str, Any]]] = {
    "write": op_write,
    "raw": _raw(True),
    "raw_nopoll": _raw(False),
    "search": op_read(
        "POST",
        "/v1/search",
        lambda i: {"query": f"user preference workflow {i % 10}", "user_id": USER_ID, "limit": 10},
    ),
    "context": op_read("GET", f"/v1/context/{PROJECT_ID}?user_id={USER_ID}&limit=50"),
    "list": op_read("GET", f"/v1/memories?user_id={USER_ID}&limit=50"),
}


def run_cell(
    client: Client,
    workload: str,
    concurrency: int,
    ops: int,
    warmup: int,
    nonce: str,
    poll_s: float,
    timeout_s: float,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one closed-loop cell.

    The first ``max(warmup, concurrency)`` dispatched operations are excluded: the
    first ``concurrency`` operations all start together, so a smaller warmup would
    leave part of that ramp in the sample.  When ``samples`` is given, every
    operation (warmup included, flagged) is appended to it for storage.
    """

    function = WORKLOADS[workload]
    warmup = max(warmup, concurrency) if warmup else 0
    total = warmup + ops
    results: list[dict[str, Any] | None] = [None] * total
    counter = {"next": 0}
    lock = threading.Lock()

    def worker() -> None:
        while True:
            with lock:
                index = counter["next"]
                if index >= total:
                    return
                counter["next"] += 1
            began = time.monotonic() - wall_start
            try:
                record = function(client, nonce, index, poll_s, timeout_s)
            except Exception as error:  # noqa: BLE001 - recorded as an error outcome
                record = {"ok": False, "error": type(error).__name__}
            record["index"], record["start_s"] = index, round(began, 6)
            record["end_s"] = round(time.monotonic() - wall_start, 6)
            record["warmup"] = index < warmup
            results[index] = record

    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    wall_start = time.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.monotonic() - wall_start
    cpu_after = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (cpu_after.ru_utime - cpu_before.ru_utime) + (cpu_after.ru_stime - cpu_before.ru_stime)

    measured = [r for r in results[warmup:] if r is not None]
    if samples is not None:
        samples.extend({"workload": workload, "concurrency": concurrency, **r} for r in results if r is not None)
    window = (max(r["end_s"] for r in measured) - min(r["start_s"] for r in measured)) if measured else 0.0
    errors: dict[str, int] = {}
    for record in measured:
        if not record["ok"]:
            errors[record["error"] or "error"] = errors.get(record["error"] or "error", 0) + 1
    good = [r for r in measured if r["ok"]]
    cell: dict[str, Any] = {
        "workload": workload,
        "concurrency": concurrency,
        "attempted": total,
        "warmup_excluded": warmup,
        "measured_attempts": len(measured),
        "ok": len(good),
        "errors": errors,
        "wall_seconds": wall,
        "measured_window_seconds": window,
        "throughput_per_s": len(measured) / window if window else None,
        "driver_cpu_cores_used": cpu / wall if wall else None,
    }
    if workload in ("write", "raw", "raw_nopoll"):
        # Only accepted requests have a meaningful accept time; refusals are counted as errors.
        cell["accept"] = summarize([r["accept_ms"] for r in measured if r.get("accepted")])
        cell["completion"] = summarize([r["complete_ms"] for r in good if "complete_ms" in r])
    else:
        cell["latency"] = summarize([r["latency_ms"] for r in good])
    return cell


def wait_ready(client: Client, timeout_s: float) -> float:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        try:
            status, _ = client.request("GET", "/health/live")
            if status == 200:
                return time.monotonic() - start
        except (http.client.HTTPException, OSError):
            pass
        time.sleep(0.5)
    raise SystemExit("service never became live")


def _redis_backlog(redis_url: str) -> int | None:
    """Pending queue entries plus in-progress jobs across every worker queue."""

    try:
        import redis  # present in the service image; the driver runs there

        connection = redis.Redis.from_url(redis_url, socket_timeout=5)
        pending = sum(connection.zcard(name) for name in QUEUES)
        # Cron jobs leave an in-progress marker for a while after they ran; only real jobs count.
        pending += sum(
            1
            for key in connection.scan_iter(match="arq:in-progress:*", count=200)
            if not key.decode().startswith("arq:in-progress:cron:")
        )
        return pending
    except Exception:  # noqa: BLE001 - fall back to the API view
        return None


def drain(client: Client, timeout_s: float, redis_url: str | None = None) -> dict[str, Any]:
    """Wait until every worker queue (graph enrichment included) is empty and idle."""

    start = time.monotonic()
    quiet_since: float | None = None
    last: Any = None
    while time.monotonic() - start < timeout_s:
        backlog = _redis_backlog(redis_url) if redis_url else None
        if backlog is None:
            status, body = client.request("GET", f"/v1/queue/status?user_id={USER_ID}")
            last = body
            depths = (body or {}).get("queues") or {}
            backlog = sum(v for v in depths.values() if isinstance(v, int)) if status == 200 else 1
            if backlog == 0 and not (body or {}).get("caught_up"):
                backlog = 1
        else:
            last = {"redis_backlog": backlog}
        if backlog == 0:
            quiet_since = quiet_since or time.monotonic()
            if time.monotonic() - quiet_since >= 2.0:  # idle for two seconds
                return {"drained": True, "seconds": time.monotonic() - start}
        else:
            quiet_since = None
        time.sleep(0.5)
    return {"drained": False, "seconds": time.monotonic() - start, "last": last}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--concurrency", default="1,4,16")
    parser.add_argument("--workloads", default="write,search,context,list")
    parser.add_argument("--write-ops", type=int, default=48)
    parser.add_argument("--read-ops", type=int, default=200)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--seed-workload", default="write")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--poll-ms", type=float, default=50)
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--drain-timeout-s", type=float, default=300)
    parser.add_argument("--redis-url", default="redis://redis:6379")
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--samples-dir", default=None, help="directory for per-request samples (default: next to --out)")
    args = parser.parse_args()

    token = open(args.token_file).read().strip() if args.token_file else None
    client = Client(args.base_url, timeout_s=60, token=token)
    result: dict[str, Any] = {"args": vars(args), "ready_seconds": wait_ready(client, 300)}
    started = time.time()

    seed_cell = run_cell(client, args.seed_workload, 4, args.seed, 0, f"{args.nonce}-seed", args.poll_ms / 1000, args.timeout_s)
    result["seed"] = seed_cell
    result["seed_drain"] = drain(client, args.drain_timeout_s, args.redis_url)

    import gzip
    import os

    samples_dir = args.samples_dir or os.path.dirname(args.out) or "."
    cells = []
    for workload in args.workloads.split(","):
        for concurrency in (int(c) for c in args.concurrency.split(",")):
            ops = args.write_ops if workload in ("write", "raw", "raw_nopoll") else args.read_ops
            cell_samples: list[dict[str, Any]] = []
            cells.append(
                run_cell(client, workload, concurrency, ops, args.warmup, f"{args.nonce}-{workload}-{concurrency}",
                         args.poll_ms / 1000, args.timeout_s, cell_samples)
            )
            with gzip.open(os.path.join(samples_dir, f"samples-{workload}-c{concurrency}.jsonl.gz"), "wt") as handle:
                for row in cell_samples:
                    handle.write(json.dumps(row) + "\n")
            if workload in ("write", "raw", "raw_nopoll"):
                drain_result = drain(client, args.drain_timeout_s, args.redis_url)
                cells[-1]["post_cell_drain"] = drain_result
                cells[-1]["end_to_end_seconds"] = cells[-1]["wall_seconds"] + drain_result["seconds"]
    result["cells"] = cells
    result["driver_started_unix"] = started
    result["driver_finished_unix"] = time.time()
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
