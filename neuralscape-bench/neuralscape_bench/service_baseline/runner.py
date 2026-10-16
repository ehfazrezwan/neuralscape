"""Host-side orchestration of one named service baseline run.

Brings up the isolated compose project for an exact image, samples container
resources, runs the in-network driver, then records a ``RunManifest`` plus the
raw results.  Reports are baselines only: no budget or target is implied.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neuralscape_bench.run_manifest import (
    BuildIdentity,
    CacheCondition,
    ConcurrencySetting,
    FailureOutcome,
    LimitKind,
    MeasurementProvenance,
    ObservationKind,
    ResourceReading,
    RunManifest,
    RunState,
    SampleExclusion,
    TimingMeasurement,
    WorkloadIdentity,
    finish_run,
)

SCHEMA = "candidate-v1"
PROJECT = "ns-g-bench"
BENCH_DIR = Path(__file__).resolve().parents[2]
COMPOSE_FILE = BENCH_DIR / "scripts" / "service_baseline" / "compose.yml"
SERVICES = ("api", "worker", "graph-worker", "ingest-worker", "fake-provider", "neo4j", "redis", "qdrant")
MEM_LIMITS = {"api": 3 << 30, "worker": 2 << 30, "graph-worker": 2 << 30, "ingest-worker": 2 << 30,
              "fake-provider": 512 << 20, "neo4j": 2 << 30, "redis": 512 << 20, "qdrant": 2 << 30}
_UNITS = {"b": 1, "kib": 1 << 10, "mib": 1 << 20, "gib": 1 << 30, "kb": 10**3, "mb": 10**6, "gb": 10**9}


def sh(*args: str, check: bool = True, env: dict[str, str] | None = None) -> str:
    done = subprocess.run(args, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if check and done.returncode:
        raise RuntimeError(f"{' '.join(args[:4])} failed: {done.stderr[-400:]}")
    return done.stdout


def parse_size(text: str) -> int:
    number = "".join(c for c in text if c in "0123456789.")
    unit = text[len(number):].strip().lower() or "b"
    return int(float(number) * _UNITS[unit])


def parse_stats_line(line: str) -> dict[str, Any] | None:
    """``docker stats --format json`` row → bytes and cpu percent."""

    try:
        row = json.loads(line)
        return {
            "name": row["Name"],
            "mem_bytes": parse_size(row["MemUsage"].split("/")[0].strip()),
            "cpu_percent": float(row["CPUPerc"].rstrip("%")),
        }
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


class StatsSampler(threading.Thread):
    """Samples ``docker stats`` for the project's containers (one pass per call)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.samples: list[dict[str, Any]] = []

    def run(self) -> None:
        while not self.stop.is_set():
            taken = time.time()
            out = sh("docker", "stats", "--no-stream", "--format", "{{json .}}",
                     *self._containers(), check=False)
            for line in out.splitlines():
                parsed = parse_stats_line(line)
                if parsed:
                    parsed["t"] = taken
                    self.samples.append(parsed)

    @staticmethod
    def _containers() -> list[str]:
        out = sh("docker", "ps", "--filter", f"label=com.docker.compose.project={PROJECT}", "--format", "{{.Names}}", check=False)
        return out.split()


def peaks(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for sample in samples:
        service = next((s for s in SERVICES if f"{PROJECT}-{s}-" in sample["name"]), None)
        if service is None:
            continue
        entry = result.setdefault(service, {"peak_mem_bytes": 0, "peak_cpu_percent": 0.0, "samples": 0})
        entry["peak_mem_bytes"] = max(entry["peak_mem_bytes"], sample["mem_bytes"])
        entry["peak_cpu_percent"] = max(entry["peak_cpu_percent"], sample["cpu_percent"])
        entry["samples"] += 1
    return result


def image_facts(image: str) -> dict[str, Any]:
    info = json.loads(sh("docker", "image", "inspect", image))[0]
    save = subprocess.Popen(["docker", "save", image], stdout=subprocess.PIPE)
    size = 0
    compressor = gzip.GzipFile(fileobj=_Counter(), mode="wb", compresslevel=1)
    counter = compressor.fileobj
    assert save.stdout is not None
    for chunk in iter(lambda: save.stdout.read(1 << 20), b""):
        compressor.write(chunk)
    compressor.close()
    save.wait()
    size = counter.total
    return {"id": info["Id"], "inspect_size_bytes": info["Size"], "layers": len(info["RootFS"]["Layers"]),
            "save_gzip_bytes": size,
            "images_listing_size": sh("docker", "image", "ls", "--format", "{{.Size}}", image).strip()}


class _Counter:
    def __init__(self) -> None:
        self.total = 0

    def write(self, data: bytes) -> int:
        self.total += len(data)
        return len(data)

    def flush(self) -> None:
        pass


def host_facts() -> dict[str, Any]:
    meminfo = Path("/proc/meminfo").read_text().splitlines()[0]
    return {
        "cpus": os.cpu_count(),
        "mem_total_kib": int(meminfo.split()[1]),
        "kernel": platform.release(),
        "loadavg": os.getloadavg(),
        "docker_version": sh("docker", "version", "--format", "{{.Server.Version}}", check=False).strip(),
        "other_running_containers": sh("docker", "ps", "--format", "{{.Names}}", check=False).split(),
    }


def compose(*args: str, env: dict[str, str]) -> str:
    return sh("docker", "compose", "-p", PROJECT, "-f", str(COMPOSE_FILE), *args, env=env)


def cold_start_seconds(env: dict[str, str], timeout_s: float = 420) -> float:
    start = time.monotonic()
    compose("up", "-d", "--wait", "--wait-timeout", str(int(timeout_s)), "neo4j", "redis", "qdrant", "fake-provider", env=env)
    compose("up", "-d", "api", "worker", "graph-worker", "ingest-worker", env=env)
    while time.monotonic() - start < timeout_s:
        probe = subprocess.run(
            ["docker", "compose", "-p", PROJECT, "-f", str(COMPOSE_FILE), "exec", "-T", "api", "python", "-c",
             "import urllib.request;urllib.request.urlopen('http://localhost:8199/health/live',timeout=3)"],
            capture_output=True, env={**os.environ, **env})
        if probe.returncode == 0:
            return time.monotonic() - start
        time.sleep(1)
    raise RuntimeError("api never became live")


def build_manifest(run_id: str, tree: str, lock_sha: str, image_id: str, results: dict[str, Any],
                   resource_peaks: dict[str, dict[str, Any]], started: datetime, finished: datetime,
                   corpus_sha: str, workload_sha: str) -> RunManifest:
    planned = RunManifest(
        schema_version=SCHEMA,
        run_id=run_id,
        state=RunState.PLANNED,
        build=BuildIdentity(schema_version=SCHEMA, git_commit=tree, image_digest=image_id, dependency_lock_sha256=lock_sha),
        workload=WorkloadIdentity(schema_version=SCHEMA, workload_id="service-baseline-v1", workload_sha256=workload_sha, corpus_sha256=corpus_sha),
        cache_condition=CacheCondition.WARM,
        concurrency=(
            ConcurrencySetting(schema_version=SCHEMA, scope="api-processes", value=1),
            ConcurrencySetting(schema_version=SCHEMA, scope="driver-threads-max", value=max(c["concurrency"] for c in results["cells"])),
        ),
        resources=tuple(
            ResourceReading(schema_version=SCHEMA, scope=service, resource="memory", unit="bytes",
                            limit_kind=LimitKind.BOUNDED, limit_value=MEM_LIMITS[service],
                            observation_kind=ObservationKind.PENDING)
            for service in SERVICES
        ),
    )
    provenance = MeasurementProvenance(schema_version=SCHEMA, collector="service-baseline-driver", collector_version="1",
                                       clock="monotonic", placement="in-network-driver-container")
    measurements = []
    for cell in results["cells"]:
        series = [("accept", cell["accept"]), ("complete", cell["completion"])] if cell["workload"] == "write" else [("latency", cell["latency"])]
        for suffix, summary in series:
            attempted = cell["attempted"]
            failed = cell["measured_attempts"] - summary["n"]
            exclusions = (SampleExclusion(schema_version=SCHEMA, kind="warmup", count=cell["warmup_excluded"],
                                          reason="discarded before measured operations"),) if cell["warmup_excluded"] else ()
            failures = (FailureOutcome(schema_version=SCHEMA, kind="error", count=failed),) if failed else ()
            measurements.append(TimingMeasurement(
                schema_version=SCHEMA, name=f"{cell['workload']}-{suffix}-c{cell['concurrency']}",
                time_unit="milliseconds", attempted_sample_count=attempted, included_sample_count=summary["n"],
                provenance=provenance, exclusions=exclusions, failure_outcomes=failures))
    readings = tuple(
        ResourceReading(schema_version=SCHEMA, scope=service, resource="memory", unit="bytes",
                        limit_kind=LimitKind.BOUNDED, limit_value=MEM_LIMITS[service],
                        **({"observation_kind": ObservationKind.MEASURED, "observed_value": resource_peaks[service]["peak_mem_bytes"]}
                           if service in resource_peaks else {"observation_kind": ObservationKind.UNKNOWN, "unknown_reason": "no_stats_sample"}))
        for service in SERVICES
    )
    return finish_run(planned, state=RunState.COMPLETED, started_at=started, finished_at=finished,
                      resources=readings, measurements=tuple(measurements))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--tree", required=True, help="40-hex commit the image was built from")
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--driver-args", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nonce = hashlib.sha256(f"{args.label}{time.time()}".encode()).hexdigest()[:10]
    env = {"NS_IMAGE": args.image, "BENCH_DIR": str(BENCH_DIR), "OUT_DIR": str(out_dir)}
    record: dict[str, Any] = {"label": args.label, "tree": args.tree, "image": args.image,
                              "image_facts": image_facts(args.image), "host_before": host_facts()}
    sampler = StatsSampler()
    started = datetime.now(timezone.utc)
    try:
        record["cold_start_seconds"] = cold_start_seconds(env)
        sampler.start()
        compose("run", "--rm", "--no-deps", "driver", "--base-url", "http://api:8199",
                "--out", "/out/driver.json", "--nonce", nonce, *args.driver_args.split(), env=env)
    finally:
        sampler.stop.set()
        if sampler.is_alive():
            sampler.join(timeout=30)
        finished = datetime.now(timezone.utc)
        record["logs_tail"] = {s: compose("logs", "--no-color", "--tail", "30", s, env=env) for s in ("api", "worker", "graph-worker")}
        compose("down", "-v", "--remove-orphans", env=env)
    record["host_after"] = host_facts()
    results = json.loads((out_dir / "driver.json").read_text())
    record["resource_peaks"] = peaks(sampler.samples)
    record["stats_samples"] = len(sampler.samples)
    workload_sha = "sha256:" + hashlib.sha256(json.dumps(results["args"] | {"nonce": None}, sort_keys=True).encode()).hexdigest()
    corpus_sha = "sha256:" + hashlib.sha256(b"service-baseline-corpus-v1:unique-notes-by-nonce-and-index").hexdigest()
    manifest = build_manifest(f"{args.label}-{nonce}", args.tree, "sha256:" + args.lock_sha256.removeprefix("sha256:"),
                              record["image_facts"]["id"], results, record["resource_peaks"], started, finished, corpus_sha, workload_sha)
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    (out_dir / "record.json").write_text(json.dumps(record, indent=2, default=str))
    (out_dir / "stats_samples.json").write_text(json.dumps(sampler.samples))


if __name__ == "__main__":
    main()
