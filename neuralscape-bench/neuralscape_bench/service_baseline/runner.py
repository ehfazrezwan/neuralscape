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

from neuralscape_bench.service_baseline.redact import redact_value
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
SERVICES = ("api", "worker", "graph-worker", "ingest-worker", "ledger-worker", "fake-provider", "neo4j", "redis", "qdrant", "pg")
MEM_LIMITS = {"pg": 1 << 30, "api": 3 << 30, "ledger-worker": 2 << 30, "worker": 2 << 30, "graph-worker": 2 << 30, "ingest-worker": 2 << 30,
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
        self.other_containers: set[str] = set()

    def run(self) -> None:
        while not self.stop.is_set():
            taken = time.time()
            everything = sh("docker", "ps", "--format", "{{.Names}}", check=False).split()
            self.other_containers.update(n for n in everything if not n.startswith(f"{PROJECT}-"))
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


TOKEN_SECRET = "bench-token-secret"
# Durability arms: environment passed to the compose stack.
PG_OWNER_PASSWORD = "bench-owner-password"
PG_RUNTIME_PASSWORD = "bench-runtime-password"
PG_DSN = "host=pg dbname=postgres user=ns_owner password=" + PG_OWNER_PASSWORD + " options='-c search_path={schema}'"
PG_RUNTIME_DSN = "host=pg dbname=postgres user=ns_runtime password=" + PG_RUNTIME_PASSWORD + " options='-c search_path={schema}'"
ARMS: dict[str, dict[str, str]] = {
    # Least-privilege runtime role: the owner migrates and grants beforehand (images that support it).
    "ledger-pg-rt": {"LEDGER_ENABLED": "true", "LEDGER_BACKEND": "postgres",
                     "LEDGER_POSTGRES_DSN": PG_RUNTIME_DSN.format(schema="ledger"),
                     "LEDGER_POSTGRES_MIGRATE_ON_OPEN": "false"},
    "plain-w2": {"UVICORN_WORKERS": "2"},
    "plain-w4": {"UVICORN_WORKERS": "4"},
    "ledger-pg": {"LEDGER_ENABLED": "true", "LEDGER_BACKEND": "postgres",
                  "LEDGER_POSTGRES_DSN": PG_DSN.format(schema="ledger")},
    "governed-pg": {"LEDGER_ENABLED": "true", "LEDGER_BACKEND": "postgres",
                    "LEDGER_POSTGRES_DSN": PG_DSN.format(schema="ledger"),
                    "AUTHORITY_ENABLED": "true", "AUTHORITY_BACKEND": "postgres",
                    "AUTHORITY_POSTGRES_DSN": PG_DSN.format(schema="authority")},
    "plain": {},
    "ledger": {"LEDGER_ENABLED": "true"},
    "governed": {"LEDGER_ENABLED": "true", "AUTHORITY_ENABLED": "true"},
    "enforce": {"LEDGER_ENABLED": "true", "AUTHORITY_ENABLED": "true", "NS_POLICY_MODE": "enforce",
                "NS_AUTH_PROFILE": "enforced", "HARNESS_WIRE_ENFORCER": "1"},
}
OVERRIDE_ENV = "NS_BENCH_COMPOSE_OVERRIDE"
# The checked-in override that makes the stack's default network internal (no route out of the host).
NO_EGRESS_OVERRIDE = BENCH_DIR / "scripts" / "service_baseline" / "compose.no-egress.yml"
# Set by main() from --compose-override or the environment; None keeps the compose file exactly as shipped.
COMPOSE_OVERRIDE: Path | None = None


def resolve_override(argument: str | None, environ: dict[str, str] | None = None) -> Path | None:
    """The compose override to apply: the argument wins over the environment variable; a missing file is an error."""

    chosen = argument or (os.environ if environ is None else environ).get(OVERRIDE_ENV) or ""
    if not chosen.strip():
        return None
    path = Path(chosen)
    if not path.is_file():
        raise ValueError(f"compose override {chosen!r} is not a file")
    return path.resolve()


def compose_argv(*args: str, override: Path | None = None) -> list[str]:
    files = ["-f", str(COMPOSE_FILE)] + (["-f", str(override)] if override is not None else [])
    return ["docker", "compose", "-p", PROJECT, *files, *args]


def compose(*args: str, env: dict[str, str]) -> str:
    return sh(*compose_argv(*args, override=COMPOSE_OVERRIDE), env=env)


def assert_network_internal() -> bool:
    """With an override the stack's network must really be internal: egress-free by construction, checked on Docker."""

    verdict = sh("docker", "network", "inspect", f"{PROJECT}_default", "--format", "{{.Internal}}").strip()
    if verdict != "true":
        raise RuntimeError(f"the bench network {PROJECT}_default is not internal (Internal={verdict!r}): refusing to measure with egress")
    return True


def start_postgres(env: dict[str, str]) -> None:
    """Start PostgreSQL and create a non-superuser owner role with one schema per store."""

    compose("up", "-d", "--wait", "pg", env=env)
    statements = (
        f"CREATE ROLE ns_owner LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD '{PG_OWNER_PASSWORD}'; "
        "CREATE SCHEMA ledger AUTHORIZATION ns_owner; CREATE SCHEMA authority AUTHORIZATION ns_owner;"
    )
    compose("exec", "-T", "pg", "psql", "-U", "postgres", "-v", "ON_ERROR_STOP=1", "-c", statements, env=env)
    if env.get("LEDGER_POSTGRES_MIGRATE_ON_OPEN") == "false":
        compose("exec", "-T", "pg", "psql", "-U", "postgres", "-v", "ON_ERROR_STOP=1", "-c",
                f"CREATE ROLE ns_runtime LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD '{PG_RUNTIME_PASSWORD}'", env=env)
        owner_dsn = PG_DSN.format(schema="ledger")
        for action, extra in (("migrate", []), ("grant", ["--role", "ns_runtime"]), ("check", ["--role", "ns_runtime"])):
            compose("run", "--rm", "--no-deps", "--workdir", "/app/neuralscape-service", "--entrypoint", "python", "seed",
                    "-m", "ledger.admin", action, "--ledger-dsn", owner_dsn, *extra, env=env)


def backend_evidence(env: dict[str, str]) -> dict[str, Any]:
    """Show which backends the run really used, from the services themselves.

    Ledger: the line the API logs when the durable ledger opens. Authority store: the
    tables and row counts in the configured store (an empty or missing store is evidence too).
    """

    evidence: dict[str, Any] = {}
    try:
        logs = compose("logs", "--no-color", "api", env=env)
        evidence["ledger_log_lines"] = [ln.split("|", 1)[-1].strip() for ln in logs.splitlines()
                                        if "durable ledger enabled" in ln][:2]
    except Exception as error:  # noqa: BLE001 - evidence gathering must not fail a run
        evidence["ledger_log_error"] = str(error)[:200]
    if env.get("LEDGER_ENABLED") == "true":
        try:
            names = compose("exec", "-T", "redis", "redis-cli", "--scan", "--pattern", "arq:result:ledger-wake:*", env=env).split()
            queued = compose("exec", "-T", "redis", "redis-cli", "zcard", "neuralscape:queue:ledger", env=env).strip()
            intents = {n.split("ledger-wake:", 1)[1].rsplit(":", 1)[0] for n in names}
            evidence["wake_jobs"] = {"results": len(names), "distinct_intents": len(intents), "still_queued": int(queued or 0)}
        except Exception as error:  # noqa: BLE001
            evidence["wake_jobs_error"] = str(error)[:200]
    if env.get("AUTHORITY_ENABLED") == "true":
        try:
            if env.get("AUTHORITY_BACKEND") == "postgres":
                tables = compose("exec", "-T", "pg", "psql", "-U", "postgres", "-tA", "-c",
                                 "select table_name from information_schema.tables where table_schema='authority' order by 1",
                                 env=env).split()
                counts = {t: int(compose("exec", "-T", "pg", "psql", "-U", "postgres", "-tA", "-c",
                                         f'select count(*) from authority."{t}"', env=env).strip()) for t in tables}
                evidence["authority"] = {"backend": "postgres", "tables": counts}
            else:
                code = ("import sqlite3,json;c=sqlite3.connect('/state/authority.db');"
                        "t=[r[0] for r in c.execute(\"select name from sqlite_master where type='table' and name not like 'sqlite_%'\")];"
                        "print(json.dumps({x:c.execute('select count(*) from \"%s\"'%x).fetchone()[0] for x in t}))")
                out = compose("exec", "-T", "api", "python", "-c", code, env=env).strip().splitlines()[-1]
                evidence["authority"] = {"backend": "sqlite", "tables": json.loads(out)}
        except Exception as error:  # noqa: BLE001
            evidence["authority_error"] = str(error)[:200]
    return evidence


def cold_start_seconds(env: dict[str, str], timeout_s: float = 420) -> float:
    start = time.monotonic()
    compose("up", "-d", "--wait", "--wait-timeout", str(int(timeout_s)), "neo4j", "redis", "qdrant", "fake-provider", env=env)
    started = ["api", "worker", "graph-worker", "ingest-worker"]
    if env.get("LEDGER_ENABLED") == "true":
        started.append("ledger-worker")
    compose("up", "-d", *started, env=env)
    while time.monotonic() - start < timeout_s:
        probe = subprocess.run(
            compose_argv("exec", "-T", "api", "python", "-c",
                         "import urllib.request;urllib.request.urlopen('http://localhost:8199/health/live',timeout=3)", override=COMPOSE_OVERRIDE),
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
        if "accept" in cell:
            series = [("accept", cell["accept"])]
            if cell["completion"]["n"]:
                series.append(("complete", cell["completion"]))
        else:
            series = [("latency", cell["latency"])]
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
    parser.add_argument("--arm", default="plain", choices=sorted(ARMS))
    parser.add_argument("--series", default="", help="series id for interleaved comparisons")
    parser.add_argument("--position", type=int, default=0, help="position of this run within its repetition")
    parser.add_argument("--rep", type=int, default=0, help="repetition number within the series")
    parser.add_argument("--compose-override", default=None,
                        help=f"extra compose file merged over the stack (e.g. {NO_EGRESS_OVERRIDE.name}); default: ${OVERRIDE_ENV}")
    args = parser.parse_args()
    global COMPOSE_OVERRIDE
    COMPOSE_OVERRIDE = resolve_override(args.compose_override)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nonce = hashlib.sha256(f"{args.label}{time.time()}".encode()).hexdigest()[:10]
    env = {"NS_IMAGE": args.image, "BENCH_DIR": str(BENCH_DIR), "OUT_DIR": str(out_dir),
           "TOKEN_SECRET": TOKEN_SECRET, **ARMS[args.arm]}
    record: dict[str, Any] = {"series": args.series, "position_in_rep": args.position, "rep": args.rep,
                              "arm": args.arm, "arm_env": ARMS[args.arm], "label": args.label, "tree": args.tree, "image": args.image,
                              "image_facts": image_facts(args.image), "host_before": host_facts()}
    sampler = StatsSampler()
    started = datetime.now(timezone.utc)
    try:
        if env.get("LEDGER_BACKEND") == "postgres":
            start_postgres(env)
        compose("run", "--rm", "--no-deps", "seed", env=env)
        record["cold_start_seconds"] = cold_start_seconds(env)
        record["compose_override"] = str(COMPOSE_OVERRIDE) if COMPOSE_OVERRIDE else None
        if COMPOSE_OVERRIDE is not None:
            record["network_internal"] = assert_network_internal()
        sampler.start()
        compose("run", "--rm", "--no-deps", "driver", "--base-url", "http://api:8199",
                "--out", "/out/driver.json", "--nonce", nonce, *args.driver_args.split(), env=env)
    finally:
        sampler.stop.set()
        if sampler.is_alive():
            sampler.join(timeout=30)
        finished = datetime.now(timezone.utc)
        record["backend_evidence"] = backend_evidence(env)
        record["logs_tail"] = {s: compose("logs", "--no-color", "--tail", "40", s, env=env) for s in ("api", "worker", "graph-worker")}
        record["policy_log_lines"] = [line for line in compose("logs", "--no-color", "api", env=env).splitlines()
                                       if "policy_decision" in line or "policy" in line.lower() and "denied" in line.lower()][:20]
        compose("--profile", "ledger", "--profile", "pg", "down", "-v", "--remove-orphans", env=env)
    record["host_after"] = host_facts()
    results = json.loads((out_dir / "driver.json").read_text())
    record["resource_peaks"] = peaks(sampler.samples)
    record["stats_samples"] = len(sampler.samples)
    record["other_containers_during_run"] = sorted(sampler.other_containers)
    workload_sha = "sha256:" + hashlib.sha256(json.dumps(results["args"] | {"nonce": None}, sort_keys=True).encode()).hexdigest()
    corpus_sha = "sha256:" + hashlib.sha256(b"service-baseline-corpus-v1:unique-notes-by-nonce-and-index").hexdigest()
    (out_dir / "record.json").write_text(json.dumps(redact_value(record), indent=2, default=str))
    (out_dir / "stats_samples.json").write_text(json.dumps(sampler.samples))
    manifest = build_manifest(f"{args.label}-{nonce}", args.tree, "sha256:" + args.lock_sha256.removeprefix("sha256:"),
                              record["image_facts"]["id"], results, record["resource_peaks"], started, finished, corpus_sha, workload_sha)
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    (out_dir / "record.json").write_text(json.dumps(redact_value(record), indent=2, default=str))
    (out_dir / "stats_samples.json").write_text(json.dumps(sampler.samples))


if __name__ == "__main__":
    main()
