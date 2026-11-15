"""Mechanics of the service baseline harness: fake provider, stats, runner parsing."""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import pytest

from neuralscape_bench.service_baseline import driver, fake_provider
from neuralscape_bench.service_baseline.runner import parse_size, parse_stats_line, peaks
from neuralscape_bench.service_baseline.stats import summarize


@pytest.fixture()
def provider():
    server = fake_provider.make_server("127.0.0.1", 0, generate_latency_ms=30, embed_latency_ms=10)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def post(base: str, path: str, body: dict) -> tuple[dict, float]:
    request = urllib.request.Request(
        base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read()), time.monotonic() - start


def test_embeddings_are_deterministic_unit_vectors_of_requested_size(provider):
    body = {"content": {"parts": [{"text": "hello"}]}, "outputDimensionality": 64}
    first, _ = post(provider, "/v1beta/models/m:embedContent", body)
    second, _ = post(provider, "/models/m:embedContent", body)
    assert first == second
    assert len(first["embedding"]["values"]) == 64
    assert sum(v * v for v in first["embedding"]["values"]) == pytest.approx(1.0, abs=1e-3)
    other, _ = post(provider, "/models/m:embedContent", {"content": {"parts": [{"text": "bye"}]}})
    assert other["embedding"]["values"] != first["embedding"]["values"]


def test_batch_embeddings_one_vector_per_request_and_openai_shape(provider):
    batch, _ = post(
        provider,
        "/models/m:batchEmbedContents",
        {"requests": [{"content": {"parts": [{"text": t}]}} for t in ("a", "b", "c")]},
    )
    assert len(batch["embeddings"]) == 3
    openai, _ = post(provider, "/v1/embeddings", {"model": "x", "input": ["a", "b"]})
    assert [d["index"] for d in openai["data"]] == [0, 1]
    assert openai["data"][0]["embedding"] == batch["embeddings"][0]["values"]


def test_extraction_output_is_deterministic_and_parses_as_facts(provider):
    from collections import Counter  # noqa: F401  (keeps the import surface obvious)

    body = {"contents": "remember my editor"}
    first, elapsed = post(provider, "/v1beta/models/g:generateContent", body)
    second, _ = post(provider, "/v1beta/models/g:generateContent", body)
    text = first["candidates"][0]["content"]["parts"][0]["text"]
    assert text == second["candidates"][0]["content"]["parts"][0]["text"]
    facts = json.loads(text)["facts"]
    assert len(facts) == 2 and all(f.startswith("[") for f in facts)
    assert first["usageMetadata"]["promptTokenCount"] >= 1
    assert elapsed >= 0.03  # the configured fixed latency is honoured


def test_schema_requests_get_a_minimal_valid_instance(provider):
    schema = {
        "type": "OBJECT",
        "properties": {
            "items": {"type": "ARRAY", "items": {"type": "STRING"}},
            "count": {"type": "INTEGER"},
            "note": {"anyOf": [{"type": "null"}, {"type": "string"}]},
        },
    }
    out, _ = post(
        provider,
        "/models/g:generateContent",
        {"contents": "x", "generationConfig": {"responseSchema": schema}},
    )
    assert json.loads(out["candidates"][0]["content"]["parts"][0]["text"]) == {
        "items": [],
        "count": 0,
        "note": "",
    }
    chat, _ = post(
        provider,
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "x"}],
         "response_format": {"type": "json_schema", "json_schema": {"schema": schema}}},
    )
    assert json.loads(chat["choices"][0]["message"]["content"])["count"] == 0


def test_unknown_path_is_rejected_and_counted(provider):
    with pytest.raises(urllib.error.HTTPError) as error:
        post(provider, "/nope", {})
    assert error.value.code == 404


def test_ref_schema_resolution():
    schema = {"$defs": {"E": {"type": "object", "properties": {"n": {"type": "integer"}}}},
              "type": "object", "properties": {"e": {"$ref": "#/$defs/E"}}}
    assert fake_provider.minimal_instance(schema) == {"e": {"n": 0}}


def test_percentiles_only_reported_when_the_sample_supports_them():
    few = summarize([1.0, 2.0, 3.0])
    assert few["n"] == 3 and few["p50_ms"] is None and few["p95_ms"] is None
    assert few["min_ms"] == 1.0 and few["max_ms"] == 3.0
    some = summarize([float(i) for i in range(1, 51)])
    assert some["p50_ms"] == 25.0 and some["p95_ms"] == 48.0 and some["p99_ms"] is None
    many = summarize([float(i) for i in range(1, 201)])
    assert many["p99_ms"] == 198.0
    assert summarize([])["n"] == 0 and summarize([])["p50_ms"] is None


def test_docker_stats_parsing_and_peaks():
    line = json.dumps({"Name": "ns-bench-baseline-api-1", "MemUsage": "525.9MiB / 3GiB", "CPUPerc": "12.50%"})
    parsed = parse_stats_line(line)
    assert parsed == {"name": "ns-bench-baseline-api-1", "mem_bytes": int(525.9 * 2**20), "cpu_percent": 12.5}
    assert parse_stats_line("garbage") is None
    assert parse_size("1.5GiB") == int(1.5 * 2**30)
    samples = [parsed, {**parsed, "mem_bytes": 1, "cpu_percent": 99.0}]
    assert peaks(samples)["api"] == {"peak_mem_bytes": parsed["mem_bytes"], "peak_cpu_percent": 99.0, "samples": 2}


def test_driver_cell_accounts_failures_and_percentile_gating():
    class Stub:
        def __init__(self):
            self.calls = 0

        def request(self, method, path, body=None):
            self.calls += 1
            if self.calls % 5 == 0:
                return 500, None
            return 200, {}

    cell = driver.run_cell(Stub(), "list", 4, ops=30, warmup=2, nonce="n", poll_s=0.01, timeout_s=1)
    # warmup is raised to the concurrency: the first four operations start together
    assert cell["attempted"] == 34 and cell["warmup_excluded"] == 4
    assert cell["measured_attempts"] == 30
    assert cell["ok"] + sum(cell["errors"].values()) == 30
    assert sum(cell["errors"].values()) > 0
    assert cell["latency"]["n"] == cell["ok"]
    assert cell["driver_cpu_cores_used"] is not None


def test_raw_workloads_and_auth_header_and_enqueue_only_mode():
    seen: list[tuple[str, str, dict | None]] = []

    class Stub:
        def request(self, method, path, body=None):
            seen.append((method, path, body))
            if method == "POST":
                return 202, {"task_id": "intent-1"}
            return 200, {"status": "completed"}

    polled = driver.WORKLOADS["raw"](Stub(), "n", 0, 0.001, 1.0)
    assert polled["ok"] and "complete_ms" in polled
    assert seen[0][0:2] == ("POST", "/v1/memories/raw") and seen[0][2]["visibility"] == "private"
    assert any(path.startswith("/v1/memories/status/") for _, path, _ in seen)

    seen.clear()
    accepted = driver.WORKLOADS["raw_nopoll"](Stub(), "n", 1, 0.001, 1.0)
    assert accepted["ok"] and "complete_ms" not in accepted
    assert [m for m, _, _ in seen] == ["POST"]  # no status polling when it is denied by policy


def test_manifest_series_for_accept_only_cells_has_no_completion_measurement():
    from neuralscape_bench.service_baseline.runner import ARMS

    assert ARMS["plain"] == {}
    assert ARMS["enforce"]["NS_POLICY_MODE"] == "enforce" and ARMS["enforce"]["HARNESS_WIRE_ENFORCER"] == "1"
    assert "AUTHORITY_ENABLED" not in ARMS["ledger"]


def test_postgres_arms_use_a_non_superuser_owner_role_and_separate_schemas():
    from neuralscape_bench.service_baseline.runner import ARMS

    ledger, governed = ARMS["ledger-pg"], ARMS["governed-pg"]
    assert ledger["LEDGER_BACKEND"] == "postgres" and "AUTHORITY_ENABLED" not in ledger
    assert governed["AUTHORITY_BACKEND"] == "postgres"
    assert "user=ns_owner" in ledger["LEDGER_POSTGRES_DSN"] and "user=postgres" not in ledger["LEDGER_POSTGRES_DSN"]
    assert "search_path=ledger" in ledger["LEDGER_POSTGRES_DSN"]
    assert "search_path=authority" in governed["AUTHORITY_POSTGRES_DSN"]


def test_warmup_covers_the_concurrency_ramp_and_samples_keep_every_request():
    class Stub:
        def request(self, method, path, body=None):
            return 200, {}

    samples: list[dict] = []
    cell = driver.run_cell(Stub(), "list", 16, ops=40, warmup=5, nonce="n", poll_s=0.01, timeout_s=1, samples=samples)
    assert cell["warmup_excluded"] == 16 and cell["measured_attempts"] == 40
    assert len(samples) == 56 and sum(1 for s in samples if s["warmup"]) == 16
    assert {"workload", "concurrency", "index", "start_s", "end_s", "ok", "latency_ms"} <= set(samples[0])
    assert cell["measured_window_seconds"] > 0
    # no warmup requested means none excluded
    assert driver.run_cell(Stub(), "list", 4, ops=8, warmup=0, nonce="n", poll_s=0.01, timeout_s=1)["warmup_excluded"] == 0


def test_refused_writes_do_not_enter_the_accept_series():
    class Stub:
        def __init__(self):
            self.calls = 0

        def request(self, method, path, body=None):
            self.calls += 1
            if method == "POST" and self.calls % 2 == 0:
                return 503, None  # refused immediately
            if method == "POST":
                time.sleep(0.02)
                return 202, {"task_id": "t"}
            return 200, {"status": "completed"}

    cell = driver.run_cell(Stub(), "raw_nopoll", 1, ops=40, warmup=0, nonce="n", poll_s=0.001, timeout_s=1)
    assert sum(cell["errors"].values()) == 20
    assert cell["accept"]["n"] == 20  # only accepted requests
    assert cell["accept"]["p50_ms"] >= 15  # not dragged down by 0 ms refusals


def test_run_selection_rule_and_summary(tmp_path):
    from neuralscape_bench.service_baseline import report

    def make(name, arm, others_during, p50):
        d = tmp_path / name
        d.mkdir()
        (d / "record.json").write_text(json.dumps({
            "arm": arm, "host_before": {"other_running_containers": []},
            "host_after": {"other_running_containers": []}, "other_containers_during_run": others_during}))
        (d / "driver.json").write_text(json.dumps({"cells": [{
            "workload": "list", "concurrency": 1, "latency": {"p50_ms": p50}, "throughput_per_s": 10.0}]}))
        return d

    dirs = [make("a1", "plain", [], 5.0), make("a2", "plain", [], 7.0), make("a3", "plain", ["other-1"], 99.0)]
    runs = report.load_runs(dirs)
    assert [r["clean"] for r in runs] == [True, True, False]
    clean = report.summarize(runs, "clean")
    assert "6.0 [5.0-7.0] (n=2)" in clean and "99.0" not in clean
    assert "(n=3)" in report.summarize(runs, "all")
    assert report.is_clean({"host_before": {"other_running_containers": ["x"]}}) is False

# ---- egress-free by construction: the override, the telemetry switches, the network check

from pathlib import Path

from neuralscape_bench.service_baseline import runner

COMPOSE = Path(runner.COMPOSE_FILE)
# The installed wheel carries the stack files next to the package; a source checkout has them under scripts/.
_WHEEL_STACK = Path(runner.__file__).resolve().parent / "stack"


def stack_file(name: str) -> Path:
    return (_WHEEL_STACK if _WHEEL_STACK.is_dir() else COMPOSE.parent) / name


def test_compose_argv_without_an_override_is_exactly_the_shipped_file():
    argv = runner.compose_argv("up", "-d", "api")
    assert argv == ["docker", "compose", "-p", runner.PROJECT, "-f", str(COMPOSE), "up", "-d", "api"]


def test_compose_argv_merges_the_override_after_the_shipped_file(tmp_path):
    override = tmp_path / "o.yml"
    argv = runner.compose_argv("down", override=override)
    assert argv[:8] == ["docker", "compose", "-p", runner.PROJECT, "-f", str(COMPOSE), "-f", str(override)]
    assert argv[-1] == "down"


def test_the_override_argument_wins_over_the_environment(tmp_path):
    first, second = tmp_path / "a.yml", tmp_path / "b.yml"
    first.write_text("x: 1\n")
    second.write_text("x: 2\n")
    assert runner.resolve_override(str(first), {runner.OVERRIDE_ENV: str(second)}) == first.resolve()
    assert runner.resolve_override(None, {runner.OVERRIDE_ENV: str(second)}) == second.resolve()
    assert runner.resolve_override(None, {}) is None
    assert runner.resolve_override("  ", {runner.OVERRIDE_ENV: ""}) is None


def test_a_missing_override_file_is_an_error_not_a_silent_egress_run(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        runner.resolve_override(str(tmp_path / "missing.yml"), {})
    with pytest.raises(ValueError, match="not a file"):
        runner.resolve_override(None, {runner.OVERRIDE_ENV: str(tmp_path)})   # a directory


def test_compose_runs_use_the_module_override(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "sh", lambda *args, **kwargs: calls.append(args) or "")
    monkeypatch.setattr(runner, "COMPOSE_OVERRIDE", tmp_path / "o.yml")
    runner.compose("ps", env={})
    assert calls[0][calls[0].index("-f", 6) + 1] == str(tmp_path / "o.yml")
    monkeypatch.setattr(runner, "COMPOSE_OVERRIDE", None)
    calls.clear()
    runner.compose("ps", env={})
    assert calls[0].count("-f") == 1


def test_the_checked_in_override_makes_the_default_network_internal():
    text = stack_file(runner.NO_EGRESS_OVERRIDE.name).read_text()
    uncommented = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert uncommented.split() == ["networks:", "default:", "internal:", "true"]


@pytest.mark.parametrize("verdict, ok", [("true\n", True), ("false\n", False), ("\n", False)])
def test_the_network_must_really_be_internal(monkeypatch, verdict, ok):
    monkeypatch.setattr(runner, "sh", lambda *args, **kwargs: verdict)
    if ok:
        assert runner.assert_network_internal() is True
    else:
        with pytest.raises(RuntimeError, match="not internal"):
            runner.assert_network_internal()


def test_the_stack_switches_off_mem0_graphiti_and_qdrant_telemetry():
    text = stack_file(COMPOSE.name).read_text()
    head = text.split("x-service-mounts:")[0]
    assert 'MEM0_TELEMETRY: "False"' in head and 'GRAPHITI_TELEMETRY_ENABLED: "false"' in head
    assert 'QDRANT__TELEMETRY_DISABLED: "true"' in text
    # every NS application service takes the shared environment block
    for service in ("api", "worker", "graph-worker", "ingest-worker"):
        joined = "\n".join(text.split(f"\n  {service}:")[1].splitlines()[:8])
        assert "*service-env" in joined, service


def test_the_synthetic_provider_is_the_only_provider_in_the_stack():
    text = stack_file(COMPOSE.name).read_text()
    assert "GOOGLE_API_KEY: fake-key" in text and "FAKE_GENAI_BASE_URL: http://fake-provider:8080" in text


# ---- the runner builds a RunManifest that the current run_manifest module accepts


def test_build_manifest_is_accepted_by_the_current_run_manifest_module():
    from datetime import datetime, timedelta, timezone

    from neuralscape_bench.run_manifest import RunState

    results = {"cells": [
        {"workload": "list", "concurrency": 4, "attempted": 20, "measured_attempts": 16, "warmup_excluded": 4,
         "latency": {"n": 16}},
        {"workload": "raw", "concurrency": 8, "attempted": 16, "measured_attempts": 16, "warmup_excluded": 0,
         "accept": {"n": 15}, "completion": {"n": 10}},
        {"workload": "raw_nopoll", "concurrency": 1, "attempted": 5, "measured_attempts": 5, "warmup_excluded": 0,
         "accept": {"n": 5}, "completion": {"n": 0}},
    ]}
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    manifest = runner.build_manifest(
        "run-1", "0" * 40, "sha256:" + "1" * 64, "sha256:" + "2" * 64, results,
        {"api": {"peak_mem_bytes": 1234}}, started, started + timedelta(seconds=5),
        "sha256:" + "3" * 64, "sha256:" + "4" * 64)
    assert manifest.state is RunState.COMPLETED
    names = {m.name for m in manifest.measurements}
    assert {"list-latency-c4", "raw-accept-c8", "raw-complete-c8", "raw_nopoll-accept-c1"} <= names
    assert "raw_nopoll-complete-c1" not in names
    assert {r.scope for r in manifest.resources} == set(runner.SERVICES)
