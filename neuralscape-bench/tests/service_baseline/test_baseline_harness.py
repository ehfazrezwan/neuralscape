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
    line = json.dumps({"Name": "ns-g-bench-api-1", "MemUsage": "525.9MiB / 3GiB", "CPUPerc": "12.50%"})
    parsed = parse_stats_line(line)
    assert parsed == {"name": "ns-g-bench-api-1", "mem_bytes": int(525.9 * 2**20), "cpu_percent": 12.5}
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
    assert cell["attempted"] == 32 and cell["warmup_excluded"] == 2
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
