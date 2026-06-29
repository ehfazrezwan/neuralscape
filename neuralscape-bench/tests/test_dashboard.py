"""Light API tests for the dashboard (results dir redirected to a tmp path)."""

import json

import pytest
from fastapi.testclient import TestClient

from neuralscape_bench import dashboard


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "RESULTS_DIR", tmp_path)
    (tmp_path / "dev-1.json").write_text(json.dumps({
        "label": "dev", "target_url": "http://x", "profile": "light", "timestamp": "2026-06-22T00:00:00Z",
        "metrics": {"read": {"search": {"p95": 100.0}}, "throughput": {"writes_per_sec": 1.0}},
    }))
    (tmp_path / "perf-1.json").write_text(json.dumps({
        "label": "perf", "target_url": "http://y", "profile": "light", "timestamp": "2026-06-22T01:00:00Z",
        "metrics": {"read": {"search": {"p95": 55.0}}, "throughput": {"writes_per_sec": 4.0}},
    }))
    return TestClient(dashboard.app)


def test_list_runs(client):
    runs = client.get("/api/runs").json()["runs"]
    assert {r["label"] for r in runs} == {"dev", "perf"}
    assert all(r["kind"] == "run" for r in runs)


def test_get_run(client):
    data = client.get("/api/runs/dev-1.json").json()
    assert data["label"] == "dev"


def test_compare(client):
    cmp = client.get("/api/compare", params={"a": "dev-1.json", "b": "perf-1.json"}).json()
    c = cmp["comparison"]
    assert c["read.search.p95"]["improved"] is True          # 100 -> 55 (lower = better)
    assert c["throughput.writes_per_sec"]["improved"] is True  # 1 -> 4 (higher = better)


def test_missing_run_404(client):
    assert client.get("/api/runs/nope.json").status_code == 404


def test_path_traversal_blocked(client):
    # name with a slash can't escape RESULTS_DIR
    assert client.get("/api/runs/..%2f..%2fetc%2fpasswd").status_code in (404, 400)
