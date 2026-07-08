"""Tests for Graphify standalone adapter.

Pure-python tests use a fake graphify binary + a patched rail so they run
without the real tool. A live smoke test is guarded on the real install.
"""

import json
import stat
import pytest
from pathlib import Path
from unittest.mock import patch

from icebench.adapters.base import Corpus, UnsupportedOp, OP_CLASSES
from icebench.rail import RailConfig, RailResult
from icebench.systems.graphify_standalone import GraphifyStandaloneAdapter

REAL_GRAPHIFY = "/data/ice/tools/graphify/.venv/bin/graphify"


def _rail_result(returncode=0, stdout="", stderr="", dnf_timed_out=False, oom=False):
    return RailResult(
        returncode=returncode, stdout=stdout, stderr=stderr,
        wall_s=1.0, peak_rss_mb=42.0, cpu_s=0.5,
        timed_out=dnf_timed_out, oom_killed=oom,
        memory_cap_mb=12288, timeout_s=3600, mechanism="ulimit",
    )


@pytest.fixture
def fake_bin(tmp_path):
    """A fake graphify executable so Path(bin).exists() passes."""
    b = tmp_path / "graphify"
    b.write_text("#!/bin/sh\nexit 0\n")
    b.chmod(b.stat().st_mode | stat.S_IEXEC)
    return str(b)


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    return Corpus(name="t", path=str(d), repo_sha="s", language="python", loc=10, file_count=1)


def test_capabilities(fake_bin):
    a = GraphifyStandaloneAdapter(graphify_bin=fake_bin)
    caps = a.capabilities()
    assert caps == {"symbol_lookup", "neighbors_1hop", "path_le4"}
    assert caps <= OP_CLASSES
    assert a.name == "graphify"


def test_unsupported_ops_raise(fake_bin, corpus):
    a = GraphifyStandaloneAdapter(graphify_bin=fake_bin)
    with pytest.raises(UnsupportedOp):
        a.query("nl_locate", {"corpus": corpus, "query": "x"})
    with pytest.raises(UnsupportedOp):
        a.query("blast_radius", {"corpus": corpus, "symbol": "x"})


def test_index_missing_binary(corpus):
    a = GraphifyStandaloneAdapter(graphify_bin="/no/such/graphify")
    r = a.index_cold(corpus)
    assert not r.ok and r.dnf and "not found" in r.dnf_reason


def test_index_incremental_na(fake_bin, corpus):
    a = GraphifyStandaloneAdapter(graphify_bin=fake_bin)
    r = a.index_incremental(corpus, ["f.py"])
    assert not r.ok and r.dnf and r.dnf_reason == "incremental_na"


def test_index_cold_parses_graph_json(fake_bin, corpus):
    """index_cold routes through the rail and parses graphify's graph.json."""
    a = GraphifyStandaloneAdapter(graphify_bin=fake_bin)

    def fake_run(cmd, cfg, cwd=None, env=None):
        # Simulate graphify writing graph.json
        out = Path(corpus.path) / "graphify-out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "graph.json").write_text(json.dumps({
            "nodes": [
                {"id": "a", "source_file": "calc.py"},
                {"id": "b", "source_file": "calc.py"},
                {"id": "c", "source_file": "utils.py"},
            ],
            "edges": [{"source": "a", "target": "b"}],
        }))
        return _rail_result(returncode=0)

    with patch("icebench.systems.graphify_standalone.adapter.run_with_rail", side_effect=fake_run):
        r = a.index_cold(corpus)
    assert r.ok
    assert r.symbols == 3
    assert r.edges == 1
    assert r.files == 2  # distinct source_file values
    assert r.peak_rss_mb == 42.0  # rail-measured resources carried through


def test_index_dnf_on_rail_breach(fake_bin, corpus):
    a = GraphifyStandaloneAdapter(graphify_bin=fake_bin)
    with patch("icebench.systems.graphify_standalone.adapter.run_with_rail",
               return_value=_rail_result(returncode=-1, dnf_timed_out=True)):
        r = a.index_cold(corpus)
    assert not r.ok and r.dnf and r.dnf_reason.startswith("timeout")


def test_query_routes_through_rail(fake_bin, corpus):
    """query() must call run_with_rail (not subprocess.run directly)."""
    a = GraphifyStandaloneAdapter(graphify_bin=fake_bin)
    # graph.json must exist for query to proceed
    out = Path(corpus.path) / "graphify-out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph.json").write_text('{"nodes": [], "edges": []}')

    with patch("icebench.systems.graphify_standalone.adapter.run_with_rail",
               return_value=_rail_result(returncode=0, stdout="Node: add()")) as m:
        q = a.query("symbol_lookup", {"corpus": corpus, "symbol": "add()"})
    assert m.called  # proves the rail was used
    assert q.ok and "add()" in q.answer["text"]


def test_query_requires_indexed_graph(fake_bin, corpus):
    a = GraphifyStandaloneAdapter(graphify_bin=fake_bin)
    q = a.query("symbol_lookup", {"corpus": corpus, "symbol": "x"})
    assert not q.ok and "not indexed" in q.answer["error"]


def test_snapshot_roundtrip(fake_bin, corpus):
    a = GraphifyStandaloneAdapter(graphify_bin=fake_bin)
    out = Path(corpus.path) / "graphify-out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph.json").write_text('{"nodes": [1], "edges": []}')

    snap = a.export_snapshot(corpus)
    assert snap and snap.ok and snap.bytes > 0
    # store size counts the whole graphify-out dir
    assert a.store_size_bytes(corpus) > 0
    # import into a fresh corpus location
    blob = Path(corpus.path) / f"snapshot-{corpus.name}.json"
    imp = a.import_snapshot(corpus, str(blob))
    assert imp and imp.ok


def test_export_snapshot_na_without_graph(fake_bin, corpus):
    a = GraphifyStandaloneAdapter(graphify_bin=fake_bin)
    assert a.export_snapshot(corpus) is None


def test_conformance(fake_bin):
    a = GraphifyStandaloneAdapter(graphify_bin=fake_bin)
    for m in ("capabilities", "index_cold", "index_incremental", "index_second",
              "store_size_bytes", "export_snapshot", "import_snapshot", "query",
              "teardown"):
        assert callable(getattr(a, m))
    assert isinstance(a.name, str) and isinstance(a.version, str)


# ---- Live smoke test (guarded on the real install) ----

@pytest.mark.skipif(not Path(REAL_GRAPHIFY).exists(), reason="graphify not installed")
def test_live_index_and_query(tmp_path):
    """End-to-end against the real graphify 0.9.10 on a tiny fixture."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n\n"
        "def use():\n    return add(1, 2)\n"
    )
    corpus = Corpus(name="live", path=str(repo), repo_sha="s",
                    language="python", loc=5, file_count=1)
    a = GraphifyStandaloneAdapter(
        graphify_bin=REAL_GRAPHIFY,
        rail=RailConfig(memory_limit_mb=4096, timeout_seconds=120),
    )
    try:
        r = a.index_cold(corpus)
        assert r.ok, f"index failed: {r.dnf_reason}"
        assert r.symbols > 0 and r.files >= 1
        q = a.query("symbol_lookup", {"corpus": corpus, "symbol": "add()"})
        assert q.ok
    finally:
        a.teardown(corpus)
