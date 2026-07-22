"""Tests for CBM standalone adapter.

Pure-python tests mock the CBM CLI transport via a patched rail so they run
without the real binary. A live smoke test is guarded on the real install.
"""

import json
import stat
import pytest
from pathlib import Path
from unittest.mock import patch

from icebench.adapters.base import Corpus, UnsupportedOp, OP_CLASSES
from icebench.rail import RailConfig, RailResult
from icebench.systems.cbm_standalone import CBMStandaloneAdapter
from icebench.systems.cbm_standalone.adapter import _cypher_quote

REAL_CBM = "/data/ice/tools/cbm/codebase-memory-mcp"


def _rail_result(returncode=0, stdout="", stderr="", timed_out=False, oom=False):
    return RailResult(
        returncode=returncode, stdout=stdout, stderr=stderr,
        wall_s=1.0, peak_rss_mb=55.0, cpu_s=0.5,
        timed_out=timed_out, oom_killed=oom,
        memory_cap_mb=12288, timeout_s=3600, mechanism="systemd",
    )


@pytest.fixture
def fake_bin(tmp_path):
    b = tmp_path / "cbm"
    b.write_text("#!/bin/sh\nexit 0\n")
    b.chmod(b.stat().st_mode | stat.S_IEXEC)
    return str(b)


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    return Corpus(name="t", path=str(d), repo_sha="s", language="python", loc=10, file_count=1)


def _adapter(fake_bin, tmp_path):
    return CBMStandaloneAdapter(cbm_bin=fake_bin, cache_dir=str(tmp_path / "cache"))


def test_capabilities(fake_bin, tmp_path):
    a = _adapter(fake_bin, tmp_path)
    caps = a.capabilities()
    # Phase 0: nl_locate wired via search_code (semantic vector search).
    assert caps == {"symbol_lookup", "neighbors_1hop", "path_le4", "nl_locate"}
    assert caps <= OP_CLASSES
    assert a.name == "cbm"


def test_unsupported_ops_raise(fake_bin, tmp_path, corpus):
    a = _adapter(fake_bin, tmp_path)
    # blast_radius remains N/A for CBM (detect_changes is git-diff based).
    with pytest.raises(UnsupportedOp):
        a.query("blast_radius", {"corpus": corpus, "symbol": "x"})


def test_index_missing_binary(corpus, tmp_path):
    a = CBMStandaloneAdapter(cbm_bin="/no/such/cbm", cache_dir=str(tmp_path))
    r = a.index_cold(corpus)
    assert not r.ok and r.dnf and "not found" in r.dnf_reason


def test_index_incremental_na(fake_bin, tmp_path, corpus):
    a = _adapter(fake_bin, tmp_path)
    r = a.index_incremental(corpus, ["f.py"])
    assert not r.ok and r.dnf and r.dnf_reason == "incremental_na"


def test_index_cold_parses_and_caches_project(fake_bin, tmp_path, corpus):
    """index_cold routes through rail, parses CBM JSON, caches the slug."""
    a = _adapter(fake_bin, tmp_path)
    payload = json.dumps({"project": "slug-x", "nodes": 24, "edges": 46, "status": "indexed"})
    with patch("icebench.systems.cbm_standalone.adapter.run_with_rail",
               return_value=_rail_result(stdout=payload)) as m:
        r = a.index_cold(corpus)
    assert m.called  # rail used for indexing
    assert r.ok and r.symbols == 24 and r.edges == 46
    assert r.peak_rss_mb == 55.0  # rail resources carried through
    assert a._project_names[corpus.name] == "slug-x"  # slug cached


def test_index_cold_deletes_cached_project(fake_bin, tmp_path, corpus):
    """index_cold clears a pre-resolved project before indexing (true cold)."""
    a = _adapter(fake_bin, tmp_path)
    a._project_names[corpus.name] = "slug-x"  # pre-resolved => delete path taken
    index_payload = json.dumps(
        {"project": "slug-x", "nodes": 5, "edges": 3, "status": "indexed"}
    )
    calls = []

    def fake_run(cmd, cfg, env=None):
        tool = cmd[2]  # [bin, "cli", <tool>, <json>]
        calls.append(tool)
        if tool == "delete_project":
            return _rail_result(stdout='{"status":"deleted"}')
        return _rail_result(stdout=index_payload)

    with patch("icebench.systems.cbm_standalone.adapter.run_with_rail", side_effect=fake_run):
        r = a.index_cold(corpus)
    assert "delete_project" in calls  # delete ran before index
    assert r.ok and r.symbols == 5


def test_index_cold_delete_failure_is_dnf(fake_bin, tmp_path, corpus):
    """A failed cold delete must be a loud DNF, never a warm 'cold' number."""
    a = _adapter(fake_bin, tmp_path)
    a._project_names[corpus.name] = "slug-x"  # pre-resolved => delete path taken

    def fake_run(cmd, cfg, env=None):
        tool = cmd[2]
        if tool == "delete_project":
            # Simulate a failed delete (nonzero exit).
            return _rail_result(returncode=1, stderr="delete failed")
        # index_repository must NOT be reached; if it is, flag it loudly.
        raise AssertionError("index_repository ran despite failed delete")

    with patch("icebench.systems.cbm_standalone.adapter.run_with_rail", side_effect=fake_run):
        r = a.index_cold(corpus)
    assert not r.ok and r.dnf
    assert "cold_delete_failed" in r.dnf_reason


def test_index_dnf_on_oom(fake_bin, tmp_path, corpus):
    a = _adapter(fake_bin, tmp_path)
    with patch("icebench.systems.cbm_standalone.adapter.run_with_rail",
               return_value=_rail_result(returncode=137, oom=True)):
        r = a.index_cold(corpus)
    assert not r.ok and r.dnf and "oom" in r.dnf_reason.lower()


def test_index_second_sigabrt_is_dnf(fake_bin, tmp_path, corpus):
    """The SIGABRT stability probe: returncode 134 => DNF, never hidden."""
    a = _adapter(fake_bin, tmp_path)
    with patch("icebench.systems.cbm_standalone.adapter.run_with_rail",
               return_value=_rail_result(returncode=134)):
        r = a.index_second(corpus)
    assert not r.ok and r.dnf and "SIGABRT" in r.dnf_reason


def test_query_marshalling_symbol_lookup(fake_bin, tmp_path, corpus):
    """symbol_lookup -> search_graph with the resolved project + name_pattern."""
    a = _adapter(fake_bin, tmp_path)
    a._project_names[corpus.name] = "slug-x"  # pre-resolve
    seen = {}

    def fake_run(cmd, cfg, cwd=None, env=None):
        # cmd = [bin, "cli", <tool>, <json-args>]
        seen["tool"] = cmd[2]
        seen["args"] = json.loads(cmd[3])
        return _rail_result(stdout=json.dumps({"total": 1, "results": [{"name": "add"}]}))

    with patch("icebench.systems.cbm_standalone.adapter.run_with_rail", side_effect=fake_run):
        q = a.query("symbol_lookup", {"corpus": corpus, "symbol": "add"})
    assert seen["tool"] == "search_graph"
    assert seen["args"]["project"] == "slug-x"
    assert seen["args"]["name_pattern"] == ".*add.*"
    assert q.ok


def test_query_marshalling_neighbors(fake_bin, tmp_path, corpus):
    """neighbors_1hop -> trace_path depth=1 direction=both."""
    a = _adapter(fake_bin, tmp_path)
    a._project_names[corpus.name] = "slug-x"
    seen = {}

    def fake_run(cmd, cfg, cwd=None, env=None):
        seen["tool"] = cmd[2]
        seen["args"] = json.loads(cmd[3])
        return _rail_result(stdout=json.dumps({"function": "add", "callers": [], "callees": []}))

    with patch("icebench.systems.cbm_standalone.adapter.run_with_rail", side_effect=fake_run):
        q = a.query("neighbors_1hop", {"corpus": corpus, "symbol": "add"})
    assert seen["tool"] == "trace_path"
    assert seen["args"] == {"project": "slug-x", "function_name": "add",
                            "direction": "both", "depth": 1}
    assert q.ok


def test_query_path_escapes_cypher(fake_bin, tmp_path, corpus):
    """path_le4 -> query_graph with single-quote-escaped names (injection guard)."""
    a = _adapter(fake_bin, tmp_path)
    a._project_names[corpus.name] = "slug-x"
    seen = {}

    def fake_run(cmd, cfg, cwd=None, env=None):
        seen["tool"] = cmd[2]
        seen["args"] = json.loads(cmd[3])
        return _rail_result(stdout=json.dumps({"columns": ["b.name"], "rows": [["add"]]}))

    with patch("icebench.systems.cbm_standalone.adapter.run_with_rail", side_effect=fake_run):
        # A name containing a single quote must be doubled, not injected.
        q = a.query("path_le4", {"corpus": corpus, "from": "o'brien", "to": "add"})
    assert seen["tool"] == "query_graph"
    cypher = seen["args"]["query"]
    assert "o''brien" in cypher  # escaped
    assert "[*1..4]" in cypher   # bounded path length
    assert "$" not in cypher     # no unsupported params
    assert q.ok


def test_cypher_quote():
    assert _cypher_quote("a'b") == "a''b"
    assert _cypher_quote("plain") == "plain"


def test_query_requires_indexed_project(fake_bin, tmp_path, corpus):
    """Without a resolved project (not indexed), query returns an error, no crash."""
    a = _adapter(fake_bin, tmp_path)
    with patch("icebench.systems.cbm_standalone.adapter.run_with_rail",
               return_value=_rail_result(stdout=json.dumps({"projects": []}))):
        q = a.query("symbol_lookup", {"corpus": corpus, "symbol": "add"})
    assert not q.ok and "not indexed" in q.answer["error"]


def test_store_size_scoped_to_corpus(fake_bin, tmp_path, corpus):
    """store_size_bytes returns THIS corpus's size_bytes, not a global sum."""
    a = _adapter(fake_bin, tmp_path)
    real = str(Path(corpus.path).resolve())
    projects = {"projects": [
        {"name": "slug-x", "root_path": real, "size_bytes": 1769472},
        {"name": "other", "root_path": "/some/other/repo", "size_bytes": 9999999},
    ]}
    with patch("icebench.systems.cbm_standalone.adapter.run_with_rail",
               return_value=_rail_result(stdout=json.dumps(projects))):
        size = a.store_size_bytes(corpus)
    assert size == 1769472  # only this corpus, NOT summed with "other"


def test_store_size_zero_when_absent(fake_bin, tmp_path, corpus):
    a = _adapter(fake_bin, tmp_path)
    with patch("icebench.systems.cbm_standalone.adapter.run_with_rail",
               return_value=_rail_result(stdout=json.dumps({"projects": []}))):
        assert a.store_size_bytes(corpus) == 0


def test_snapshot_roundtrip(fake_bin, tmp_path, corpus):
    a = _adapter(fake_bin, tmp_path)
    a._project_names[corpus.name] = "slug-x"
    # Create a fake CBM store file
    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "slug-x.db").write_bytes(b"SQLITE-STORE-BYTES")

    snap = a.export_snapshot(corpus)
    assert snap and snap.ok and snap.bytes == len(b"SQLITE-STORE-BYTES")
    blob = Path(corpus.path) / f"snapshot-{corpus.name}.db"
    imp = a.import_snapshot(corpus, str(blob))
    assert imp and imp.ok


def test_export_snapshot_na_without_store(fake_bin, tmp_path, corpus):
    a = _adapter(fake_bin, tmp_path)
    a._project_names[corpus.name] = "slug-x"  # resolved but no db file
    assert a.export_snapshot(corpus) is None


def test_conformance(fake_bin, tmp_path):
    a = _adapter(fake_bin, tmp_path)
    for m in ("capabilities", "index_cold", "index_incremental", "index_second",
              "store_size_bytes", "export_snapshot", "import_snapshot", "query",
              "teardown"):
        assert callable(getattr(a, m))
    assert isinstance(a.name, str) and isinstance(a.version, str)


# ---- Live smoke test (guarded on the real install) ----

@pytest.mark.skipif(not Path(REAL_CBM).exists(), reason="cbm binary not installed")
def test_live_index_query_teardown(tmp_path):
    """End-to-end against the real CBM 0.9.0 on a tiny fixture."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n\n"
        "def use():\n    return add(1, 2)\n"
    )
    corpus = Corpus(name="livecbm", path=str(repo), repo_sha="s",
                    language="python", loc=5, file_count=1)
    a = CBMStandaloneAdapter(
        cbm_bin=REAL_CBM,
        cache_dir=str(tmp_path / "cbm_cache"),
        rail=RailConfig(memory_limit_mb=4096, timeout_seconds=120),
    )
    try:
        r = a.index_cold(corpus)
        assert r.ok, f"index failed: {r.dnf_reason}"
        assert r.symbols > 0
        assert a.store_size_bytes(corpus) > 0
        q = a.query("neighbors_1hop", {"corpus": corpus, "symbol": "add"})
        assert q.ok
    finally:
        a.teardown(corpus)
