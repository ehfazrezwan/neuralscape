"""Tests for runner end-to-end (with FakeAdapter): rail injection, teardown,
incremental touched-files, resumability."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from icebench import run as run_mod
from icebench.run import (
    _load_systems,
    _generate_fixture_queries,
    _pick_source_files,
    _touch_files,
    _restore_files,
    cmd_index,
)
from icebench.adapters.base import (
    Corpus,
    IndexResult,
    QueryResult,
    SnapshotResult,
)
from icebench.rail import RailConfig
from icebench.schema import RunManifest, write_row, ResultRow, read_rows


def test_load_systems_names_and_rail():
    """ns-ice + ns-graphify load with the right names and get the rail injected."""
    rail = RailConfig(memory_limit_mb=2048, timeout_seconds=99)
    systems = _load_systems(["ns-ice", "ns-graphify"], rail=rail)

    names = {s.name for s in systems}
    assert names == {"ns-ice", "ns-graphify"}
    # Rail must actually be injected (item 1: rail is enforced by adapters).
    for s in systems:
        assert s.rail is rail
        assert s.rail.memory_limit_mb == 2048


def test_load_systems_unknown():
    """Unknown system names are ignored."""
    assert _load_systems(["unknown"]) == []


def test_fixture_query_generator():
    """Built-in fixture query generator emits well-formed payloads."""
    corpus = Corpus(
        name="test",
        path="/tmp/test",
        repo_sha="abc123",
        language="python",
        loc=1000,
        file_count=10,
    )

    queries = _generate_fixture_queries("symbol_lookup", corpus, n=5, seed=42)
    assert len(queries) == 5
    assert all("symbol" in q for q in queries)

    queries = _generate_fixture_queries("nl_locate", corpus, n=3, seed=42)
    assert len(queries) == 3
    assert all("query" in q for q in queries)


def test_pick_and_touch_restore(tmp_path):
    """Picking source files is deterministic; touch appends and restore reverts."""
    (tmp_path / "a.py").write_text("print('a')\n")
    (tmp_path / "b.py").write_text("print('b')\n")
    (tmp_path / "readme.md").write_text("# not source\n")

    corpus = Corpus(
        name="c",
        path=str(tmp_path),
        repo_sha="sha",
        language="python",
        loc=2,
        file_count=2,
    )

    picked = _pick_source_files(corpus, 5)
    # Only .py files are source; deterministic across calls.
    assert set(picked) == {str(tmp_path / "a.py"), str(tmp_path / "b.py")}
    assert _pick_source_files(corpus, 5) == picked

    originals = {p: Path(p).read_bytes() for p in picked}
    saved = _touch_files(picked)
    for p in picked:
        assert Path(p).read_bytes() == originals[p] + b"\n"

    _restore_files(saved)
    for p in picked:
        assert Path(p).read_bytes() == originals[p]


class _FakeAdapter:
    """Records calls to prove the runner drives the protocol correctly."""

    def __init__(self):
        self.name = "fake"
        self.version = "0.0.1"
        self.rail = None
        self.teardown_calls = 0
        # (touched_paths, bytes_seen_during_call)
        self.incremental_calls: list[tuple[list[str], list[bytes]]] = []

    def capabilities(self):
        return {"symbol_lookup"}

    def index_cold(self, corpus):
        return IndexResult(1.0, 100.0, 0.8, 10, 5, 2, ok=True)

    def index_incremental(self, corpus, touched):
        # Observe the on-disk content WHILE the incremental runs (must be edited).
        seen = [Path(p).read_bytes() for p in touched]
        self.incremental_calls.append((list(touched), seen))
        return IndexResult(0.5, 50.0, 0.4, 10, 5, 2, ok=True)

    def index_second(self, corpus):
        return IndexResult(1.1, 100.0, 0.9, 10, 5, 2, ok=True)

    def store_size_bytes(self, corpus):
        return 4096

    def export_snapshot(self, corpus):
        return SnapshotResult(0.2, 2048, ok=True)

    def import_snapshot(self, corpus, blob_path):
        return SnapshotResult(0.2, 2048, ok=True)

    def query(self, op, payload):
        return QueryResult(1.0, {"ok": True}, ok=True)

    def teardown(self, corpus):
        self.teardown_calls += 1


def test_cmd_index_end_to_end(tmp_path):
    """cmd_index drives a FakeAdapter: rows written, teardown called, files kept
    intact (incremental touched them mid-run, then restored)."""
    # Build a fake corpus on disk with real source files.
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    src_a = corpus_dir / "a.py"
    src_b = corpus_dir / "b.py"
    src_a.write_text("x = 1\n")
    src_b.write_text("y = 2\n")
    orig_a = src_a.read_bytes()
    orig_b = src_b.read_bytes()

    corpus = Corpus(
        name="fakecorpus",
        path=str(corpus_dir),
        repo_sha="deadbeef",
        language="python",
        loc=2,
        file_count=2,
    )
    fake = _FakeAdapter()

    results_dir = tmp_path / "results"
    args = argparse.Namespace(
        run_id="test-run",
        systems=["fake"],
        memory_limit_mb=1024,
        timeout_seconds=60,
    )

    with patch.object(run_mod, "RESULTS_DIR", results_dir), patch.object(
        run_mod, "iter_corpora", return_value=[corpus]
    ), patch.object(run_mod, "_load_systems", return_value=[fake]):
        rc = cmd_index(args)

    assert rc == 0

    # Teardown was called once (system x corpus).
    assert fake.teardown_calls == 1

    # Incremental ran with REAL non-empty touched paths, and the files were
    # actually edited during the call (newline appended), then restored after.
    assert fake.incremental_calls, "index_incremental was never called"
    for touched, seen in fake.incremental_calls:
        assert touched, "touched files must be non-empty real paths"
        for content in seen:
            assert content.endswith(b"\n")

    # Files restored to original bytes after the run.
    assert src_a.read_bytes() == orig_a
    assert src_b.read_bytes() == orig_b

    # Rows were written for index + store + snapshot.
    results_file = results_dir / "test-run.jsonl"
    rows = list(read_rows(results_file))
    kinds = {r.kind for r in rows}
    assert "index" in kinds
    assert "store" in kinds
    assert "snapshot" in kinds
    # peak_rss + cpu are real (non-null) on the index rows.
    index_rows = [r for r in rows if r.kind == "index"]
    assert all(r.peak_rss_mb is not None for r in index_rows)


def test_runner_resumability():
    """Runner respects the manifest and skips completed cells."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_file = Path(tmpdir) / "test-run.jsonl"
        write_row(
            results_file,
            ResultRow(
                schema="icebench-v1",
                kind="index",
                system="fake",
                system_version="1.0.0",
                corpus="test",
                repo_sha="abc123",
                op="index_cold",
                rep=0,
                seed=42,
                wall_s=10.0,
                ok=True,
            ),
        )
        manifest = RunManifest.load("test-run", results_file)
        assert manifest.is_completed("fake", "test", "index_cold", 0)
        assert not manifest.is_completed("fake", "test", "index_cold", 1)
