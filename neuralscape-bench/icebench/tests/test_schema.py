"""Tests for schema and manifest."""

import json
import tempfile
from pathlib import Path

import pytest

from icebench.schema import (
    ResultRow,
    write_row,
    read_rows,
    RunManifest,
    SCHEMA_VERSION,
)


def test_result_row_creation():
    """Test creating a result row."""
    row = ResultRow(
        schema=SCHEMA_VERSION,
        kind="index",
        system="ns-ice",
        system_version="1.0.0",
        corpus="small-py",
        repo_sha="abc123",
        op="index_cold",
        rep=0,
        seed=42,
        wall_s=10.5,
        peak_rss_mb=512,
        cpu_s=8.2,
        ok=True,
    )

    assert row.schema == SCHEMA_VERSION
    assert row.kind == "index"
    assert row.system == "ns-ice"
    assert row.ok is True
    assert row.dnf is False
    assert row.ts is not None


def test_write_and_read_rows():
    """Test writing and reading JSONL rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_file = Path(tmpdir) / "results.jsonl"

        # Write some rows
        rows = [
            ResultRow(
                schema=SCHEMA_VERSION,
                kind="index",
                system="ns-ice",
                system_version="1.0.0",
                corpus="small-py",
                repo_sha="abc123",
                op="index_cold",
                rep=i,
                seed=42,
                wall_s=10.0 + i,
                ok=True,
            )
            for i in range(3)
        ]

        for row in rows:
            write_row(results_file, row)

        # Read back
        read_back = list(read_rows(results_file))
        assert len(read_back) == 3
        assert all(r.system == "ns-ice" for r in read_back)
        assert [r.rep for r in read_back] == [0, 1, 2]


def test_manifest_resume():
    """Test manifest resumability."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_file = Path(tmpdir) / "results.jsonl"

        # Write some completed cells
        write_row(
            results_file,
            ResultRow(
                schema=SCHEMA_VERSION,
                kind="index",
                system="ns-ice",
                system_version="1.0.0",
                corpus="small-py",
                repo_sha="abc123",
                op="index_cold",
                rep=0,
                seed=42,
                wall_s=10.0,
                ok=True,
            ),
        )
        write_row(
            results_file,
            ResultRow(
                schema=SCHEMA_VERSION,
                kind="index",
                system="ns-ice",
                system_version="1.0.0",
                corpus="small-py",
                repo_sha="abc123",
                op="index_cold",
                rep=1,
                seed=42,
                wall_s=10.5,
                ok=True,
            ),
        )

        # Load manifest
        manifest = RunManifest.load("test-run", results_file)

        # Check completed cells
        assert manifest.is_completed("ns-ice", "small-py", "index_cold", 0)
        assert manifest.is_completed("ns-ice", "small-py", "index_cold", 1)
        assert not manifest.is_completed("ns-ice", "small-py", "index_cold", 2)
        assert not manifest.is_completed("ns-graphify", "small-py", "index_cold", 0)

        # Mark a new cell
        manifest.mark_completed("ns-ice", "small-py", "index_cold", 2)
        assert manifest.is_completed("ns-ice", "small-py", "index_cold", 2)


def test_dnf_row():
    """Test DNF (did not finish) row."""
    row = ResultRow(
        schema=SCHEMA_VERSION,
        kind="index",
        system="cbm",
        system_version="1.0.0",
        corpus="large-py",
        repo_sha="abc123",
        op="index_cold",
        rep=0,
        seed=42,
        wall_s=3600.0,
        ok=False,
        dnf=True,
        dnf_reason="timeout",
    )

    assert row.dnf is True
    assert row.dnf_reason == "timeout"
    assert row.ok is False
