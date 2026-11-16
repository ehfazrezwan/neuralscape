"""Snapshot import must be idempotent, and its live qualification must be
scoped to its own code_space (persisted-volume robustness)."""

import ast
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from adapters.code_graph.native_engine import NativeEngine

_IT = Path(__file__).resolve().parent.parent / "integration_tests" / "test_native_snapshot_neo4j.py"


def _artifact(snapshot):
    h = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    env = {
        "header": {"format_version": "1.0", "code_space": "cs", "repo": "r",
                   "symbol_count": 0, "edge_count": 0, "content_hash": h},
        "snapshot": snapshot,
    }
    return gzip.compress(json.dumps(env, sort_keys=True).encode(), mtime=1)


def _snapshot():
    a = {"code_space": "cs", "fqn": "pkg.a"}
    b = {"code_space": "cs", "fqn": "pkg.b"}
    return {
        "nodes": [
            {"labels": ["CodeSymbol", "Aux"], "properties": a},
            {"labels": ["CodeSymbol"], "properties": b},
        ],
        "edges": [{
            "type": "CALLS", "properties": {},
            "source": {"labels": ["CodeSymbol"], "properties": a},
            "target": {"labels": ["CodeSymbol"], "properties": b},
        }],
    }


def test_import_twice_issues_only_merge_writes_with_identical_statements():
    engine = NativeEngine(repo_path="/tmp/x", code_space="cs",
                          bridge=MagicMock(), settings=MagicMock())
    issued = []
    with patch.object(engine, "_run_cypher",
                      side_effect=lambda c, **p: issued.append(c) or []):
        engine.import_snapshot(_artifact(_snapshot()))
        first = list(issued)
        engine.import_snapshot(_artifact(_snapshot()))
    second = issued[len(first):]
    assert first and first == second
    assert not any("CREATE" in c.upper().replace("MERGE", "") for c in issued)
    assert all("MERGE" in c.upper() for c in issued)


def test_live_snapshot_integration_queries_are_scoped_to_code_space():
    """An unscoped ``MATCH (n)`` counts nodes leaked by earlier runs on a
    persisted Neo4j volume (gate failure: 9 == 3)."""
    tree = ast.parse(_IT.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = " ".join(node.value.split())
            if text.startswith("MATCH (n) RETURN") or text == "MATCH (n)":
                offenders.append(text)
    assert offenders == []
