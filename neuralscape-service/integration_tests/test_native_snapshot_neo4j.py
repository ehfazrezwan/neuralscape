"""Live Neo4j qualification for native snapshot imports.

This module is intentionally outside ``tests/`` and is invoked explicitly by
the dedicated CI job. A missing or unreachable database is a test failure.
"""

from __future__ import annotations

import asyncio
import copy
import gzip
import hashlib
import json
import os
import threading
import uuid
from types import SimpleNamespace

import pytest
from neo4j import AsyncGraphDatabase

from adapters.code_graph.native_engine import NativeEngine


def _snapshot_artifact(snapshot: dict, *, content_hash: str | None = None) -> bytes:
    snapshot_json = json.dumps(snapshot, sort_keys=True)
    code_space = next(
        (
            node.get("properties", {}).get("code_space")
            for node in snapshot["nodes"]
            if node.get("properties", {}).get("code_space")
        ),
        "code--ci--snapshot",
    )
    envelope = {
        "header": {
            "format_version": "1.0",
            "code_space": code_space,
            "repo": "snapshot-ci",
            "symbol_count": sum(
                "CodeSymbol" in node.get("labels", [])
                for node in snapshot["nodes"]
            ),
            "edge_count": len(snapshot["edges"]),
            "content_hash": content_hash
            or hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest(),
        },
        "snapshot": snapshot,
    }
    return gzip.compress(
        json.dumps(envelope, sort_keys=True).encode("utf-8"),
        mtime=1,
    )


def _submit(loop: asyncio.AbstractEventLoop, awaitable):
    return asyncio.run_coroutine_threadsafe(awaitable, loop).result(timeout=30)


@pytest.fixture(scope="session")
def live_database():
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(
        target=run_loop,
        name="snapshot-neo4j-event-loop",
        daemon=True,
    )
    thread.start()
    assert started.wait(timeout=5), "Neo4j event loop did not start"

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        _submit(loop, driver.verify_connectivity())
        yield SimpleNamespace(_loop=loop), driver
    finally:
        try:
            _submit(loop, driver.close())
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            assert not thread.is_alive(), "Neo4j event loop did not stop"
            loop.close()


@pytest.fixture
def live_engine(live_database):
    bridge, driver = live_database
    code_space = f"code--ci--snapshot-{uuid.uuid4().hex}"
    engine = NativeEngine(
        repo_path="/tmp/snapshot-ci",
        code_space=code_space,
        bridge=bridge,
        settings=SimpleNamespace(),
        driver=driver,
    )
    engine._run_cypher(
        "MATCH (n) WHERE n.code_space = $code_space DETACH DELETE n",
        code_space=code_space,
    )
    try:
        yield engine
    finally:
        engine._run_cypher(
            "MATCH (n) WHERE n.code_space = $code_space DETACH DELETE n",
            code_space=code_space,
        )


def _node_count(engine: NativeEngine) -> int:
    rows = engine._run_cypher(
        "MATCH (n) WHERE n.code_space = $code_space RETURN count(n) AS count",
        code_space=engine.code_space,
    )
    return rows[0]["count"]


def test_database_receipt_uses_neo4j_5_default_database(live_engine):
    rows = live_engine._run_cypher(
        "CALL dbms.components() YIELD name, versions, edition "
        "RETURN name, versions, edition, currentDatabase() AS database"
    )
    assert rows
    assert rows[0]["database"] == "neo4j"
    assert any(
        str(version).startswith("5.")
        for row in rows
        for version in row["versions"]
    )


@pytest.mark.parametrize(
    ("core_label", "specific_identity"),
    [
        ("CodeRepo", {}),
        ("CodeFile", {"path": "src/live test.py"}),
        ("CodeSymbol", {"fqn": "pkg.live.symbol"}),
        ("CodeAnchor", {"repo": "snapshot-ci", "fqn": "pkg.live.symbol"}),
    ],
)
def test_core_only_identity_gains_auxiliary_label_union_without_duplicates(
    live_engine,
    core_label,
    specific_identity,
):
    identity = {"code_space": live_engine.code_space, **specific_identity}
    seed = {**identity, "sentinel": "retain-me"}
    live_engine._run_cypher(
        f"CREATE (n:`{core_label}`) SET n += $properties",
        properties=seed,
    )

    auxiliary_a = ["Auxiliary with space", "punctuation:[]()"]
    auxiliary_b = ["tick`auxiliary", "雪 auxiliary", r"literal\u0060auxiliary"]
    properties_a = {**identity, "payload": "first", "first key": "first value"}
    properties_b = {**identity, "payload": "second", "雪 key": "雪 value"}

    live_engine.import_snapshot(
        _snapshot_artifact(
            {
                "nodes": [
                    {
                        "labels": [*auxiliary_a, core_label],
                        "properties": properties_a,
                    }
                ],
                "edges": [],
            }
        )
    )
    second_artifact = _snapshot_artifact(
        {
            "nodes": [
                {
                    "labels": [core_label, *auxiliary_b],
                    "properties": properties_b,
                }
            ],
            "edges": [],
        }
    )
    live_engine.import_snapshot(second_artifact)
    live_engine.import_snapshot(second_artifact)

    rows = live_engine._run_cypher(
        f"""
        MATCH (n:`{core_label}`)
        WHERE all(key IN keys($identity) WHERE n[key] = $identity[key])
        RETURN count(n) AS count,
               collect(labels(n)) AS label_sets,
               collect(properties(n)) AS property_sets
        """,
        identity=identity,
    )
    assert rows[0]["count"] == 1
    assert set(rows[0]["label_sets"][0]) == {
        core_label,
        *auxiliary_a,
        *auxiliary_b,
    }
    properties = rows[0]["property_sets"][0]
    assert properties["sentinel"] == "retain-me"
    assert properties["payload"] == "second"
    assert properties["first key"] == "first value"
    assert properties["雪 key"] == "雪 value"


def test_arbitrary_identifiers_and_property_maps_round_trip_safely(live_engine):
    source_labels = [
        "CodeSymbol",
        " ",
        " whitespace label ",
        "punctuation:[]()!?",
        "tick`) SET n.compromised = true //",
        "雪 λ label",
        r"literal\u0060) SET n.compromised = true //",
    ]
    source_properties = {
        "code_space": live_engine.code_space,
        "fqn": "pkg.source",
        " key with spaces ": " spaced value ",
        "punctuation:[]()!?": "punctuation value",
        "tick`key": "tick`value",
        "雪 key": "雪 value",
        r"literal\u0060key": r"literal\u0060value",
    }
    target_properties = {
        "code_space": live_engine.code_space,
        "fqn": "pkg.target",
        "role": "target",
    }
    anchor_properties = {
        "code_space": live_engine.code_space,
        "repo": "snapshot-ci",
        "fqn": "pkg.source",
    }
    collision_properties = {
        "src_code_space": "edge-src-value",
        "tgt_fqn": "edge-target-value",
        "code_space": "edge-code-space-value",
        "fqn": "edge-fqn-value",
        "source": "not-the-source-map",
        "target": "not-the-target-map",
        "edge_properties": "still-edge-data",
        " odd edge key ": "odd edge value",
        "tick`edge": "tick`edge value",
        "雪 edge": "雪 edge value",
        r"literal\u0060edge": r"literal\u0060edge value",
    }
    injection_relationship = (
        r"TYPE` with space\u0060) SET s.compromised = true //"
    )
    snapshot = {
        "nodes": [
            {"labels": source_labels, "properties": source_properties},
            {
                "labels": ["Target auxiliary", "CodeSymbol"],
                "properties": target_properties,
            },
            {
                "labels": ["CodeAnchor", "Anchor auxiliary"],
                "properties": anchor_properties,
            },
        ],
        "edges": [
            {
                "type": "CALLS",
                "properties": collision_properties,
                "source": {
                    "labels": source_labels,
                    "properties": source_properties,
                },
                "target": {
                    "labels": ["CodeSymbol", "Target auxiliary"],
                    "properties": target_properties,
                },
            },
            {
                "type": injection_relationship,
                "properties": {},
                "source": {
                    "labels": source_labels,
                    "properties": source_properties,
                },
                "target": {
                    "labels": ["CodeSymbol", "Target auxiliary"],
                    "properties": target_properties,
                },
            },
            {
                "type": "ANCHORED",
                "properties": {},
                "source": {
                    "labels": source_labels,
                    "properties": source_properties,
                },
                "target": {
                    "labels": ["CodeAnchor", "Anchor auxiliary"],
                    "properties": anchor_properties,
                },
            },
        ],
    }
    artifact = _snapshot_artifact(snapshot)
    live_engine.import_snapshot(artifact)
    live_engine.import_snapshot(artifact)

    source_rows = live_engine._run_cypher(
        """
        MATCH (n:CodeSymbol {code_space: $code_space, fqn: $fqn})
        RETURN labels(n) AS labels, properties(n) AS properties
        """,
        code_space=live_engine.code_space,
        fqn="pkg.source",
    )
    assert len(source_rows) == 1
    assert set(source_rows[0]["labels"]) == set(source_labels)
    assert source_rows[0]["properties"] == source_properties

    relationship_rows = live_engine._run_cypher(
        """
        MATCH (s:CodeSymbol {code_space: $code_space, fqn: $source_fqn})-[r]->(t)
        RETURN type(r) AS type, properties(r) AS properties,
               labels(t) AS target_labels, t.fqn AS target_fqn,
               t.repo AS target_repo
        """,
        code_space=live_engine.code_space,
        source_fqn="pkg.source",
    )
    assert len(relationship_rows) == 3
    relationships = {row["type"]: row for row in relationship_rows}
    assert set(relationships) == {"CALLS", injection_relationship, "ANCHORED"}
    assert relationships["CALLS"]["properties"] == collision_properties
    assert relationships[injection_relationship]["properties"] == {}
    assert relationships["ANCHORED"]["properties"] == {}
    assert relationships["CALLS"]["target_fqn"] == "pkg.target"
    assert relationships[injection_relationship]["target_fqn"] == "pkg.target"
    assert relationships["ANCHORED"]["target_fqn"] == "pkg.source"
    assert relationships["ANCHORED"]["target_repo"] == "snapshot-ci"
    assert "CodeAnchor" in relationships["ANCHORED"]["target_labels"]

    all_nodes = live_engine._run_cypher(
        "MATCH (n) RETURN labels(n) AS labels, properties(n) AS properties"
    )
    assert len(all_nodes) == 3
    assert all("compromised" not in row["properties"] for row in all_nodes)


def _preflight_snapshot(code_space: str) -> dict:
    source = {"code_space": code_space, "fqn": "pkg.source"}
    target = {"code_space": code_space, "fqn": "pkg.target"}
    return {
        "nodes": [
            {"labels": ["CodeSymbol"], "properties": source},
            {"labels": ["CodeSymbol"], "properties": target},
        ],
        "edges": [
            {
                "type": "CALLS",
                "properties": {},
                "source": {"labels": ["CodeSymbol"], "properties": source},
                "target": {"labels": ["CodeSymbol"], "properties": target},
            }
        ],
    }


def test_non_string_relationship_type_rejects_before_writes(live_engine):
    snapshot = _preflight_snapshot(live_engine.code_space)
    snapshot["edges"][0]["type"] = ["CALLS"]
    with pytest.raises(ValueError, match=r"edge\[0\]\.type: expected a string"):
        live_engine.import_snapshot(_snapshot_artifact(snapshot))
    assert _node_count(live_engine) == 0


@pytest.mark.parametrize(
    ("location", "message"),
    [
        ("node", r"node\[2\]\.labels: label 1 must not be empty"),
        ("source", r"edge\[1\]\.source\.labels: label 1 must not be empty"),
        ("target", r"edge\[1\]\.target\.labels: label 1 must not be empty"),
        ("relationship", r"edge\[1\]\.type: must not be empty"),
    ],
)
def test_late_empty_identifiers_reject_before_writes(
    live_engine,
    location,
    message,
):
    snapshot = _preflight_snapshot(live_engine.code_space)
    if location == "node":
        snapshot["nodes"].append(
            {
                "labels": ["CodeSymbol", ""],
                "properties": {
                    "code_space": live_engine.code_space,
                    "fqn": "pkg.invalid",
                },
            }
        )
    else:
        invalid = copy.deepcopy(snapshot["edges"][0])
        if location == "relationship":
            invalid["type"] = ""
        else:
            invalid[location]["labels"] = ["CodeSymbol", ""]
        snapshot["edges"].append(invalid)

    with pytest.raises(ValueError, match=message):
        live_engine.import_snapshot(_snapshot_artifact(snapshot))
    assert _node_count(live_engine) == 0


def test_malformed_late_labels_reject_before_writes(live_engine):
    snapshot = _preflight_snapshot(live_engine.code_space)
    snapshot["nodes"].append(
        {
            "labels": ["AuxiliaryOnly"],
            "properties": {
                "code_space": live_engine.code_space,
                "fqn": "pkg.invalid",
            },
        }
    )
    with pytest.raises(ValueError, match="expected exactly one core label"):
        live_engine.import_snapshot(_snapshot_artifact(snapshot))
    assert _node_count(live_engine) == 0


def test_hash_rejection_precedes_database_writes(live_engine):
    snapshot = _preflight_snapshot(live_engine.code_space)
    snapshot["nodes"][0]["labels"] = ["CodeSymbol", ""]
    with pytest.raises(ValueError, match="content_hash mismatch"):
        live_engine.import_snapshot(
            _snapshot_artifact(snapshot, content_hash="not-the-content-hash")
        )
    assert _node_count(live_engine) == 0
