"""Integration: ``scripts/migrate-group-ids.cypher`` against REAL Neo4j.

Seeds Entity / Episodic / Community / Saga nodes and RELATES_TO / MENTIONS /
HAS_MEMBER / HAS_EPISODE / NEXT_EPISODE relationships under five group ids
(``project:p``, ``project--p``, ``shared--project--p``, ``user--u--project--p``,
``global``), runs the migration file statement by statement through the driver
with ``target_prefix`` bound, and asserts every row's final id and every
reported count. A second run must migrate nothing. Bad prefixes must abort
before any write.

The migration rewrites EVERY legacy-id row in the target database, not only
the rows seeded here, so run it against a disposable database (the gate's).
Seeded rows carry a per-run ``run`` property and are removed before and after.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import pytest
from conftest import unavailable
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

_HERE = Path(__file__).resolve().parent
# Checkout: repo-level scripts/. Packaged test image: copied by the Dockerfile test stage.
CYPHER = next(
    (
        p
        for p in (
            _HERE.parents[1] / "scripts" / "migrate-group-ids.cypher",
            _HERE / "repo_scripts" / "migrate-group-ids.cypher",
        )
        if p.is_file()
    ),
    _HERE.parents[1] / "scripts" / "migrate-group-ids.cypher",
)

NODE_LABELS = ["Entity", "Episodic", "Community", "Saga"]
# (relationship type, start label, end label)
REL_TYPES = [
    ("RELATES_TO", "Entity", "Entity"),
    ("MENTIONS", "Episodic", "Entity"),
    ("HAS_MEMBER", "Community", "Entity"),
    ("HAS_EPISODE", "Saga", "Episodic"),
    ("NEXT_EPISODE", "Episodic", "Episodic"),
]
LEGACY = re.compile(r"^project(:|--)")


@pytest.fixture(scope="session")
def cypher_file() -> Path:
    """The migration file under test must exist.

    The packaged test image copies it in explicitly (Dockerfile, test stage).
    Under NS_INTEGRATION_REQUIRED (gate / Actions) a missing file is a FAILURE,
    never a skip, so a gate can not pass without exercising the Cypher.
    """
    if not CYPHER.is_file():
        unavailable(f"the migration under test is missing: {CYPHER}")
    return CYPHER


def _gids(pid: str, uid: str) -> list[str]:
    return [
        f"project:{pid}",
        f"project--{pid}",
        f"shared--project--{pid}",
        f"user--{uid}--project--{pid}",
        "global",
    ]


def _expected(gid: str, pid: str, target: str) -> str:
    if gid in (f"project:{pid}", f"project--{pid}"):
        return target + pid
    return gid


def _statements() -> list[str]:
    body = "\n".join(
        line for line in CYPHER.read_text().splitlines() if not line.lstrip().startswith("//")
    )
    return [s.strip() for s in body.split(";") if s.strip()]


@pytest.fixture
def graph(cypher_file, neo4j_conn):
    driver = GraphDatabase.driver(neo4j_conn.uri, auth=(neo4j_conn.user, neo4j_conn.password))
    run = f"it-mig-{uuid.uuid4().hex[:10]}"

    def cy(query: str, **params):
        with driver.session(database=neo4j_conn.database) as s:
            return s.run(query, **params).data()

    def clean():
        cy("MATCH (n {run: $run}) DETACH DELETE n", run=run)

    clean()
    try:
        yield cy, run
    finally:
        clean()
        driver.close()


def _seed(cy, run: str, pid: str, uid: str):
    gids = _gids(pid, uid)
    anchor = f"{run}-anchor"  # not a legacy id: never touched by the migration
    for label in NODE_LABELS:
        cy(
            f"UNWIND $gids AS g CREATE (n:{label} {{run: $run, uuid: $run + '-{label}-' + g, group_id: g}})",
            gids=gids, run=run,
        )
    for rtype, la, lb in REL_TYPES:
        cy(
            f"UNWIND $gids AS g "
            f"CREATE (a:{la} {{run: $run, uuid: $run + '-a-{rtype}-' + g, group_id: $anchor}})"
            f"-[:{rtype} {{run: $run, uuid: $run + '-r-{rtype}-' + g, group_id: g}}]->"
            f"(b:{lb} {{run: $run, uuid: $run + '-b-{rtype}-' + g, group_id: $anchor}})",
            gids=gids, run=run, anchor=anchor,
        )


def _legacy_counts(cy) -> dict[str, int]:
    pred = "{v}.group_id STARTS WITH 'project:' OR {v}.group_id STARTS WITH 'project--'"
    counts = {
        label: cy(f"MATCH (n:{label}) WHERE {pred.format(v='n')} RETURN count(n) AS c")[0]["c"]
        for label in NODE_LABELS
    }
    counts["relationships"] = cy(
        f"MATCH ()-[r]->() WHERE {pred.format(v='r')} RETURN count(r) AS c"
    )[0]["c"]
    return counts


def _foreign_legacy_count(cy, run: str) -> int:
    """Legacy-id nodes and relationships NOT seeded by this run (the migration would rewrite them)."""
    pred = "{v}.group_id STARTS WITH 'project:' OR {v}.group_id STARTS WITH 'project--'"
    nodes = cy(
        f"MATCH (n) WHERE ({pred.format(v='n')}) AND (n.run IS NULL OR n.run <> $run) "
        "RETURN count(n) AS c", run=run,
    )[0]["c"]
    rels = cy(
        f"MATCH ()-[r]->() WHERE ({pred.format(v='r')}) AND (r.run IS NULL OR r.run <> $run) "
        "RETURN count(r) AS c", run=run,
    )[0]["c"]
    return nodes + rels


def _assert_no_foreign_legacy_rows(cy, run: str) -> None:
    n = _foreign_legacy_count(cy, run)
    if n:
        pytest.fail(
            f"{n} legacy-id node(s)/relationship(s) not seeded by this test exist in the target "
            "database; the migration would rewrite them. Point NEO4J_* at a disposable database.",
            pytrace=False,
        )


def test_target_database_holds_no_foreign_legacy_rows(graph):
    cy, run = graph
    assert _foreign_legacy_count(cy, run) == 0


def _run_file(cy, target_prefix: str) -> dict[str, int]:
    """Execute the file statement by statement; return {label: migrated} plus the verify counts."""
    migrated: dict[str, int] = {}
    for stmt in _statements():
        rows = cy(stmt, target_prefix=target_prefix)
        assert len(rows) == 1
        row = rows[0]
        if "label" in row:
            migrated[row["label"]] = row["migrated"]
        elif "remaining_nodes" in row:
            migrated["remaining_nodes"] = row["remaining_nodes"]
        elif "remaining_edges" in row:
            migrated["remaining_edges"] = row["remaining_edges"]
    return migrated


@pytest.mark.parametrize("target", ["user--u1--project--", "shared--project--"])
def test_cypher_rewrites_every_label_and_relationship_exactly(graph, target):
    cy, run = graph
    pid, uid = f"p{uuid.uuid4().hex[:6]}", "u1"
    _assert_no_foreign_legacy_rows(cy, run)
    _seed(cy, run, pid, uid)
    before = _legacy_counts(cy)
    # at least the seeded legacy rows: 2 per label, 2 per relationship type
    assert all(before[label] >= 2 for label in NODE_LABELS)
    assert before["relationships"] >= 2 * len(REL_TYPES)

    result = _run_file(cy, target)

    # every label pass reports exactly the legacy rows that existed before
    for label in NODE_LABELS:
        assert result[label] == before[label], label
    assert result["relationships"] == before["relationships"]
    assert result["remaining_nodes"] == 0
    assert result["remaining_edges"] == 0

    # every seeded row ends up exactly where it should
    for label in NODE_LABELS:
        for gid in _gids(pid, uid):
            got = cy(
                f"MATCH (n:{label} {{uuid: $u}}) RETURN n.group_id AS g",
                u=f"{run}-{label}-{gid}",
            )
            assert got == [{"g": _expected(gid, pid, target)}], (label, gid)
    for rtype, _, _ in REL_TYPES:
        for gid in _gids(pid, uid):
            got = cy(
                f"MATCH ()-[r:{rtype} {{uuid: $u}}]->() RETURN r.group_id AS g",
                u=f"{run}-r-{rtype}-{gid}",
            )
            assert got == [{"g": _expected(gid, pid, target)}], (rtype, gid)
    # anchors (non-legacy ids) are untouched
    assert {r["g"] for r in cy("MATCH (n {run: $run, group_id: $a}) RETURN n.group_id AS g", run=run, a=f"{run}-anchor")} == {f"{run}-anchor"}
    assert cy("MATCH (n {run: $run}) WHERE n.group_id STARTS WITH 'project:' OR n.group_id STARTS WITH 'project--' RETURN count(n) AS c", run=run) == [{"c": 0}]

    # idempotent: a second run migrates nothing and changes nothing
    second = _run_file(cy, target)
    for label in NODE_LABELS:
        assert second[label] == 0, label
    assert second["relationships"] == 0
    assert second["remaining_nodes"] == 0 and second["remaining_edges"] == 0
    got = cy("MATCH (n:Entity {uuid: $u}) RETURN n.group_id AS g", u=f"{run}-Entity-project:{pid}")
    assert got == [{"g": target + pid}]


@pytest.mark.parametrize(
    "bad", ["", "shared-project--", "project--", "user----project--", "user--u--project", "user--x--ws--project--", "shared--project", "x"]
)
def test_cypher_rejects_bad_target_prefix_before_writing(graph, bad):
    cy, run = graph
    pid, uid = f"p{uuid.uuid4().hex[:6]}", "u1"
    _seed(cy, run, pid, uid)
    guard = _statements()[0]
    with pytest.raises(ClientError):
        cy(guard, target_prefix=bad)
    # the guard is the first statement: nothing was rewritten
    assert cy(
        "MATCH (n:Entity {uuid: $u}) RETURN n.group_id AS g", u=f"{run}-Entity-project:{pid}"
    ) == [{"g": f"project:{pid}"}]


def test_cypher_guard_accepts_valid_prefixes(graph):
    cy, _ = graph
    guard = _statements()[0]
    for ok in ("shared--project--", "user--u1--project--", "user--a-b--project--"):
        assert cy(guard, target_prefix=ok) == [{"target_prefix_valid": 1}]
