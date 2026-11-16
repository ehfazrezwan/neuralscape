"""Inventory against REAL Redis, Qdrant and Neo4j holding synthetic legacy data.

Needs MIGRATION_TEST_REDIS_URL, MIGRATION_TEST_QDRANT_URL and
MIGRATION_TEST_NEO4J_URI (+ MIGRATION_TEST_NEO4J_PASSWORD); skipped otherwise
(failed under NS_INTEGRATION_REQUIRED=1).  The sources are read-only: the test proves the
stores are unchanged by the inventory.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

_NEEDED = ("MIGRATION_TEST_REDIS_URL", "MIGRATION_TEST_QDRANT_URL", "MIGRATION_TEST_NEO4J_URI")


@pytest.fixture(autouse=True, scope="module")
def _require_stores():
    """Missing stores fail under NS_INTEGRATION_REQUIRED=1 (the gate) and skip otherwise."""

    missing = [name for name in _NEEDED if not os.environ.get(name)]
    if missing:
        if os.environ.get("NS_INTEGRATION_REQUIRED") == "1":
            pytest.fail(f"NS_INTEGRATION_REQUIRED=1 but {missing} not set")
        pytest.skip("MIGRATION_TEST_* not set (policy migration real-store integration)")


redis = pytest.importorskip("redis")
qdrant_client = pytest.importorskip("qdrant_client")
neo4j = pytest.importorskip("neo4j")

import policy_migration as pm  # noqa: E402
from authority_store_sql import SqliteAuthorityStore  # noqa: E402

T = "tenant-a"
RUN = uuid.uuid4().hex[:8]


@pytest.fixture(scope="module")
def stores():
    from identity_store import EMAIL_HASH, SUB_HASH
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    r = redis.Redis.from_url(os.environ["MIGRATION_TEST_REDIS_URL"], decode_responses=True)
    r.delete(SUB_HASH, EMAIL_HASH)
    r.hset(SUB_HASH, mapping={"sub-alice": "alice", "sub-frank-1": "frank", "sub-frank-2": "frank"})
    r.hset(EMAIL_HASH, mapping={"alice@example.com": "alice", "a@b-c.com": "gina", "a-b@c.com": "gina"})

    q = QdrantClient(url=os.environ["MIGRATION_TEST_QDRANT_URL"])
    collection = f"legacy_{RUN}"
    q.create_collection(collection, vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    payloads = [
        {"user_id": "alice", "metadata": {"owner_user_id": "alice", "visibility": "private"}},
        {"user_id": "alice", "metadata": {"owner_user_id": "alice", "visibility": "private", "project_id": "p1"}},
        {"user_id": "alice", "metadata": {"owner_user_id": "alice", "visibility": "shared", "project_id": "p1"}},
        {"user_id": "bob", "metadata": {"owner_user_id": "bob", "visibility": "private"}},
        {"user_id": "carol", "metadata": {"owner_user_id": "carol", "visibility": "standard"}},
        {"user_id": "frank", "metadata": {"owner_user_id": "frank", "visibility": "private"}},
        {"user_id": "x", "metadata": {"owner_user_id": "x", "visibility": "private"}},
        {"user_id": "x--project--y", "metadata": {"owner_user_id": "x--project--y", "visibility": "private"}},
    ]
    q.upsert(collection, [PointStruct(id=i + 1, vector=[0.1, 0.2, 0.3, 0.4], payload=p) for i, p in enumerate(payloads)])

    driver = neo4j.GraphDatabase.driver(
        os.environ["MIGRATION_TEST_NEO4J_URI"], auth=("neo4j", os.environ.get("MIGRATION_TEST_NEO4J_PASSWORD", ""))
    )
    deadline = time.time() + 90
    while True:
        try:
            driver.verify_connectivity()
            break
        except Exception:
            if time.time() > deadline:
                raise
            time.sleep(2)
    groups = ["user--alice", "user--alice--project--p1", "shared", "shared--project--p1",
              "standard", "user--x--project--y", "user--x"]
    with driver.session() as session:
        session.run("MATCH (n:LegacyProbe) DETACH DELETE n")
        for group in groups:
            session.run("CREATE (:LegacyProbe {group_id: $g})", g=group)
    yield {"redis": r, "qdrant": q, "collection": collection, "driver": driver, "groups": groups}
    with driver.session() as session:
        session.run("MATCH (n:LegacyProbe) DETACH DELETE n")
    driver.close()
    q.delete_collection(collection)
    r.delete(SUB_HASH, EMAIL_HASH)


def test_sources_read_real_legacy_data_without_modifying_it(stores) -> None:
    from identity_store import EMAIL_HASH, SUB_HASH

    r, q = stores["redis"], stores["qdrant"]
    before = (r.hgetall(SUB_HASH), r.hgetall(EMAIL_HASH), q.count(stores["collection"]).count)
    aliases = pm.RedisAliasSource(os.environ["MIGRATION_TEST_REDIS_URL"]).read()
    pools = list(pm.QdrantPoolSource(os.environ["MIGRATION_TEST_QDRANT_URL"], stores["collection"], batch=3).read())
    source = pm.Neo4jGroupSource(os.environ["MIGRATION_TEST_NEO4J_URI"], "neo4j", os.environ.get("MIGRATION_TEST_NEO4J_PASSWORD", ""))
    try:
        groups = source.read()
    finally:
        source.close()
    after = (r.hgetall(SUB_HASH), r.hgetall(EMAIL_HASH), q.count(stores["collection"]).count)
    assert before == after
    assert aliases[0]["sub-alice"] == "alice" and len(pools) == 8        # scroll paged over batches of 3
    assert set(stores["groups"]) <= groups
    assert pm.PoolObservation("alice", "shared", "p1", None) in pools


def test_real_inventory_classifies_quarantines_and_applies(stores, tmp_path) -> None:
    aliases = pm.RedisAliasSource(os.environ["MIGRATION_TEST_REDIS_URL"]).read()
    pools = pm.QdrantPoolSource(os.environ["MIGRATION_TEST_QDRANT_URL"], stores["collection"]).read()
    source = pm.Neo4jGroupSource(os.environ["MIGRATION_TEST_NEO4J_URI"], "neo4j", os.environ.get("MIGRATION_TEST_NEO4J_PASSWORD", ""))
    try:
        groups = {g for g in source.read() if g in stores["groups"]}   # only this test's synthetic groups
    finally:
        source.close()
    inventory = pm.collect_inventory(aliases, pools, groups)
    report = pm.build_report(inventory, extra_user_ids=["carol"])
    assert report.unique == ["alice", "bob", "carol", "x"]
    assert set(report.quarantined) == {"frank", "gina", "x--project--y"}
    assert report.groups["user--x--project--y"].problem == "ambiguous_owner_project"
    store = SqliteAuthorityStore(str(tmp_path / "real.db"))
    try:
        result = pm.apply_plan(store, inventory, T, dictators=("carol",), standards_enabled=True)
        assert result.changed and result.subjects_created == 5            # 4 unique users + legacy key
        assert not pm.apply_plan(store, inventory, T, dictators=("carol",), standards_enabled=True).changed
        policy = store.current_policy(T)
        quarantined_subjects = {pm.resolve_binding(store, "local", u) for u in ("frank", "gina", "x--project--y")}
        assert quarantined_subjects == {None}
        assert {s.subject_id for s in policy.statements} <= set(result.subject_ids.values()) | {pm.resolve_binding(store, "local", pm.LEGACY_KEY_SUBJECT)}
    finally:
        store.close()
