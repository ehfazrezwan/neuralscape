"""Integration: graph-contradiction supersession against REAL Neo4j + Qdrant.

Seeds RELATES_TO edges directly in Neo4j (no LLM), tombstones through the
production ``supersede_contradicted_rows`` against a real Qdrant client
(in-process local mode, no server needed), and checks the real recall filter.
What it proves is the part unit fakes cannot: the two Cypher statements'
predicates (``invalid_at`` / ``expired_at`` window, ``NOT new_episode IN
episodes``, live-edge counting) behave on a real graph, flag ON.

The LLM-driven half (Graphiti's ``add_episode`` actually invalidating the
edge) needs a Gemini key and is covered by the flag-on VPS integration run.

Lives outside ``tests/`` (like ``test_native_snapshot_neo4j.py``) so the
default host/container suites never see it; invoke explicitly with a live
Neo4j: ``NEO4J_URI=... NEO4J_USER=... NEO4J_PASSWORD=... [NEO4J_DATABASE=neo4j]
uv run pytest integration_tests/test_contradiction_supersession_neo4j.py -v``.
Connection comes from the shared ``neo4j_conn`` fixture in ``conftest.py``: an
unreachable database skips (or fails on the integration gate), it never breaks
collection.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from config import settings

pytestmark = pytest.mark.integration


class _DbDriver:
    """Adapt a neo4j AsyncDriver so ``.session()`` targets the configured db."""

    def __init__(self, driver, database):
        self._d = driver
        self._db = database

    def session(self):
        return self._d.session(database=self._db)


@pytest.fixture
def graph_env(monkeypatch, neo4j_conn):
    from neo4j import AsyncGraphDatabase
    from qdrant_client import QdrantClient, models

    from memory_service import MemoryService

    monkeypatch.setattr(settings, "graph_supersession_enabled", True)
    group = f"it-supersede-{uuid.uuid4().hex[:8]}"
    loop = asyncio.new_event_loop()
    driver = AsyncGraphDatabase.driver(
        neo4j_conn.uri, auth=(neo4j_conn.user, neo4j_conn.password)
    )

    qdrant = QdrantClient(":memory:")
    collection = f"it_{uuid.uuid4().hex[:8]}"
    qdrant.create_collection(
        collection, vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE)
    )
    monkeypatch.setattr(settings, "qdrant_collection", collection)

    svc = MemoryService()
    svc._memory = MagicMock()
    svc._memory.vector_store.client = qdrant
    svc._graphiti = MagicMock()
    svc._graphiti.driver = _DbDriver(driver, neo4j_conn.database)
    svc._bridge = MagicMock()
    svc._run_on_bridge = lambda coro, timeout=None: loop.run_until_complete(coro)

    def cypher(q, **p):
        async def _go():
            async with svc._graphiti.driver.session() as s:
                r = await s.run(q, **p)
                return await r.data()

        return loop.run_until_complete(_go())

    try:
        yield svc, qdrant, collection, group, cypher
    finally:
        cypher("MATCH ()-[r:RELATES_TO {group_id:$g}]->() DELETE r", g=group)
        cypher("MATCH (n:IntegrationSupersede {group_id:$g}) DETACH DELETE n", g=group)
        loop.run_until_complete(driver.close())
        loop.close()


def _seed_edge(cypher, group, uid, fact, episodes, *, invalid=False, expired=False, when=None):
    when = when or datetime.now(timezone.utc)
    cypher(
        "MERGE (a:IntegrationSupersede {group_id:$g, name:'a'}) "
        "MERGE (b:IntegrationSupersede {group_id:$g, name:'b'}) "
        "CREATE (a)-[r:RELATES_TO {uuid:$u, group_id:$g, fact:$f, episodes:$eps}]->(b) "
        "FOREACH (_ IN CASE WHEN $inv THEN [1] ELSE [] END | SET r.invalid_at = datetime($w)) "
        "FOREACH (_ IN CASE WHEN $exp THEN [1] ELSE [] END | SET r.expired_at = datetime($w))",
        g=group, u=uid, f=fact, eps=episodes, inv=invalid, exp=expired, w=when.isoformat(),
    )


def _row(qdrant, collection, pid, user, episode, vec=(1.0, 0.0, 0.0)):
    from qdrant_client import models

    qdrant.upsert(
        collection,
        [
            models.PointStruct(
                id=pid,
                vector=list(vec),
                payload={
                    "data": "fact", "user_id": user, "created_at": "2026-01-01T00:00:00+00:00",
                    "metadata": {"graph_episode_uuid": episode, "graph_episode_kind": "fact", "visibility": "private", "scope": "global"},
                },
            )
        ],
    )


def _meta(qdrant, collection, pid):
    return qdrant.retrieve(collection, [pid])[0].payload["metadata"]


def test_real_cypher_supersedes_fully_invalidated_single_row_only(graph_env):
    from memory.supersession import supersede_contradicted_rows

    svc, qdrant, collection, group, cypher = graph_env
    start = datetime.now(timezone.utc) - timedelta(seconds=1)
    old_full, old_partial, new = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    _row(qdrant, collection, old_full, "u1", "ep-full")
    _row(qdrant, collection, old_partial, "u1", "ep-partial")
    _row(qdrant, collection, new, "u1", "ep-new")
    # Graphiti appends the new episode to an old edge it dedups onto, then
    # invalidates it: the contradicted edge cites [old, new].
    _seed_edge(cypher, group, "e-full", "Berlin", ["ep-full", "ep-new"], invalid=True, expired=True)
    _seed_edge(cypher, group, "e-p1", "Berlin2", ["ep-partial"], invalid=True, expired=True)
    _seed_edge(cypher, group, "e-p2", "Acme", ["ep-partial"])  # still live
    # new episode's own born-expired edge and a delete-cascade edge: not contradictions
    _seed_edge(cypher, group, "e-own", "Paris", ["ep-new"], invalid=True, expired=True)
    _seed_edge(cypher, group, "e-cascade", "x", ["ep-cascade"], expired=True)

    done = supersede_contradicted_rows(
        svc, group_id=group, owner_user_id="u1", write_started_at=start,
        new_episode_uuid="ep-new", new_memory_ids={new}, superseded_by=new,
    )

    assert done == [old_full]
    assert _meta(qdrant, collection, old_full)["dream_tombstoned"] is True
    assert _meta(qdrant, collection, old_full)["superseded_by"] == new
    assert not _meta(qdrant, collection, old_partial).get("dream_tombstoned")
    assert not _meta(qdrant, collection, new).get("dream_tombstoned")


def test_real_cypher_window_excludes_edges_invalidated_before_the_write(graph_env):
    from memory.supersession import supersede_contradicted_rows

    svc, qdrant, collection, group, cypher = graph_env
    pid = str(uuid.uuid4())
    _row(qdrant, collection, pid, "u1", "ep-old")
    _seed_edge(cypher, group, "e-old", "Berlin", ["ep-old"], invalid=True, expired=True,
               when=datetime.now(timezone.utc) - timedelta(days=1))

    done = supersede_contradicted_rows(
        svc, group_id=group, owner_user_id="u1",
        write_started_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        new_episode_uuid="ep-new", new_memory_ids=set(),
    )

    assert done == []
    assert not _meta(qdrant, collection, pid).get("dream_tombstoned")


def test_real_filter_hides_superseded_row_from_recall(graph_env):
    from memory.supersession import supersede_contradicted_rows

    svc, qdrant, collection, group, cypher = graph_env
    old, new = str(uuid.uuid4()), str(uuid.uuid4())
    _row(qdrant, collection, old, "u1", "ep-old")
    _row(qdrant, collection, new, "u1", "ep-new")
    _seed_edge(cypher, group, "e1", "Berlin", ["ep-old"], invalid=True, expired=True)
    supersede_contradicted_rows(
        svc, group_id=group, owner_user_id="u1",
        write_started_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        new_episode_uuid="ep-new", new_memory_ids={new},
    )

    hits = svc._search_personal_pool(
        m=svc._memory, user_id="u1", query_embedding=[1.0, 0.0, 0.0], project_id=None,
        categories=None, scope=None, domain=None, observation_type=None, concepts=None,
        limit=10, query="",
    )
    assert {h.id for h in hits} == {new}


# ── Dedup-hit stale-graph check (default path) on real Neo4j ──


def _stale_case(graph_env, edges, *, episode="ep-x"):
    """Seed ``edges`` for ``episode`` + a Qdrant row stamped with it; return
    (episode_edge_counts, graph_row_is_stale) from the real code paths."""
    from extensions.dreaming.graph_patcher import episode_edge_counts

    svc, qdrant, collection, group, cypher = graph_env
    pid = str(uuid.uuid4())
    _row(qdrant, collection, pid, "u1", episode)
    for uid, invalid in edges:
        _seed_edge(cypher, group, uid, "fact", [episode], invalid=invalid, expired=invalid)
    svc._memory.vector_store.client = qdrant
    counts = svc._run_on_bridge(
        episode_edge_counts(svc._graphiti.driver, group_id=group, episode_uuid=episode)
    )
    # graph_row_is_stale derives group_id from (visibility, user, project):
    # patch the builder so it targets the unique test group.
    from unittest.mock import patch

    with patch("memory.write._build_group_id", return_value=group):
        stale = svc.graph_row_is_stale(pid, "u1", None, "private")
    return counts, stale


def test_stale_check_episode_with_live_edges_is_current(graph_env):
    counts, stale = _stale_case(graph_env, [("s-live-1", False), ("s-dead-1", True)])
    assert counts == {"total": 2, "live": 1}
    assert stale is False


def test_stale_check_episode_with_all_edges_invalidated_is_stale(graph_env):
    counts, stale = _stale_case(graph_env, [("s-dead-1", True), ("s-dead-2", True)])
    assert counts == {"total": 2, "live": 0}
    assert stale is True


def test_stale_check_episode_with_no_edges_is_current(graph_env):
    counts, stale = _stale_case(graph_env, [])
    assert counts == {"total": 0, "live": 0}
    assert stale is False


def test_live_edge_facts_and_dominance_on_the_one_fact_two_edges_shape(graph_env):
    """Gate shape: one episode yields a residence edge (now invalidated) and an
    incidental edge (still live). Real Cypher must report the live fact, and the
    probe must still find the dead edge by its single episode."""
    from extensions.dreaming.graph_patcher import find_contradicted_edges, live_edge_facts_by_episode

    svc, qdrant, collection, group, cypher = graph_env
    start = datetime.now(timezone.utc) - timedelta(seconds=1)
    _seed_edge(cypher, group, "g-dead", "Alice resides in the city of Paris.", ["ep-one"],
               invalid=True, expired=True)
    _seed_edge(cypher, group, "g-live", "Paris is a city situated in France.", ["ep-one"])

    live = svc._run_on_bridge(
        live_edge_facts_by_episode(svc._graphiti.driver, group_id=group, episode_uuids=["ep-one", "ep-none"]))
    assert live == {"ep-one": ["Paris is a city situated in France."], "ep-none": []}
    dead = svc._run_on_bridge(
        find_contradicted_edges(svc._graphiti.driver, group_id=group, write_started_at=start,
                                new_episode_uuid="ep-new"))
    assert [(e["uuid"], e["episodes"]) for e in dead] == [("g-dead", ["ep-one"])]
