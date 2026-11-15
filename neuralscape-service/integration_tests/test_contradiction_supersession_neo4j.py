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
A missing or unreachable database is a test failure, not a skip.
"""

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from config import settings

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DB = os.environ.get("NEO4J_DATABASE", "neo4j")

pytestmark = pytest.mark.integration


class _DbDriver:
    """Adapt a neo4j AsyncDriver so ``.session()`` targets the configured db."""

    def __init__(self, driver):
        self._d = driver

    def session(self):
        return self._d.session(database=NEO4J_DB)


@pytest.fixture
def graph_env(monkeypatch):
    from neo4j import AsyncGraphDatabase
    from qdrant_client import QdrantClient, models

    from memory_service import MemoryService

    monkeypatch.setattr(settings, "graph_supersession_enabled", True)
    group = f"it-supersede-{uuid.uuid4().hex[:8]}"
    loop = asyncio.new_event_loop()
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

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
    svc._graphiti.driver = _DbDriver(driver)
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
                    "metadata": {"graph_episode_uuid": episode, "visibility": "private", "scope": "global"},
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
    _seed_edge(cypher, group, "e-full", "Berlin", ["ep-full"], invalid=True, expired=True)
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
