"""Real Neo4j + Graphiti qualification: a queued graph job enriches the CURRENT visibility.

A memory is stored ``shared``; a graph job is queued with that visibility; the
owner narrows the memory to ``private`` BEFORE the job runs. The job must put
the memory's episode into the owner's PRIVATE group and publish nothing into
the shared pool. Previously the job used the visibility frozen at enqueue time
and wrote the narrowed memory into the shared graph group.

Invoked explicitly (outside ``tests/``). A missing or unreachable service is a
test FAILURE, never a skip. Required env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
QDRANT_URL, GOOGLE_API_KEY (embeddings + Graphiti extraction).
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import os
import uuid

import pytest

pytestmark = pytest.mark.integration

REQUIRED_ENV = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "QDRANT_URL", "GOOGLE_API_KEY")


@pytest.fixture(scope="module")
def svc():
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    assert not missing, f"integration environment incomplete, missing: {missing}"
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    assert service._graphiti is not None and service._bridge is not None, "graph store not initialised"
    yield service


def _cypher(svc, query: str, **params):
    async def _run():
        async with svc._graphiti.driver.session() as session:
            result = await session.run(query, **params)
            return await result.data()

    return svc._run_on_bridge(_run(), timeout=60.0)


def _episodes_with(svc, group_id: str, token: str):
    return _cypher(
        svc,
        "MATCH (e:Episodic {group_id: $g}) WHERE e.content CONTAINS $t RETURN e.uuid AS uuid",
        g=group_id, t=token,
    )


def test_job_queued_as_shared_but_narrowed_to_private_lands_in_the_private_group(svc):
    import worker

    uid = f"it-narrow-{uuid.uuid4().hex[:10]}"
    token = f"Quillfeather{uuid.uuid4().hex[:8]}"
    content = f"My dog is named {token} and lives in Reykjavik."
    try:
        stored, created = svc.store_raw(
            content=content, user_id=uid, category="personal_fact", visibility="shared",
            add_to_graph=False, return_created=True,
        )
        assert created and stored
        mid = stored[0].id

        # the owner narrows it while the (shared) graph job is still queued
        svc.patch_memory(mid, uid, {"visibility": "private"})

        res = asyncio.run(worker.process_graph_enrichment(
            {"service": svc}, mid, content, uid, None, "shared",   # the STALE queued visibility
        ))
        assert res["enriched"] is True, res

        assert _episodes_with(svc, f"user--{uid}", token), "episode should live in the owner's private group"
        assert _episodes_with(svc, "shared", token) == [], "narrowed memory leaked into the shared graph group"
        hits = svc.search_graph(token, user_id="someone-else", limit=10)
        blob = " ".join(str(x) for key in ("edges", "episodes", "nodes") for x in (hits.get(key) or []))
        assert token.lower() not in blob.lower()
    finally:
        try:
            svc.delete_memories(user_id=uid, include_shared=True)
        finally:
            _cypher(svc, "MATCH (n) WHERE n.group_id IN [$a, 'shared'] AND coalesce(n.content, n.summary, '') CONTAINS $t "
                         "DETACH DELETE n", a=f"user--{uid}", t=token)
