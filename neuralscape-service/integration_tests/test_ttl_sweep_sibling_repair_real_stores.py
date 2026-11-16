"""Real-store proof: the TTL expiry sweep repairs the siblings of an expired fact.

Real Neo4j + Qdrant + Graphiti with the LLM. A conversation is written through
``extract_and_store`` (ONE episode, N facts). One fact (chosen by its marker) is
given an ``expires_at`` in the past and ``expire_old_memories`` runs, as the
nightly cron does. Then:

* the expired fact's vector row and its episode are gone, none of the
  episode's edges is live, and the deleted text is unreachable through graph
  search;
* every sibling keeps its vector row, lost its stale episode ref, and got a
  re-enrichment job (one per sibling, none for the expired row);
* running those jobs (as the cron's enqueue would) gives each sibling live graph
  facts again in the user's group, and the expired fact is not in any episode.

Uses ``stores``, ``llm_key``, ``flags``, ``unique_ns``. Needs Neo4j, Qdrant and
GOOGLE_API_KEY.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from integration_tests.cascade_markers import VICTIM_MARKER, conversation, select_victims


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns):
    from qdrant_client import QdrantClient

    flags(qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url)
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    assert service._graphiti is not None and service._bridge is not None, "graph store not initialised"
    group = f"user--{unique_ns.user_id}"
    unique_ns.on_cleanup(lambda: _cypher(service, "MATCH (n {group_id: $g}) DETACH DELETE n", g=group))
    return service


def _cypher(svc, query: str, **params):
    async def _run():
        async with svc._graphiti.driver.session() as session:
            result = await session.run(query, **params)
            return await result.data()

    return svc._run_on_bridge(_run(), timeout=60.0)


def _rows(svc, uid):
    return svc._scroll_all_user_memories(uid)


def _episodes(svc, group):
    return _cypher(svc, "MATCH (e:Episodic {group_id: $g}) RETURN e.uuid AS uuid, e.content AS content", g=group)


def test_expiring_one_fact_of_a_conversation_repairs_the_siblings(svc, unique_ns):
    import worker

    uid = unique_ns.user_id
    group = f"user--{uid}"
    stored = svc.extract_and_store(conversation(), user_id=uid)
    assert len(stored) >= 2, f"extraction produced {len(stored)} fact(s); need >= 2 siblings"
    rows = _rows(svc, uid)
    ep_uuids = {(r["payload"].get("metadata") or {}).get("graph_episode_uuid") for r in rows}
    assert None not in ep_uuids and len(ep_uuids) == 1, f"expected ONE shared episode, got {ep_uuids}"
    (episode_uuid,) = ep_uuids

    victim_ids = select_victims([(r["id"], r["payload"].get("data", "")) for r in rows])
    assert len(victim_ids) == 1, f"expected exactly one fact mentioning {VICTIM_MARKER!r}"
    victim = next(r for r in rows if r["id"] == victim_ids[0])
    victim_text = victim["payload"].get("data", "")
    survivors = [r["id"] for r in rows if r["id"] != victim["id"]]

    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    svc._memory.vector_store.client.set_payload(
        collection_name=unique_ns.qdrant_collection, payload={"expires_at": past},
        points=[victim["id"]], key="metadata",
    )

    out = svc.expire_old_memories()

    assert out["deleted_count"] == 1 and out["per_user"] == {uid: 1}, out
    assert out["graph_cascade_unresolved_ids"] == [], out
    live = {r["id"] for r in _rows(svc, uid)}
    assert victim["id"] not in live and set(survivors) <= live
    for r in _rows(svc, uid):
        assert not (r["payload"].get("metadata") or {}).get("graph_episode_uuid"), "sibling kept a stale episode ref"
    assert not any(e["uuid"] == episode_uuid for e in _episodes(svc, group))
    still_live = _cypher(
        svc,
        "MATCH ()-[r:RELATES_TO {group_id: $g}]->() WHERE r.expired_at IS NULL "
        "AND $ep IN coalesce(r.episodes, []) RETURN r.uuid AS uuid",
        g=group, ep=episode_uuid,
    )
    assert still_live == []
    assert svc.search_episodes_fulltext(VICTIM_MARKER, user_id=uid) == []

    jobs = out["graph_jobs"]
    assert sorted(j["memory_id"] for j in jobs) == sorted(survivors), "one re-enrichment job per sibling, none for the expired row"
    assert out["graph_sibling_unrepaired_ids"] == [] and out["graph_sibling_repair_failed"] is False

    for job in jobs:
        res = asyncio.run(worker.process_graph_enrichment(
            {"service": svc}, job["memory_id"], job["content"], job["user_id"],
            job.get("project_id"), job.get("visibility"), job.get("source_ref"),
        ))
        assert res["enriched"] is True, res
    for sid in survivors:
        row = next(r for r in _rows(svc, uid) if r["id"] == sid)
        new_ep = (row["payload"].get("metadata") or {}).get("graph_episode_uuid")
        assert new_ep, f"sibling {sid} did not regain an episode ref"
        found = _cypher(svc, "MATCH (e:Episodic {uuid: $u}) RETURN e.group_id AS g", u=new_ep)
        assert found and found[0]["g"] == group
    assert all(victim_text.lower() not in e["content"].lower() for e in _episodes(svc, group))
