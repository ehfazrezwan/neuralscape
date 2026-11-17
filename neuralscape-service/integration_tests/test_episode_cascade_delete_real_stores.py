"""Real-store qualification of the shared-episode delete cascade.

Real Neo4j + real Qdrant + real Graphiti with the LLM. A fresh, unique user
writes a small conversation through ``extract_and_store`` (ONE episode, N
extracted facts). Then:

1. deleting one fact removes its vector row, deletes the episode node from
   Neo4j, leaves no live edge of that episode, and no graph search / episode
   recall returns the deleted text;
2. every sibling gets a re-enrichment job; running ``process_graph_enrichment``
   for them restores graph evidence in the same group;
3. a bulk private delete of the user leaves zero rows and zero live edges in
   the user's group.

Uses the integration fixtures (``stores``, ``llm_key``, ``flags``, ``unique_ns``):
the rows live in a throwaway Qdrant collection dropped afterwards, the user's
graph group is purged by ``unique_ns.on_cleanup``. On the gate
(``NS_INTEGRATION_REQUIRED=1``) missing stores or key fail the test.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import time

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
    service._get_memory()  # real mem0 + Graphiti; raises if a store is unreachable
    assert service._graphiti is not None and service._bridge is not None, "graph store not initialised"
    return service


@pytest.fixture
def user(svc, unique_ns):
    uid = unique_ns.user_id

    def _purge():
        # vectors are dropped with the collection; remove every graph node/edge in the user's group
        _cypher(svc, "MATCH (n {group_id: $g}) DETACH DELETE n", g=f"user--{uid}")
        _cypher(svc, "MATCH ()-[r:RELATES_TO {group_id: $g}]->() DELETE r", g=f"user--{uid}")

    unique_ns.on_cleanup(_purge)
    return uid


def _cypher(svc, query: str, **params):
    async def _run():
        async with svc._graphiti.driver.session() as session:
            result = await session.run(query, **params)
            return await result.data()

    return svc._run_on_bridge(_run(), timeout=60.0)


def _rows(svc, uid):
    return svc._scroll_all_user_memories(uid)


def _group(uid):
    return f"user--{uid}"


def _live_edges(svc, uid):
    return _cypher(
        svc,
        "MATCH ()-[r:RELATES_TO {group_id: $g}]->() WHERE r.expired_at IS NULL RETURN r.uuid AS uuid, r.fact AS fact",
        g=_group(uid),
    )


def _episodes(svc, uid):
    return _cypher(
        svc,
        "MATCH (e:Episodic {group_id: $g}) RETURN e.uuid AS uuid, e.content AS content",
        g=_group(uid),
    )


def _wait(predicate, timeout=30.0, step=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


def test_delete_one_fact_repairs_siblings_then_bulk_wipe(svc, user):
    import worker

    stored = svc.extract_and_store(conversation(), user_id=user)
    assert len(stored) >= 2, f"extraction produced {len(stored)} fact(s); need >= 2 siblings"

    rows = _rows(svc, user)
    ep_uuids = {
        (r["payload"].get("metadata") or {}).get("graph_episode_uuid") for r in rows
    }
    assert None not in ep_uuids and len(ep_uuids) == 1, f"expected ONE shared episode, got {ep_uuids}"
    (episode_uuid,) = ep_uuids
    assert any(e["uuid"] == episode_uuid for e in _episodes(svc, user))

    # --- delete ONE fact (chosen deterministically: the one carrying the seed token) --
    victim_ids = select_victims([(r["id"], r["payload"].get("data", "")) for r in rows])
    victims = [r for r in rows if r["id"] in victim_ids]
    assert len(victims) == 1, (
        f"expected exactly one extracted fact mentioning {VICTIM_MARKER!r}, got "
        f"{[r['payload'].get('data') for r in rows]}"
    )
    victim = victims[0]
    mentioned = [
        r["uuid"] for r in _cypher(
            svc,
            "MATCH (e:Episodic {uuid: $u})-[:MENTIONS]->(n:Entity) RETURN n.uuid AS uuid",
            u=episode_uuid,
        )
    ]
    victim_text = victim["payload"].get("data", "")
    survivors = [r["id"] for r in rows if r["id"] != victim["id"]]
    out = svc.delete_memory(victim["id"], user)
    assert out["graph_cascade"] == "resolved"

    # vector store: row gone, siblings still there with the stale ref cleared
    live_ids = {r["id"] for r in _rows(svc, user)}
    assert victim["id"] not in live_ids and set(survivors) <= live_ids
    for r in _rows(svc, user):
        assert not (r["payload"].get("metadata") or {}).get("graph_episode_uuid")

    # graph: episode node gone, none of its edges live
    assert not any(e["uuid"] == episode_uuid for e in _episodes(svc, user))
    still_live = _cypher(
        svc,
        "MATCH ()-[r:RELATES_TO {group_id: $g}]->() WHERE r.expired_at IS NULL "
        "AND $ep IN coalesce(r.episodes, []) RETURN r.uuid AS uuid",
        g=_group(user), ep=episode_uuid,
    )
    assert still_live == []
    # entity nodes the episode mentioned: removed, or summary cleared (never left
    # carrying text synthesised from the deleted conversation)
    if mentioned:
        left = _cypher(
            svc, "MATCH (n:Entity) WHERE n.uuid IN $u RETURN n.uuid AS uuid, n.summary AS summary",
            u=mentioned,
        )
        assert all(not (n["summary"] or "").strip() for n in left), left
    # nothing a reader can reach contains the deleted fact
    hits = svc.search_graph(victim_text, user_id=user, limit=10)
    blob = " ".join(
        str(x) for key in ("edges", "episodes", "nodes") for x in (hits.get(key) or [])
    ).lower()
    assert VICTIM_MARKER.lower() not in blob, "deleted fact text is still reachable through graph search"
    assert svc.search_episodes_fulltext(VICTIM_MARKER, user_id=user) == []

    # --- siblings get re-enrichment jobs; run them inline ----------------
    jobs = out["graph_jobs"]
    assert sorted(j["memory_id"] for j in jobs) == sorted(survivors)
    assert out["graph_sibling_unrepaired_ids"] == [] and out["graph_sibling_repair_failed"] is False
    for job in jobs:
        res = asyncio.run(worker.process_graph_enrichment(
            {"service": svc}, job["memory_id"], job["content"], job["user_id"],
            job.get("project_id"), job.get("visibility"), job.get("source_ref"),
        ))
        assert res["enriched"] is True, res
    for sid in survivors:
        row = next(r for r in _rows(svc, user) if r["id"] == sid)
        new_ep = (row["payload"].get("metadata") or {}).get("graph_episode_uuid")
        assert new_ep, f"sibling {sid} did not regain an episode ref"
        found = _cypher(
            svc, "MATCH (e:Episodic {uuid: $u}) RETURN e.group_id AS g", u=new_ep
        )
        assert found and found[0]["g"] == _group(user)
    # and the deleted fact is not back in any episode body
    assert all(victim_text.lower() not in e["content"].lower() for e in _episodes(svc, user))

    # --- bulk private delete ---------------------------------------------
    res = svc.delete_memories(user_id=user)
    assert res["graph_cascade_unresolved_ids"] == []
    assert _rows(svc, user) == []
    assert _live_edges(svc, user) == []
    assert _episodes(svc, user) == []
