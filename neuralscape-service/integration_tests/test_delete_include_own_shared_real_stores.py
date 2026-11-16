"""Real-store proof of the opt-in ``include_shared`` bulk delete (#479).

Real Qdrant + real Neo4j/Graphiti (with the LLM for graph enrichment). One caller
owns a private row, a shared row and a ``standard`` row (the caller is a listed
dictator, the only way a standard row can be written); a second user owns a
shared row in the same project pool. Each row is stored through the real
``store_raw`` write path with inline graph enrichment, so each carries a real
Graphiti episode.

``delete_memories(include_shared=True)`` must remove exactly the caller's private
and own shared rows (vector rows AND their graph episodes) and leave the caller's
standard row and the other user's shared row, with their episodes, untouched.
The default (``include_shared=False``) keeps the caller's shared row.

Uses ``stores``, ``llm_key``, ``flags``, ``unique_ns``: every row lives in a
throwaway Qdrant collection and unique graph groups, cleaned up afterwards.
Needs Neo4j, Qdrant and GOOGLE_API_KEY (embeddings + graph extraction).
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import uuid

import pytest


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns):
    from qdrant_client import QdrantClient

    flags(
        qdrant_collection=unique_ns.qdrant_collection,
        qdrant_url=stores.qdrant_url,
        standards_enabled=True,
        dictator_user_ids=unique_ns.user_id,
    )
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    assert service._graphiti is not None and service._bridge is not None, "graph store not initialised"
    return service


def _cypher(svc, query: str, **params):
    async def _run():
        async with svc._graphiti.driver.session() as session:
            result = await session.run(query, **params)
            return await result.data()

    return svc._run_on_bridge(_run(), timeout=60.0)


def _row_meta(svc, row_id) -> dict:
    """The stored row's payload metadata (empty dict when the row is missing)."""
    for pt in svc._memory.vector_store.client.retrieve(
        collection_name=_collection(), ids=[row_id], with_payload=True
    ):
        meta = (pt.payload or {}).get("metadata") or {}
        if isinstance(meta.get("metadata"), dict):
            meta = meta["metadata"]
        return meta
    return {}


def _episode_uuid(svc, row_id):
    return _row_meta(svc, row_id).get("graph_episode_uuid")


def _stored_group(svc, row_id) -> str:
    """The graph group a stored row's episode really lives in.

    Computed from the row's own stored scope/project/visibility with the product's
    group builder, never assumed from what the test asked for.
    """
    from memory.groups import _build_group_id

    meta = _row_meta(svc, row_id)
    return _build_group_id(
        meta.get("visibility"), meta.get("owner_user_id"), meta.get("project_id"), meta.get("workspace")
    )


# The statements that remove ONE episode's own footprint. Every statement is
# constrained by the episode uuid ($ep); none is ever constrained by group alone,
# because the standard group is org-wide and may hold other runs' facts.
_FOOTPRINT_STATEMENTS = (
    # Edges only this episode produced.
    "MATCH ()-[r:RELATES_TO]->() "
    "WHERE $ep IN coalesce(r.episodes, []) AND size(r.episodes) = 1 DELETE r",
    # Graphiti may have merged this fact onto an edge from another episode: keep it,
    # drop only this episode from its provenance list.
    "MATCH ()-[r:RELATES_TO]->() "
    "WHERE $ep IN coalesce(r.episodes, []) AND size(r.episodes) > 1 "
    "SET r.episodes = [x IN r.episodes WHERE x <> $ep]",
    # Entities this episode mentions that no other episode mentions and no edge uses.
    "MATCH (e:Episodic {uuid: $ep})-[:MENTIONS]->(n:Entity) "
    "WHERE NOT EXISTS { MATCH (o:Episodic)-[:MENTIONS]->(n) WHERE o.uuid <> $ep } "
    "AND NOT (n)-[:RELATES_TO]-() DETACH DELETE n",
    # Finally the episode itself (the MENTIONS edges above needed it to exist).
    "MATCH (e:Episodic {uuid: $ep}) DETACH DELETE e",
)


def _delete_episode_footprint(svc, episode_uuid):
    """Remove only the edges, entities and episode node this one episode produced."""
    for statement in _FOOTPRINT_STATEMENTS:
        _cypher(svc, statement, ep=episode_uuid)


def _collection():
    from config import settings

    return settings.qdrant_collection


def _episode_exists(svc, episode_uuid) -> bool:
    return bool(_cypher(svc, "MATCH (e:Episodic {uuid: $u}) RETURN e.uuid AS uuid", u=episode_uuid))


def _live_edge_uuids(svc, group_id, episode_uuid):
    """Live RELATES_TO edges in ``group_id`` that the given episode produced."""
    rows = _cypher(
        svc,
        "MATCH ()-[r:RELATES_TO {group_id: $g}]->() "
        "WHERE r.expired_at IS NULL AND $ep IN coalesce(r.episodes, []) RETURN r.uuid AS uuid",
        g=group_id, ep=episode_uuid,
    )
    return {r["uuid"] for r in rows}


def _live_ids(svc, uid):
    return {r["id"] for r in svc._scroll_all_user_memories(uid)}


def _seed(svc, unique_ns):
    me, other, project = unique_ns.user_id, unique_ns.user_id + "-other", unique_ns.project_id
    tag = uuid.uuid4().hex[:6]

    def store(uid, text, visibility, scope="project"):
        out = svc.store_raw(
            content=f"{text} zq{tag}", user_id=uid, category="convention", scope=scope,
            project_id=project if scope == "project" else None, visibility=visibility,
            add_to_graph=True,
        )
        return out[0].id

    ids = {
        "private": store(me, "The Quillfeather billing service deploys on Fridays.", "private"),
        "shared": store(me, "The Quillfeather team standup is at nine every weekday.", "shared"),
        # The product forces every standard row to global scope with no project, so
        # this row's episode lands in the org-wide ``standard`` group (not a group
        # named after the test). The test therefore touches only what THIS episode
        # produced there (see _delete_episode_footprint), never the whole group.
        "standard": store(me, "All Quillfeather services must expose a health endpoint.", "standard",
                          scope="global"),
        "others_shared": store(other, "The Quillfeather release notes live in the wiki.", "shared"),
    }
    episodes = {k: _episode_uuid(svc, v) for k, v in ids.items()}
    missing = [k for k, v in episodes.items() if not v]
    assert not missing, f"graph enrichment produced no episode for {missing}; the seed is not real"
    for key, ep in episodes.items():
        assert _episode_exists(svc, ep), f"seed episode for {key} is not in Neo4j"

    # Read the standard row back instead of assuming where it landed. If the product
    # ever stops forcing standards global, fail here, loudly, not in a wrong group.
    std_meta = _row_meta(svc, ids["standard"])
    assert std_meta.get("scope") == "global", std_meta
    assert not std_meta.get("project_id"), std_meta
    assert std_meta.get("visibility") == "standard", std_meta
    groups = {
        "standard": _stored_group(svc, ids["standard"]),
        "others_shared": _stored_group(svc, ids["others_shared"]),
    }
    assert groups["standard"] == "standard", groups
    assert groups["others_shared"] == f"shared--project--{project}", groups

    def cleanup():
        # Standard group is org-wide: remove only this episode's own footprint, and do
        # it first (it needs the episode's MENTIONS edges).
        _delete_episode_footprint(svc, episodes["standard"])
        _cypher(
            svc, "MATCH (n) WHERE n.uuid IN $u DETACH DELETE n",
            u=[v for k, v in episodes.items() if k != "standard"],
        )
        # Groups below are unique to this test (user/project names come from unique_ns).
        for g in (
            f"user--{me}", f"user--{me}--project--{project}",
            f"user--{other}", f"user--{other}--project--{project}",
            f"shared--project--{project}",
        ):
            _cypher(svc, "MATCH (n {group_id: $g}) DETACH DELETE n", g=g)
            _cypher(svc, "MATCH ()-[r:RELATES_TO {group_id: $g}]->() DELETE r", g=g)

    unique_ns.on_cleanup(cleanup)
    return me, other, ids, episodes, groups


def test_include_shared_true_deletes_exactly_own_private_and_shared_rows(svc, unique_ns):
    me, other, ids, episodes, groups = _seed(svc, unique_ns)
    shared_group, standard_group = groups["others_shared"], groups["standard"]
    # Non-vacuous: the edges of the other user's shared episode and of the standard
    # episode are live before the delete and must be byte-for-byte the same after.
    shared_edges_before = _live_edge_uuids(svc, shared_group, episodes["others_shared"])
    standard_edges_before = _live_edge_uuids(svc, standard_group, episodes["standard"])
    assert shared_edges_before, "seed produced no live edges for the other user's shared episode"
    assert standard_edges_before, "seed produced no live edges for the standard episode in its real group"

    out = svc.delete_memories(user_id=me, include_shared=True)

    assert out["failed_ids"] == [] and out["truncated"] is False, out
    assert out["graph_cascade_unresolved_ids"] == [], out
    # own private + own shared: vector rows gone, graph episodes gone
    assert ids["private"] not in _live_ids(svc, me) and ids["shared"] not in _live_ids(svc, me)
    assert not _episode_exists(svc, episodes["private"])
    assert not _episode_exists(svc, episodes["shared"])
    # own standard row and the other user's shared row survive, episodes included
    assert ids["standard"] in _live_ids(svc, me)
    assert ids["others_shared"] in _live_ids(svc, other)
    assert _episode_exists(svc, episodes["standard"])
    assert _episode_exists(svc, episodes["others_shared"])
    # The shared and standard groups are never bulk-expired: the other user's facts
    # and the caller's standard facts (this episode's edges) keep expired_at IS NULL.
    assert _live_edge_uuids(svc, shared_group, episodes["others_shared"]) == shared_edges_before
    assert _live_edge_uuids(svc, standard_group, episodes["standard"]) == standard_edges_before


def test_include_shared_false_keeps_the_callers_shared_row(svc, unique_ns):
    me, other, ids, episodes, _groups = _seed(svc, unique_ns)

    out = svc.delete_memories(user_id=me)

    assert out["failed_ids"] == [], out
    assert ids["private"] not in _live_ids(svc, me)
    assert not _episode_exists(svc, episodes["private"])
    assert ids["shared"] in _live_ids(svc, me) and _episode_exists(svc, episodes["shared"])
    assert ids["standard"] in _live_ids(svc, me)
    assert ids["others_shared"] in _live_ids(svc, other)
