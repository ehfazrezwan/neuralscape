"""``UPDATE_COMMUNITIES`` on, against real Neo4j, Qdrant and Gemini.

Real graph path: ``MemoryService.enrich_graph`` -> ``MemoryGraph.add`` ->
``Graphiti.add_episode(update_communities=True)`` -> the community build, read
back through the service's ``get_graph_communities`` (what
``GET /v1/graph/communities`` and the ``get_graph_communities`` MCP tool call).

Before the fix two upstream defects kept the endpoint empty with the flag on:
``add_episode`` crashed after storing the episode unless exactly two entities
resolved, and nothing created the first community of a group.

Cost: three short episodes with the flag on, one with it off (each is a handful
of Gemini extraction calls) plus the community summaries.  Env: the ``stores``
variables and ``GOOGLE_API_KEY`` (``llm_key``).
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import pytest

EPISODES = (
    "Alice Marlowe is a staff engineer at Quillfeather Labs in Lisbon.",
    "Quillfeather Labs builds zeppelin navigation software, and Alice Marlowe leads its navigation team.",
    "Bram Okafor reports to Alice Marlowe on the navigation team at Quillfeather Labs.",
)


def _service(stores, unique_ns, flags, *, communities: bool):
    from qdrant_client import QdrantClient

    flags(
        update_communities=communities,
        llm_gateway_enabled=False,  # AI Studio path, no gateway side effects
        llm_gateway_graphiti_enabled=False,
        qdrant_url=stores.qdrant_url,
        qdrant_collection=unique_ns.qdrant_collection,
        neo4j_uri=stores.neo4j_uri,
        neo4j_user=stores.neo4j_user,
        neo4j_password=stores.neo4j_password,
        neo4j_database=stores.neo4j_database,
    )

    from memory_service import MemoryService

    svc = MemoryService()
    svc._get_memory()
    assert svc._graphiti is not None and svc._bridge is not None, "graph store not initialised"
    assert svc._memory.graph._update_communities is communities

    def drop_collections():
        client = QdrantClient(url=stores.qdrant_url)
        for name in (unique_ns.qdrant_collection, f"{unique_ns.qdrant_collection}_entities"):
            try:
                client.delete_collection(name)
            except Exception:  # noqa: BLE001 - already gone
                pass

    def stop_bridge():
        bridge = svc._bridge
        if bridge is not None:
            bridge._loop.call_soon_threadsafe(bridge._loop.stop)

    unique_ns.on_cleanup(stop_bridge)
    unique_ns.on_cleanup(drop_collections)
    return svc


def _cypher(svc, query: str, **params):
    async def _run():
        async with svc._graphiti.driver.session() as session:
            result = await session.run(query, **params)
            return await result.data()

    return svc._run_on_bridge(_run(), timeout=60.0)


def _write(svc, uid: str, content: str) -> None:
    stored, created = svc.store_raw(
        content=content, user_id=uid, category="personal_fact", visibility="private",
        add_to_graph=False, return_created=True,
    )
    assert created and stored
    # enrich_graph swallows errors and reports them as False: a crash inside add_episode shows up here.
    assert svc.enrich_graph(
        content=content, user_id=uid, project_id=None, visibility="private", memory_id=stored[0].id
    ) is True, f"graph write failed for: {content!r}"


def _register_cleanup(svc, unique_ns, uid: str, group: str) -> None:
    def cleanup():
        svc.delete_memories(user_id=uid, include_shared=True)
        _cypher(svc, "MATCH (n) WHERE n.group_id = $g DETACH DELETE n", g=group)

    unique_ns.on_cleanup(cleanup)


def test_flag_on_builds_and_serves_communities(stores, llm_key, unique_ns, flags):
    from memory.groups import _build_group_id

    svc = _service(stores, unique_ns, flags, communities=True)
    uid = unique_ns.user_id
    group = _build_group_id("private", uid)
    _register_cleanup(svc, unique_ns, uid, group)

    assert svc.get_graph_communities(user_id=uid) == [], "the group starts with no communities"

    sentinel = f"{unique_ns.group_prefix}-sentinel"
    other_group = f"{unique_ns.group_prefix}_other"
    _cypher(
        svc,
        "CREATE (:Community {uuid: $u, name: 'sentinel', group_id: $g, summary: 'untouched', created_at: datetime()})",
        u=sentinel, g=other_group,
    )
    unique_ns.on_cleanup(lambda: _cypher(svc, "MATCH (c:Community {uuid: $u}) DETACH DELETE c", u=sentinel))

    for content in EPISODES:
        _write(svc, uid, content)

    communities = svc.get_graph_communities(user_id=uid)
    assert communities, "UPDATE_COMMUNITIES is on and three related episodes were written, yet no community exists"
    for community in communities:
        assert community["group_id"] == group
        assert community["name"].strip(), "a community needs a generated name"
        assert community["summary"].strip(), "a community needs a generated summary"

    members = _cypher(
        svc,
        "MATCH (c:Community {group_id: $g})-[:HAS_MEMBER]->(e:Entity) RETURN count(DISTINCT e) AS n",
        g=group,
    )[0]["n"]
    entities = _cypher(svc, "MATCH (e:Entity {group_id: $g}) RETURN count(e) AS n", g=group)[0]["n"]
    assert entities >= 2, "the episodes extracted no entities; the model, not the flag, is at fault"
    assert members >= 2, f"communities exist but hold {members} of {entities} entities"

    # The build is scoped to the written group: another group's community survives it
    # (``Graphiti.build_communities`` would have deleted every community in the database).
    survivors = _cypher(svc, "MATCH (c:Community {uuid: $u}) RETURN count(c) AS n", u=sentinel)[0]["n"]
    assert survivors == 1, "building communities for one group deleted another group's community"


def test_flag_off_leaves_the_endpoint_empty(stores, llm_key, unique_ns, flags):
    """The contrast: same write, flag off, nothing builds a community."""

    from memory.groups import _build_group_id

    svc = _service(stores, unique_ns, flags, communities=False)
    uid = unique_ns.user_id
    group = _build_group_id("private", uid)
    _register_cleanup(svc, unique_ns, uid, group)

    _write(svc, uid, EPISODES[0])

    assert _cypher(svc, "MATCH (e:Entity {group_id: $g}) RETURN count(e) AS n", g=group)[0]["n"] >= 1
    assert svc.get_graph_communities(user_id=uid) == []
