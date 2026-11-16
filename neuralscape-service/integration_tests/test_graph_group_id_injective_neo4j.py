"""Real Neo4j + Graphiti, GRAPH_GROUP_ID_INJECTIVE_ENABLED on: write and read share one encoded group.

For a user id containing ``--`` and for a dotted user id, a graph write through
the real ``enrich_graph`` lands in the ``nsg1--`` group the builder names, the
default read path (``search_graph`` over ``_get_group_ids``) finds it there, and
nothing is stored under the legacy (aliasable / Graphiti-invalid) name.

Env: the ``stores`` fixture's variables plus GOOGLE_API_KEY (``llm_key``).
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import uuid

import pytest


@pytest.fixture(scope="module")
def svc(stores, llm_key):
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


@pytest.mark.parametrize("shape", ["double-dash", "dotted"])
def test_write_and_read_use_the_same_encoded_group(svc, unique_ns, flags, shape):
    from memory.groups import ENCODED_GROUP_PREFIX, _build_group_id, decode_group_id

    flags(graph_group_id_injective_enabled=True)
    uid = f"{unique_ns.user_id}--x" if shape == "double-dash" else f"{unique_ns.user_id}.dot"
    token = f"Quillfeather{uuid.uuid4().hex[:8]}"
    content = f"My dog is named {token} and lives in Reykjavik."
    group = _build_group_id("private", uid)
    legacy = f"user--{uid}"
    assert group.startswith(ENCODED_GROUP_PREFIX) and decode_group_id(group).owner == uid

    def cleanup():
        svc.delete_memories(user_id=uid, include_shared=True)
        _cypher(svc, "MATCH (n) WHERE n.group_id IN [$a, $b] DETACH DELETE n", a=group, b=legacy)

    unique_ns.on_cleanup(cleanup)

    stored, created = svc.store_raw(
        content=content, user_id=uid, category="personal_fact", visibility="private",
        add_to_graph=False, return_created=True,
    )
    assert created and stored
    assert svc.enrich_graph(content=content, user_id=uid, project_id=None, visibility="private",
                            memory_id=stored[0].id) is True

    in_group = _cypher(svc, "MATCH (e:Episodic {group_id: $g}) WHERE e.content CONTAINS $t RETURN e.uuid AS u",
                       g=group, t=token)
    assert in_group, "episode was not written to the encoded group"
    assert _cypher(svc, "MATCH (n {group_id: $g}) RETURN count(n) AS c", g=legacy)[0]["c"] == 0

    hits = svc.search_graph(token, user_id=uid, limit=10)
    blob = " ".join(str(x) for key in ("edges", "episodes", "nodes") for x in (hits.get(key) or []))
    assert token.lower() in blob.lower(), "the default read path did not find the write in the same group"
