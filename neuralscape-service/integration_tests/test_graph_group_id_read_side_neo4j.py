"""Real Neo4j + Graphiti, GRAPH_GROUP_ID_INJECTIVE_ENABLED on: every read path follows the write.

A fact written through the real ``enrich_graph`` for a user id containing ``--``
into a project whose id contains ``.`` is stored under the encoded group the
builder names.  The visibility-narrowed graph search (``_search_graph_for_visibility``,
what ``search()`` calls), with and without the episode leg, the default
``search_graph`` read set and ``list_projects`` all find it there.  A different
user whose plain id collides with the owner's LEGACY name sees nothing: not the
fact, not the project.

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


def _blob(hits: dict) -> str:
    return " ".join(str(x) for key in ("edges", "episodes", "nodes") for x in (hits.get(key) or [])).lower()


def test_reads_find_the_encoded_write_and_a_colliding_plain_user_sees_nothing(svc, unique_ns, flags):
    from memory.groups import ENCODED_GROUP_PREFIX, _build_group_id

    flags(graph_group_id_injective_enabled=True)
    plain = unique_ns.user_id  # the colliding user: plain id, reads user--<plain>--project--<legacy>
    owner = f"{plain}--project--legacy"  # legacy name of its private group == plain's project group
    project = f"{unique_ns.project_id}.v2"  # '.' needs encoding
    token = f"Quillfeather{uuid.uuid4().hex[:8]}"
    content = f"My dog is named {token} and lives in Reykjavik."
    group = _build_group_id("private", owner, project)
    assert group.startswith(ENCODED_GROUP_PREFIX)
    legacy = f"user--{owner}--project--{project}"

    def cleanup():
        svc.delete_memories(user_id=owner, include_shared=True)
        _cypher(svc, "MATCH (n) WHERE n.group_id IN [$a, $b] DETACH DELETE n", a=group, b=legacy)

    unique_ns.on_cleanup(cleanup)

    stored, created = svc.store_raw(
        content=content, user_id=owner, category="personal_fact", visibility="private",
        project_id=project, add_to_graph=False, return_created=True,
    )
    assert created and stored
    assert svc.enrich_graph(content=content, user_id=owner, project_id=project, visibility="private",
                            memory_id=stored[0].id) is True
    assert _cypher(svc, "MATCH (e:Episodic {group_id: $g}) WHERE e.content CONTAINS $t RETURN e.uuid AS u",
                   g=group, t=token), "the write did not land in the encoded group"

    # narrowed private branch, with and without the episode leg
    for episodes in (False, True):
        hits = svc._search_graph_for_visibility(
            query=token, user_id=owner, project_id=project, limit=10, visibility="private",
            include_shared=True, include_episodes=episodes)
        assert token.lower() in _blob(hits), f"narrowed private search missed the write (episodes={episodes})"
    # opt-out-of-shared branch and the default read set
    hits = svc._search_graph_for_visibility(
        query=token, user_id=owner, project_id=project, limit=10, visibility=None,
        include_shared=False, include_episodes=True)
    assert token.lower() in _blob(hits)
    assert token.lower() in _blob(svc.search_graph(token, user_id=owner, project_id=project, limit=10))

    # list_projects finds the encoded project
    assert project in svc.list_projects(owner)

    # the colliding plain user: same legacy group name, no access
    colliding = svc._search_graph_for_visibility(
        query=token, user_id=plain, project_id="legacy--project--" + project, limit=10, visibility="private",
        include_shared=True, include_episodes=True)
    assert token.lower() not in _blob(colliding), "plain user read the encoded user's fact"
    assert token.lower() not in _blob(svc.search_graph(token, user_id=plain, project_id=f"legacy--project--{project}", limit=10))
    assert project not in svc.list_projects(plain)
