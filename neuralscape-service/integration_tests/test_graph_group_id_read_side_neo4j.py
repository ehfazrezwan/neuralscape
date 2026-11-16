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


def _write(svc, owner, project, content):
    stored, created = svc.store_raw(
        content=content, user_id=owner, category="personal_fact", visibility="private",
        project_id=project, add_to_graph=False, return_created=True,
    )
    assert created and stored
    assert svc.enrich_graph(content=content, user_id=owner, project_id=project, visibility="private",
                            memory_id=stored[0].id) is True


def test_reads_find_the_encoded_write_and_a_colliding_plain_user_sees_nothing(svc, unique_ns, flags):
    from memory.groups import ENCODED_GROUP_PREFIX, _build_group_id

    flags(graph_group_id_injective_enabled=True)
    plain = unique_ns.user_id
    # Legacy naming aliases these two: user 'owner' (no project) and user 'plain' in
    # project 'legacy' both map to 'user--<plain>--project--legacy'. Both ids and the
    # project are plain-alphabet, so the OLD read code queried that shared name.
    owner = f"{plain}--project--legacy"
    project = f"{unique_ns.project_id}.v2"  # '.' needs encoding
    tok_a = f"Quillfeather{uuid.uuid4().hex[:8]}"  # owner, dotted project
    tok_b = f"Quillfeather{uuid.uuid4().hex[:8]}"  # owner, no project (the aliased group)
    group_a = _build_group_id("private", owner, project)
    group_b = _build_group_id("private", owner)
    legacy_b = f"user--{plain}--project--legacy"
    assert group_a.startswith(ENCODED_GROUP_PREFIX) and group_b.startswith(ENCODED_GROUP_PREFIX)
    assert group_b != legacy_b

    def cleanup():
        svc.delete_memories(user_id=owner, include_shared=True)
        _cypher(svc, "MATCH (n) WHERE n.group_id IN [$a, $b, $c] DETACH DELETE n",
                a=group_a, b=group_b, c=legacy_b)

    unique_ns.on_cleanup(cleanup)

    _write(svc, owner, project, f"My dog is named {tok_a} and lives in Reykjavik.")
    _write(svc, owner, None, f"My cat is named {tok_b} and lives in Oslo.")
    assert _cypher(svc, "MATCH (e:Episodic {group_id: $g}) WHERE e.content CONTAINS $t RETURN e.uuid AS u",
                   g=group_a, t=tok_a), "write A did not land in its encoded group"
    assert _cypher(svc, "MATCH (e:Episodic {group_id: $g}) WHERE e.content CONTAINS $t RETURN e.uuid AS u",
                   g=group_b, t=tok_b), "write B did not land in its encoded group"
    assert _cypher(svc, "MATCH (n {group_id: $g}) RETURN count(n) AS c", g=legacy_b)[0]["c"] == 0

    # the owner finds both through the narrowed private branch, episode leg on and off
    for episodes in (False, True):
        for tok, proj in ((tok_a, project), (tok_b, None)):
            hits = svc._search_graph_for_visibility(
                query=tok, user_id=owner, project_id=proj, limit=10, visibility="private",
                include_shared=True, include_episodes=episodes)
            assert tok.lower() in _blob(hits), f"narrowed private search missed {tok} (episodes={episodes})"
    hits = svc._search_graph_for_visibility(
        query=tok_a, user_id=owner, project_id=project, limit=10, visibility=None,
        include_shared=False, include_episodes=True)
    assert tok_a.lower() in _blob(hits)
    assert tok_a.lower() in _blob(svc.search_graph(tok_a, user_id=owner, project_id=project, limit=10))
    assert project in svc.list_projects(owner)

    # the colliding PLAIN user: plain id, plain project 'legacy' -> legacy name == owner's old group
    for episodes in (False, True):
        colliding = svc._search_graph_for_visibility(
            query=tok_b, user_id=plain, project_id="legacy", limit=10, visibility="private",
            include_shared=True, include_episodes=episodes)
        assert tok_b.lower() not in _blob(colliding), "plain user read the encoded owner's fact"
    opt_out = svc._search_graph_for_visibility(
        query=tok_b, user_id=plain, project_id="legacy", limit=10, visibility=None,
        include_shared=False, include_episodes=True)
    assert tok_b.lower() not in _blob(opt_out)
    assert tok_b.lower() not in _blob(svc.search_graph(tok_b, user_id=plain, project_id="legacy", limit=10))
    assert project not in svc.list_projects(plain)


def test_plain_users_legacy_group_is_not_read_by_the_colliding_dash_owner(svc, unique_ns, flags):
    """Discriminates on the pre-change code: the PLAIN user writes into the legacy group
    ``user--<plain>--project--legacy`` (plain ids stay legacy-named with the flag on).  The old
    narrowed branch built that same name from the ``--`` owner (id ``<plain>--project--legacy``,
    no project), so the owner read the plain user's fact.  With the builder the owner queries its
    own encoded group, so it must not see it, while the plain user still does.
    """
    from memory.groups import _build_group_id

    flags(graph_group_id_injective_enabled=True)
    plain = unique_ns.user_id
    owner = f"{plain}--project--legacy"
    token = f"Quillfeather{uuid.uuid4().hex[:8]}"
    legacy = _build_group_id("private", plain, "legacy")
    assert legacy == f"user--{plain}--project--legacy"  # plain tuple keeps its legacy name

    unique_ns.on_cleanup(lambda: (
        svc.delete_memories(user_id=plain, include_shared=True),
        _cypher(svc, "MATCH (n {group_id: $g}) DETACH DELETE n", g=legacy),
    ))
    _write(svc, plain, "legacy", f"My parrot is named {token} and lives in Lisbon.")
    assert _cypher(svc, "MATCH (e:Episodic {group_id: $g}) WHERE e.content CONTAINS $t RETURN e.uuid AS u",
                   g=legacy, t=token)

    mine = svc._search_graph_for_visibility(
        query=token, user_id=plain, project_id="legacy", limit=10, visibility="private",
        include_shared=True, include_episodes=True)
    assert token.lower() in _blob(mine)
    for visibility, include_shared in (("private", True), (None, False)):
        theirs = svc._search_graph_for_visibility(
            query=token, user_id=owner, project_id=None, limit=10, visibility=visibility,
            include_shared=include_shared, include_episodes=True)
        assert token.lower() not in _blob(theirs), "the '--' owner read the plain user's legacy group"
    assert token.lower() not in _blob(svc.search_graph(token, user_id=owner, limit=10))
