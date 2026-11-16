"""Real Neo4j + Qdrant + Graphiti, episode recall leg on (its default): ``search()`` returns the verbatim
episode row of a fact even when the scored rows alone fill ``limit``.

Episode rows carry no score, so fusion ranks them below every scored row; a plain top-``limit`` slice then
dropped all of them on any populated project, and the leg never surfaced.  Here the project holds more
vector memories than the limit, and the one fact enriched into the graph must still come back as a
``source: episode`` row, after the scored rows, without the list growing past ``limit``.

Env: the ``stores`` fixture's variables plus GOOGLE_API_KEY (``llm_key``).
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import time
import uuid

import pytest

LIMIT = 8
FILLER = 14  # more vector memories than LIMIT: the scored legs alone fill the list


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


def test_episode_row_survives_a_full_scored_list(svc, unique_ns, flags):
    from memory.groups import _build_group_id

    flags(graph_episode_recall_enabled=True)
    user, project = unique_ns.user_id, unique_ns.project_id
    token = f"Quillfeather{uuid.uuid4().hex[:8]}"
    fact = f"The {token} service keeps its cache layer in the Lisbon region."
    group = _build_group_id("private", user, project)
    unique_ns.on_cleanup(lambda: (
        svc.delete_memories(user_id=user, include_shared=True),
        _cypher(svc, "MATCH (n {group_id: $g}) DETACH DELETE n", g=group),
    ))

    for i in range(FILLER):  # vector rows only: cheap, and each one ranks like the fact
        svc.store_raw(
            content=f"The {token} service number {i} keeps its cache layer in region {i}.",
            user_id=user, category="tech_stack", visibility="private", project_id=project, add_to_graph=False,
        )
    stored, created = svc.store_raw(
        content=fact, user_id=user, category="tech_stack", visibility="private", project_id=project,
        add_to_graph=False, return_created=True,
    )
    assert created and stored
    assert svc.enrich_graph(content=fact, user_id=user, project_id=project, visibility="private",
                            memory_id=stored[0].id) is True
    assert _cypher(svc, "MATCH (e:Episodic {group_id: $g}) WHERE e.content CONTAINS $t RETURN e.uuid AS u",
                   g=group, t=token), "the episode was not written to the graph"

    deadline = time.monotonic() + 30  # the episode fulltext index catches up a moment after the write
    while True:
        results = svc.search(fact, user_id=user, project_id=project, scope="project", visibility="private",
                             include_shared=False, limit=LIMIT)
        episodes = [r for r in results if r.source == "episode" and token in r.memory]
        if episodes or time.monotonic() > deadline:
            break
        time.sleep(2)

    assert episodes, f"no episode row among {len(results)} results {[r.source for r in results]}"
    assert len(results) <= LIMIT
    assert episodes[0].score is None
    assert results[-1].source == "episode", "unscored rows rank after every scored row"
