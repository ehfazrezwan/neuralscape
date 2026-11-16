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


def test_episode_row_is_found_on_a_default_recall_across_all_groups(svc, unique_ns, flags):
    """The request shape of the REST route with only ``query``, ``project_id`` and ``limit``: no scope, no
    visibility, shared pool included, so the graph legs search the caller's private, shared and project
    groups together. The fulltext group filter used to require only the LAST group of that list, so an
    episode in the caller's private project group never matched.

    The episode comes from a CONVERSATION write (``extract_and_store``, the entry point the service uses):
    its episode is the transcript while the stored memory is the extractor's distilled fact, so the two are
    not content twins. A raw write would not do: its episode text equals the memory text, and recall's
    twin-drop (``_deduplicate_responses``) then correctly returns that text once, as the memory row."""
    from memory.groups import _build_group_id, _get_group_ids

    flags(graph_episode_recall_enabled=True)
    user, project = unique_ns.user_id, unique_ns.project_id
    token = "Marblequill"  # letters only: the extractor keeps an invented proper noun but drops digit suffixes
    group = _build_group_id("private", user, project)
    unique_ns.on_cleanup(lambda: (
        svc.delete_memories(user_id=user, include_shared=True),
        _cypher(svc, "MATCH (n {group_id: $g}) DETACH DELETE n", g=group),
    ))

    messages = [
        {"role": "user", "content": (
            f"For my notes: the {token} gateway is where we terminate our tunnels, and as far as I can "
            "tell it runs out of the Oslo datacentre these days."
        )},
        {"role": "assistant", "content": "Understood, I will keep that in mind."},
    ]
    stored = svc.extract_and_store(messages, user_id=user, project_id=project)
    assert stored, "the extractor stored no memory from the conversation"
    episode_rows = _cypher(
        svc, "MATCH (e:Episodic {group_id: $g}) WHERE e.content CONTAINS $t RETURN e.content AS c", g=group, t=token
    )
    assert episode_rows, "the episode was not written to the graph"

    # Precondition: no stored memory is a content twin of the episode row recall would build from it. A
    # verbatim extraction would make recall drop the episode row as a twin of the memory row and this test
    # would fail for a reason that has nothing to do with the group filter.
    episode_text = svc.normalize_memory_content(f"[verbatim session excerpt] {episode_rows[0]['c'][:600]}")
    for memory in stored:
        twin = svc.find_duplicate_content(svc.normalize_memory_content(memory.memory), [episode_text])
        assert twin is None, (
            f"precondition: the extractor kept {memory.memory!r} verbatim from the transcript, so the episode "
            "row would be dropped as its twin; use a transcript line the extractor will paraphrase"
        )

    groups = _get_group_ids(user, project)
    assert group in groups and groups[-1] != group, "precondition: the episode's group is not the last one searched"

    deadline = time.monotonic() + 30
    while True:
        results = svc.search(f"{token} gateway tunnels", user_id=user, project_id=project, limit=30)
        episodes = [r for r in results if r.source == "episode" and token in r.memory]
        if episodes or time.monotonic() > deadline:
            break
        time.sleep(2)

    assert episodes, f"no episode row among {len(results)} results {[r.source for r in results]}"
    assert episodes[0].score is None
