"""Admin episode delete (``DELETE /v1/graph/episodes/{uuid}``) against a REAL Neo4j and a REAL Qdrant.

Same seeded scenario as ``test_graph_support_aware_removal_integration.py``: alice's memories A and B,
whose episodes both support the edge X->Y, while only A's episode supports X->Z.  Deleting A's episode
through the admin route must

* remove the edge only that episode supports (X->Z) and the entity nothing else references,
* keep the edge another episode still supports (X->Y), now supported by B's episode only,
* clear the episode reference on memory A's row (the row itself stays), and leave B's row untouched,
* leave no lock node behind.

A shared-group episode is removable only by the owner of the memory that references it: a second user who
can read the shared group is refused (403) and nothing is touched.

The junk-episode sweep (``DELETE /v1/graph/episodes/junk``) deletes the junk episodes the caller may delete and
SKIPS (never aborts on) a junk episode in the shared group that another user owns, reporting it in
``skipped_count`` / ``skipped_uuids`` -- on a dry run too, where ``junk_count`` is what a real run deletes.  The same
episode is deleted by its owner, so the skip is the permission rule and not the episode's content.

A missing store fails the test on the gate and skips it elsewhere (``integration_tests/README.md``).
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import pytest

from integration_tests.test_graph_support_aware_removal_integration import (  # noqa: F401  (fixtures + helpers)
    MEM_A,
    MEM_B,
    OWNER,
    _clean_rows,
    edge,
    exists,
    graph,
    locks_left,
    make_service,
    node,
    q,
    rows_collection,
    seed,
)


def _delete_episode_through_the_api(service, monkeypatch, flags, episode_uuid, caller=OWNER):
    from fastapi.testclient import TestClient

    import main

    monkeypatch.setattr(main, "_service", service)
    flags(default_user_id=caller)
    flags(ledger_enabled=False)
    flags(neuralscape_auth_disabled=True)
    flags(neuralscape_user_token_secret="")
    flags(neuralscape_api_key="")
    return TestClient(main.app).delete(f"/v1/graph/episodes/{episode_uuid}")


def test_admin_episode_delete_cascades_edges_and_clears_the_row_reference(graph, rows_collection, monkeypatch, flags):
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)

    response = _delete_episode_through_the_api(service, monkeypatch, flags, ids["EA"])
    assert response.status_code == 200, response.text

    assert not exists(graph, "Episodic", ids["EA"]) and exists(graph, "Episodic", ids["EB"])
    shared = edge(graph, ids["XY"])
    assert shared is not None and shared["expired"] is False
    assert shared["episodes"] == [ids["EB"]]  # still live, supported by B only
    assert edge(graph, ids["XZ"]) is None  # its only support was the deleted episode
    assert not exists(graph, "Entity", ids["Z"])
    assert node(graph, ids["X"])["summary"] == "" and node(graph, ids["Y"])["summary"] == ""
    assert locks_left(graph) == 0

    assert service._memory.get(MEM_A) is not None  # the memory row itself stays
    assert service._memory.meta(MEM_A).get("graph_episode_uuid") is None  # no dangling reference
    assert service._memory.meta(MEM_B).get("graph_episode_uuid") == ids["EB"]


def test_admin_episode_delete_of_an_unknown_episode_is_404(graph, rows_collection, monkeypatch, flags):
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    response = _delete_episode_through_the_api(service, monkeypatch, flags, "no-such-episode")
    assert response.status_code == 404
    assert exists(graph, "Episodic", ids["EA"])


SHARED_MEMORY = "00000000-0000-4000-8000-0000000000bb"


def _seed_shared(graph, service):
    import uuid as _uuid

    graphiti, bridge = graph
    ids = {"EP": str(_uuid.uuid4()), "E1": str(_uuid.uuid4()), "A": str(_uuid.uuid4()), "B": str(_uuid.uuid4())}
    q(graphiti, bridge, "MATCH (n {group_id: 'shared', name: 'nsit-shared'}) DETACH DELETE n")
    q(graphiti, bridge, """
        CREATE (a:Entity {uuid: $A, group_id: 'shared', name: 'nsit-shared', summary: 's', created_at: datetime()})
        CREATE (b:Entity {uuid: $B, group_id: 'shared', name: 'nsit-shared', summary: 's', created_at: datetime()})
        CREATE (e:Episodic {uuid: $EP, group_id: 'shared', name: 'nsit-shared', content: 'shared fact', source: 'text',
                            source_description: 'memory', created_at: datetime(), valid_at: datetime(),
                            entity_edges: [$E1]})
        CREATE (e)-[:MENTIONS {uuid: randomUUID(), group_id: 'shared', created_at: datetime()}]->(a)
        CREATE (e)-[:MENTIONS {uuid: randomUUID(), group_id: 'shared', created_at: datetime()}]->(b)
        CREATE (a)-[:RELATES_TO {uuid: $E1, group_id: 'shared', name: 'R', fact: 'f', episodes: [$EP],
                                 created_at: datetime()}]->(b)
        """, **ids)
    service._memory.put(SHARED_MEMORY, "shared fact",
                        {"owner_user_id": OWNER, "visibility": "shared", "graph_episode_uuid": ids["EP"]})
    return ids


def test_reader_of_the_shared_group_cannot_delete_the_owners_episode_but_the_owner_can(
    graph, rows_collection, monkeypatch, flags
):
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    shared = _seed_shared(graph, service)
    try:
        refused = _delete_episode_through_the_api(service, monkeypatch, flags, shared["EP"], caller="nsit-reader")
        assert refused.status_code == 403, refused.text
        assert exists(graph, "Episodic", shared["EP"])
        assert edge(graph, shared["E1"]) is not None and edge(graph, shared["E1"])["expired"] is False
        assert service._memory.meta(SHARED_MEMORY).get("graph_episode_uuid") == shared["EP"]

        allowed = _delete_episode_through_the_api(service, monkeypatch, flags, shared["EP"], caller=OWNER)
        assert allowed.status_code == 200, allowed.text
        assert not exists(graph, "Episodic", shared["EP"])
        assert edge(graph, shared["E1"]) is None
        assert service._memory.meta(SHARED_MEMORY).get("graph_episode_uuid") is None
    finally:
        graphiti, bridge = graph
        q(graphiti, bridge, "MATCH (n {group_id: 'shared', name: 'nsit-shared'}) DETACH DELETE n")


def _sweep_through_the_api(service, monkeypatch, flags, caller, dry_run):
    from fastapi.testclient import TestClient

    import main

    monkeypatch.setattr(main, "_service", service)
    flags(default_user_id=caller)
    flags(ledger_enabled=False)
    flags(neuralscape_auth_disabled=True)
    flags(neuralscape_user_token_secret="")
    flags(neuralscape_api_key="")
    return TestClient(main.app).delete("/v1/graph/episodes/junk", params={"dry_run": "true" if dry_run else "false"})


def _seed_junk(graph, service, *, group, memory_id, owner, visibility, name):
    """One junk ("assistant:" log) episode in ``group`` plus the row that records its owner."""
    import uuid as _uuid

    graphiti, bridge = graph
    episode = str(_uuid.uuid4())
    q(graphiti, bridge, """
        CREATE (:Episodic {uuid: $EP, group_id: $g, name: $name, content: 'assistant: nsit junk log', source: 'text',
                           source_description: 'memory', created_at: datetime(), valid_at: datetime(),
                           entity_edges: []})
        """, EP=episode, g=group, name=name)
    service._memory.put(memory_id, "assistant: nsit junk log",
                        {"owner_user_id": owner, "visibility": visibility, "graph_episode_uuid": episode})
    return episode


def test_junk_sweep_skips_another_users_shared_episode_reports_it_and_still_deletes_its_own(
    graph, rows_collection, monkeypatch, flags
):
    import uuid as _uuid

    from memory.groups import _build_group_id

    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    reader = f"nsit-sweeper-{_uuid.uuid4().hex[:8]}"
    own_group = _build_group_id("private", reader, None, None)
    name = f"nsit-junk-{_uuid.uuid4().hex[:8]}"
    graphiti, bridge = graph
    try:
        mine = _seed_junk(graph, service, group=own_group, memory_id=str(_uuid.uuid4()), owner=reader,
                          visibility="private", name=name)
        theirs = _seed_junk(graph, service, group="shared", memory_id=str(_uuid.uuid4()), owner=OWNER,
                            visibility="shared", name=name)

        # Dry run: what it would delete excludes the refused episode, which is reported as skipped.
        dry = _sweep_through_the_api(service, monkeypatch, flags, reader, dry_run=True)
        assert dry.status_code == 200, dry.text
        body = dry.json()
        assert theirs in body.get("skipped_uuids", []), body
        assert body["skipped_count"] == len(body["skipped_uuids"]) >= 1, body
        assert mine not in body.get("skipped_uuids", []), body
        assert body["junk_count"] >= 1 and body["breakdown"]["global"]["junk_count"] >= 1, body
        assert exists(graph, "Episodic", mine) and exists(graph, "Episodic", theirs)  # nothing deleted yet

        # Real run: its own junk is deleted, the other user's shared episode is skipped and untouched.
        real = _sweep_through_the_api(service, monkeypatch, flags, reader, dry_run=False)
        assert real.status_code == 200, real.text
        body = real.json()
        assert body["deleted_count"] == len(body["breakdown"]["global"]["deleted_uuids"]) >= 1, body
        assert mine in body["breakdown"]["global"]["deleted_uuids"], body
        assert theirs not in body["breakdown"]["global"]["deleted_uuids"], body
        assert theirs in body["skipped_uuids"] and body["skipped_count"] == len(body["skipped_uuids"]), body
        assert theirs in body["breakdown"]["global"]["skipped_uuids"], body
        assert not exists(graph, "Episodic", mine)
        assert exists(graph, "Episodic", theirs)

        # Control: the same episode IS deleted when its owner sweeps, so the skip was the permission rule.
        owner_run = _sweep_through_the_api(service, monkeypatch, flags, OWNER, dry_run=False)
        assert owner_run.status_code == 200, owner_run.text
        owner_body = owner_run.json()
        assert theirs in owner_body["breakdown"]["global"]["deleted_uuids"], owner_body
        assert theirs not in owner_body.get("skipped_uuids", []), owner_body
        assert not exists(graph, "Episodic", theirs)
    finally:
        q(graphiti, bridge, "MATCH (n:Episodic {name: $name}) DETACH DELETE n", name=name)
