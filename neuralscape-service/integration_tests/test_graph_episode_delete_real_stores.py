"""Admin episode delete (``DELETE /v1/graph/episodes/{uuid}``) against a REAL Neo4j and a REAL Qdrant.

Same seeded scenario as ``test_graph_support_aware_removal_integration.py``: alice's memories A and B,
whose episodes both support the edge X->Y, while only A's episode supports X->Z.  Deleting A's episode
through the admin route must

* remove the edge only that episode supports (X->Z) and the entity nothing else references,
* keep the edge another episode still supports (X->Y), now supported by B's episode only,
* clear the episode reference on memory A's row (the row itself stays), and leave B's row untouched,
* leave no lock node behind.

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
    rows_collection,
    seed,
)


def _delete_episode_through_the_api(service, monkeypatch, flags, episode_uuid):
    from fastapi.testclient import TestClient

    import main

    monkeypatch.setattr(main, "_service", service)
    flags(default_user_id=OWNER)
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
