"""``list_projects`` lists projects that exist only in the vector store, against REAL Qdrant.

Rows are written with ``store_raw(add_to_graph=False)``, so no graph episode
exists for either project (the "enrichment pending / graph failing" state).
Before the fix the list came from Neo4j group ids alone and was empty here.
Another user's private project must never appear, a shared one must.

One small embedding call per row (``llm_key``); nothing else spends tokens.
Fails if ``list_projects`` goes back to deriving projects from the graph only.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (must precede any mem0/graphiti import)

import pytest


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns):
    from qdrant_client import QdrantClient

    flags(qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url)
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    return service


def _store(svc, uid, text, project, visibility="private"):
    svc.store_raw(
        content=text, user_id=uid, category="convention", scope="project",
        project_id=project, visibility=visibility, add_to_graph=False,
    )


def test_projects_with_rows_but_no_graph_episodes_are_listed(svc, unique_ns):
    uid = unique_ns.user_id
    other = f"{uid}-other"
    p1, p2 = f"{unique_ns.project_id}-a", f"{unique_ns.project_id}-b"
    secret, team = f"{unique_ns.project_id}-secret", f"{unique_ns.project_id}-team"

    _store(svc, uid, f"{p1} uses tabs for indentation in the billing module.", p1)
    _store(svc, uid, f"{p2} deploys on Fridays after the freeze review.", p2)
    _store(svc, other, f"{secret} keeps its roadmap in a private notebook.", secret)
    _store(svc, other, f"{team} publishes release notes every other Tuesday.", team, visibility="shared")

    out = svc.list_projects_status(uid)
    assert p1 in out["projects"] and p2 in out["projects"], out
    assert secret not in out["projects"], "another user's private project leaked"
    assert team in out["projects"], "shared project row not listed"
    assert out["projects"] == sorted(set(out["projects"]))
    # Neo4j is healthy in this stack: no degraded marker.
    assert "degraded" not in out, out
