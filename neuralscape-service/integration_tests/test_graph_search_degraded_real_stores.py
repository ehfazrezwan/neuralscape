"""``search_graph`` reports a graph outage as degraded, against REAL Neo4j / Qdrant.

Real: the MemoryService, its Graphiti bridge and the Neo4j driver.  The outage
is a real one: the live Neo4j driver is closed under the service, so the next
graph search fails inside Graphiti exactly as it would with the database
unreachable.  No data is written (nothing to clean in Neo4j) and no LLM tokens
are spent on the search itself.

Fails if the degraded marker is removed: the outage case then returns the
plain empty lists, indistinguishable from the healthy case.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (must precede any mem0/graphiti import)

import pytest


@pytest.fixture
def service(stores, llm_key, unique_ns, monkeypatch):
    import config

    s = config.settings
    for name, value in {
        "qdrant_url": stores.qdrant_url,
        "qdrant_collection": unique_ns.qdrant_collection,
        "neo4j_uri": stores.neo4j_uri,
        "neo4j_user": stores.neo4j_user,
        "neo4j_password": stores.neo4j_password,
        "neo4j_database": stores.neo4j_database,
    }.items():
        monkeypatch.setattr(s, name, value)

    from memory_service import MemoryService
    from qdrant_client import QdrantClient

    svc = MemoryService()
    svc._get_memory()
    assert svc._graphiti is not None and svc._bridge is not None, "graph store not initialised"

    def cleanup():
        try:
            client = QdrantClient(url=stores.qdrant_url)
            for name in (unique_ns.qdrant_collection, f"{unique_ns.qdrant_collection}_entities"):
                if client.collection_exists(name):
                    client.delete_collection(name)
        finally:
            try:
                svc.close()
            except Exception:  # the driver is deliberately closed in the outage test
                pass

    unique_ns.on_cleanup(cleanup)
    return svc


def test_healthy_graph_reports_no_degraded_marker(service, unique_ns):
    out = service.search_graph("anything at all", unique_ns.user_id, limit=3)
    assert "degraded" not in out and "error" not in out
    assert set(out) == {"edges", "nodes", "episodes", "communities"}


def test_unreachable_graph_reports_degraded(service, unique_ns):
    healthy = service.search_graph("anything at all", unique_ns.user_id, limit=3)
    assert "degraded" not in healthy

    # A real outage: the live driver is gone, so the next Graphiti query fails.
    service._run_on_bridge(service._graphiti.driver.close(), timeout=30.0)

    out = service.search_graph("anything at all", unique_ns.user_id, limit=3)
    assert out.get("degraded") is True
    assert out.get("error") == "graph_unavailable"
    assert out["edges"] == [] and out["nodes"] == []
