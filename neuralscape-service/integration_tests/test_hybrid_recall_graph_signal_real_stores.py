"""Hybrid recall reports a lost graph leg, against REAL Neo4j / Qdrant.

Real: the MemoryService, its Graphiti bridge, the Neo4j driver and Qdrant. One
vector-only row is stored (no graph write). The outage is real: the live Neo4j
driver is closed under the service, so the graph leg of ``search`` fails inside
Graphiti. The vector row must still come back and the response must say the
graph leg failed; a healthy graph must carry no signal.

Fails if the signal is removed: the outage then looks like "no graph facts".
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


def _store(service, unique_ns):
    return service.store_raw(
        content="zq hybrid signal row about the release train",
        user_id=unique_ns.user_id, category="convention", scope="global",
        visibility="private", add_to_graph=False,
    )[0].id


def test_healthy_graph_recall_has_no_signal(service, unique_ns):
    _store(service, unique_ns)
    out = service.search("release train", unique_ns.user_id, limit=5)
    assert out.graph_leg is None


def test_unreachable_graph_recall_reports_failed_and_keeps_vector_rows(service, unique_ns):
    mem_id = _store(service, unique_ns)
    service._run_on_bridge(service._graphiti.driver.close(), timeout=30.0)

    out = service.search("release train", unique_ns.user_id, limit=5)
    assert mem_id in {r.id for r in out}
    assert out.graph_leg == "failed"
