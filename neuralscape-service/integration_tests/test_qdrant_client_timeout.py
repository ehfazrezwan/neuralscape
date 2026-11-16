"""The service's real Qdrant client carries the configured timeout (default 30 s).

Audit: the service built its Qdrant client with no timeout, so qdrant-client's ~5 s default applied
to every call, including dreaming pool scrolls and 500-id retrieves on a loaded host.
Real Qdrant, no LLM: the memory service is initialised and the live client is inspected, then a
real call is made through it.
"""

import third_party_telemetry  # noqa: F401  (must precede any mem0 import)
from qdrant_client import QdrantClient


def _service(stores, unique_ns, monkeypatch, timeout):
    import config
    from memory_service import MemoryService

    collection = unique_ns.qdrant_collection
    cleanup = QdrantClient(url=stores.qdrant_url)
    unique_ns.on_cleanup(lambda: cleanup.delete_collection(collection))
    monkeypatch.setattr(config.settings, "qdrant_url", stores.qdrant_url)
    monkeypatch.setattr(config.settings, "qdrant_collection", collection)
    monkeypatch.setattr(config.settings, "qdrant_timeout", timeout)
    monkeypatch.setattr(config.settings, "neo4j_uri", stores.neo4j_uri)
    monkeypatch.setattr(config.settings, "neo4j_user", stores.neo4j_user)
    monkeypatch.setattr(config.settings, "neo4j_password", stores.neo4j_password)
    monkeypatch.setattr(config.settings, "neo4j_database", stores.neo4j_database)
    svc = MemoryService()
    return svc, svc._get_memory()


def test_service_client_uses_configured_timeout(stores, llm_key, unique_ns, monkeypatch):
    svc, memory = _service(stores, unique_ns, monkeypatch, 41)
    client = memory.vector_store.client
    # Private attribute on purpose: qdrant-client exposes no public getter for the configured timeout.
    assert client._client._timeout == 41
    # The client is live against the real store with that timeout.
    assert client.collection_exists(unique_ns.qdrant_collection)


def test_default_timeout_is_not_the_client_default(stores, llm_key, unique_ns, monkeypatch):
    import config

    default = config.Settings.model_fields["qdrant_timeout"].default
    svc, memory = _service(stores, unique_ns, monkeypatch, default)
    assert memory.vector_store.client._client._timeout == default == config.DEFAULT_QDRANT_TIMEOUT_S == 30
