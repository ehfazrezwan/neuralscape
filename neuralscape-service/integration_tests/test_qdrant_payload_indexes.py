"""Real Qdrant: payload indexes for every filtered field, and the BM25 signal.

Audit:
- bug:vec.qdrant-point-store:no-payload-index-on-metadata-project-id-metadata-c
- bug:vec.bm25-sparse:pre-v3-collection-silently-loses-the-keyword-leg

Needs the gate's stores (see conftest.py); skips cleanly without them.
"""

import time

import third_party_telemetry  # noqa: F401  (must precede any mem0 import)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

EXPECTED = (
    "hash",
    "metadata.project_id",
    "metadata.category",
    "metadata.scope",
    "metadata.visibility",
    "metadata.dream_tombstoned",
    "metadata.source_ref.external_id",
    "created_at",
)


def _wait_for_indexes(client, collection, timeout=30.0):
    deadline = time.monotonic() + timeout
    while True:
        schema = client.get_collection(collection).payload_schema
        missing = [k for k in EXPECTED if k not in schema]
        if not missing or time.monotonic() > deadline:
            return missing, sorted(schema)
        time.sleep(0.5)


def test_service_init_alone_creates_indexes(stores, llm_key, unique_ns, monkeypatch):
    """A real MemoryService against the gate's Qdrant: _get_memory() and nothing else.

    No search, no timeline, no direct call of the ensure methods: the init
    hook alone must leave every filter field indexed (workers dedup-probe
    before any search).
    """
    import config
    from memory_service import MemoryService

    collection = unique_ns.qdrant_collection
    client = QdrantClient(url=stores.qdrant_url)
    unique_ns.on_cleanup(lambda: client.delete_collection(collection))
    monkeypatch.setattr(config.settings, "qdrant_url", stores.qdrant_url)
    monkeypatch.setattr(config.settings, "qdrant_collection", collection)
    monkeypatch.setattr(config.settings, "neo4j_uri", stores.neo4j_uri)
    monkeypatch.setattr(config.settings, "neo4j_user", stores.neo4j_user)
    monkeypatch.setattr(config.settings, "neo4j_password", stores.neo4j_password)
    monkeypatch.setattr(config.settings, "neo4j_database", stores.neo4j_database)

    svc = MemoryService()
    svc._get_memory()
    thread = svc._index_ensure_thread
    thread.join(timeout=30)
    missing, have = _wait_for_indexes(client, collection)
    assert not missing, f"no payload index on {missing}; have {have}"
    # A fresh sparse-capable collection: health reports the slot.
    assert svc.vector_store_capabilities()["bm25_sparse"] is True


def test_pre_v3_collection_reports_no_bm25_slot(stores, unique_ns):
    """A collection created without the sparse slot is reported, not silently dense-only."""
    from types import SimpleNamespace

    from mem0.vector_stores.qdrant import Qdrant
    from memory_service import MemoryService

    collection = unique_ns.qdrant_collection
    client = QdrantClient(url=stores.qdrant_url)
    unique_ns.on_cleanup(lambda: client.delete_collection(collection))
    client.create_collection(collection, vectors_config=VectorParams(size=4, distance=Distance.COSINE))

    store = Qdrant(collection_name=collection, embedding_model_dims=4, client=client, url=None, api_key=None, path=None, on_disk=False)
    assert store._has_bm25_slot is False

    svc = MemoryService()
    svc._memory = SimpleNamespace(vector_store=store)
    assert svc.vector_store_capabilities() == {"bm25_sparse": False, "bm25_encoder": "not_loaded"}
