"""Real Qdrant: payload indexes for every filtered field, and the BM25 signal.

Audit:
- bug:vec.qdrant-point-store:no-payload-index-on-metadata-project-id-metadata-c
- bug:vec.bm25-sparse:pre-v3-collection-silently-loses-the-keyword-leg

Needs QDRANT_URL (see conftest.py); skips cleanly without it.
"""

from types import SimpleNamespace

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


def _service(client, collection, monkeypatch, has_bm25=True):
    import config
    from memory_service import MemoryService

    monkeypatch.setattr(config.settings, "qdrant_collection", collection)
    svc = MemoryService()
    svc._memory = SimpleNamespace(vector_store=SimpleNamespace(client=client, _has_bm25_slot=has_bm25, _bm25_encoder=None))
    return svc


def test_indexes_present_after_ensure_and_idempotent(stores, unique_ns, monkeypatch):
    collection = unique_ns.qdrant_collection
    client = QdrantClient(url=stores.qdrant_url)
    unique_ns.on_cleanup(lambda: client.delete_collection(collection))
    client.create_collection(collection, vectors_config=VectorParams(size=4, distance=Distance.COSINE))

    svc = _service(client, collection, monkeypatch)
    svc._ensure_filter_indexes()
    svc._ensure_created_at_index()
    # A restart: a new service against the already-indexed collection.
    _service(client, collection, monkeypatch)._ensure_filter_indexes()

    schema = client.get_collection(collection).payload_schema
    missing = [k for k in EXPECTED if k not in schema]
    assert not missing, f"no payload index on {missing}; have {sorted(schema)}"


def test_pre_v3_collection_reports_no_bm25_slot(stores, unique_ns, monkeypatch):
    """A collection created without the sparse slot is reported, not silently dense-only."""
    from mem0.vector_stores.qdrant import Qdrant

    collection = unique_ns.qdrant_collection
    client = QdrantClient(url=stores.qdrant_url)
    unique_ns.on_cleanup(lambda: client.delete_collection(collection))
    client.create_collection(collection, vectors_config=VectorParams(size=4, distance=Distance.COSINE))

    store = Qdrant(collection_name=collection, embedding_model_dims=4, client=client, url=None, api_key=None, path=None, on_disk=False)
    assert store._has_bm25_slot is False

    svc = _service(client, collection, monkeypatch)
    svc._memory.vector_store = store
    assert svc.vector_store_capabilities() == {"bm25_sparse": False, "bm25_encoder": "not_loaded"}
