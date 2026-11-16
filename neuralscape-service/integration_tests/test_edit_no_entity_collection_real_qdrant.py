"""A content edit and a delete must not create mem0's ``<collection>_entities`` collection (real Qdrant).

mem0 creates that collection lazily on the first entity write and only its native
search reads it; the service never calls that search.  The edit path used to
re-link entities on every content edit, so the first edit created the collection
and paid an embedding call plus an insert per entity.  This drives the service's
real ``patch_memory`` / ``delete_memory`` over a real mem0 ``Memory`` on real
Qdrant and asserts the collection never appears.  The embedder is a local
deterministic stub (no LLM call; the embedding is not what is under test).

Env: QDRANT_URL via the ``stores`` fixture (unreachable fails under NS_INTEGRATION_REQUIRED).
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (must precede any mem0/graphiti import)

import uuid
from types import SimpleNamespace

import pytest

DIMS = 768  # embedding_dims in Settings.get_mem0_config()


def _embed(text, action="add"):
    vec = [0.0] * DIMS
    vec[sum(map(ord, text)) % DIMS] = 1.0
    return vec


@pytest.fixture
def service_settings(stores, unique_ns, monkeypatch):
    import config
    from qdrant_client import QdrantClient

    s = config.settings
    for name, value in {"qdrant_url": stores.qdrant_url, "qdrant_collection": unique_ns.qdrant_collection}.items():
        monkeypatch.setattr(s, name, value)
    if getattr(s, "llm_gateway_enabled", False):
        monkeypatch.setattr(s, "llm_gateway_enabled", False)
    if not s.google_api_key:
        monkeypatch.setattr(s, "google_api_key", "unused-no-llm-call")
    monkeypatch.setattr(s, "redis_url", stores.redis_url)

    def drop_collections():
        client = QdrantClient(url=stores.qdrant_url)
        for name in (unique_ns.qdrant_collection, f"{unique_ns.qdrant_collection}_entities"):
            try:
                client.delete_collection(name)
            except Exception:  # noqa: BLE001 - already gone
                pass

    unique_ns.on_cleanup(drop_collections)
    return s


@pytest.fixture
def world(service_settings, unique_ns):
    from mem0 import Memory
    from memory.core import _memory_from_config
    from memory_service import MemoryService

    memory = _memory_from_config(Memory, service_settings.get_mem0_config())
    memory.embedding_model = SimpleNamespace(embed=_embed)
    svc = MemoryService()
    svc._memory = memory
    svc._graphiti = None
    svc._bridge = None
    svc.owner = unique_ns.user_id
    return svc


def _collections(client) -> set[str]:
    return {c.name for c in client.get_collections().collections}


def test_content_edit_and_delete_never_create_the_entity_collection(world, unique_ns):
    entities = f"{unique_ns.qdrant_collection}_entities"
    vs = world._memory.vector_store
    client = vs.client
    owner = world.owner

    pid = str(uuid.uuid4())
    text = "Alice Johnson joined Acme Corporation in Berlin"
    vs.insert(
        vectors=[_embed(text)],
        ids=[pid],
        payloads=[{
            "data": text, "hash": "h", "created_at": "2026-01-01T00:00:00+00:00", "user_id": owner,
            "metadata": {"owner_user_id": owner, "visibility": "private", "category": "personal_fact",
                         "scope": "global", "project_id": None, "tags": []},
        }],
    )
    assert entities not in _collections(client)

    out = world.patch_memory(pid, owner, {"content": "Bob Smith left Globex Industries for Tokyo"})
    assert out["memory"].memory.startswith("Bob Smith")
    assert entities not in _collections(client), "a content edit created mem0's unread entity collection"

    world.delete_memory(pid, owner)
    assert client.retrieve(unique_ns.qdrant_collection, ids=[pid]) == []
    assert entities not in _collections(client), "edit + delete created mem0's unread entity collection"
