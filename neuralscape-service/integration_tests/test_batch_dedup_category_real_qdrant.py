"""One batch with identical text under two categories stores two rows.

Real Qdrant (throwaway collection) and the real embedder. ``store_raw_batch`` collapses repeats
inside a batch with the same identity the store-level exact-hash dedup uses, and that identity
includes the category. Before the fix the in-batch key left the category out, so the second item
was absorbed into the first and only one row was written.

Needs Neo4j (the service initialises its graph store), Qdrant, Redis and GOOGLE_API_KEY.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import pytest

pytestmark = pytest.mark.integration


def test_same_text_two_categories_in_one_batch_are_two_rows(flags, stores, llm_key, unique_ns):
    from qdrant_client import QdrantClient

    from memory_service import MemoryService

    flags(qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url)
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    service = MemoryService()
    service._get_memory()
    uid = unique_ns.user_id
    text = f"Keeps the harbour log in pencil ({uid[-6:]})"
    items = [
        {"content": text, "user_id": uid, "category": "preference"},
        {"content": text, "user_id": uid, "category": "personal_fact"},
        {"content": text, "user_id": uid, "category": "preference"},
    ]

    results = service.store_raw_batch(items)

    assert len(results) == 3
    assert results[0].id == results[2].id, "same text, same category: one row"
    assert results[0].id != results[1].id, "same text, other category: its own row"
    rows = [r["payload"] for r in service._scroll_all_user_memories(uid)]
    assert sorted((r.get("metadata") or {}).get("category") for r in rows) == ["personal_fact", "preference"], rows
