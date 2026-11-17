"""Keyword scan vs reference workspaces, on real Qdrant.

``keyword_search`` (the grep-style pass behind ``ask`` at the medium and high
tiers) must return a memory-workspace row that matches the keywords and must
not return a reference-workspace row that matches them, exactly as recall's
default and the session-start context do.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import uuid

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


def test_keyword_scan_skips_reference_rows_and_keeps_memory_rows(svc, unique_ns):
    uid, project = unique_ns.user_id, unique_ns.project_id
    tag = uuid.uuid4().hex[:8]
    mem = svc.store_raw(
        content=f"zq{tag} vermiculite pipeline runs every Thursday", user_id=uid,
        category="convention", scope="project", project_id=project,
        visibility="private", add_to_graph=False,
    )[0].id
    ref = svc.store_raw(
        content=f"zq{tag} vermiculite pipeline chapter of the reference book", user_id=uid,
        category="domain_knowledge", scope="project", project_id=project,
        visibility="private", add_to_graph=False,
        workspace="ref-trading_strategy--book",
    )[0].id

    rows, _capped = svc.keyword_search(uid, [f"zq{tag}", "vermiculite"], project_id=project)
    ids = {r.id for r in rows}
    assert mem in ids, ids
    assert ref not in ids, ids
