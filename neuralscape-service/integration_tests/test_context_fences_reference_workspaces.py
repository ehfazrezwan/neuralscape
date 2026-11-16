"""Session-start context vs reference workspaces, on real Qdrant.

An ingest-shaped caller-owned reference row (``workspace=ref-...``) and an
ordinary memory row share a project. ``get_project_context`` and
``get_global_context`` must return only the memory row, with ``total`` equal
to what is returned (the same filter drives the count), exactly as recall's
default does.
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


def _ids(ctx):
    return {m.id for rows in ctx.categories.values() for m in rows}


def test_context_returns_memory_rows_only_and_total_matches(svc, unique_ns):
    uid, project = unique_ns.user_id, unique_ns.project_id
    tag = uuid.uuid4().hex[:8]
    mem = svc.store_raw(
        content=f"zq-{tag} deploys go through the release train", user_id=uid,
        category="convention", scope="project", project_id=project,
        visibility="private", add_to_graph=False,
    )[0].id
    ref = svc.store_raw(
        content=f"zq-{tag} chapter 3 of the reference book", user_id=uid,
        category="domain_knowledge", scope="project", project_id=project,
        visibility="private", add_to_graph=False,
        workspace="ref-trading_strategy--book",
    )[0].id
    gmem = svc.store_raw(
        content=f"zq-{tag} I prefer terse answers", user_id=uid,
        category="preference", scope="global", visibility="private", add_to_graph=False,
    )[0].id
    gref = svc.store_raw(
        content=f"zq-{tag} reference glossary entry", user_id=uid,
        category="domain_knowledge", scope="global", visibility="private",
        add_to_graph=False, workspace="ref-trading_strategy--book",
    )[0].id

    ctx = svc.get_project_context(uid, project)
    assert _ids(ctx) == {mem, gmem}, _ids(ctx)
    assert ref not in _ids(ctx) and gref not in _ids(ctx)
    assert ctx.total == 2 and ctx.returned == 2 and ctx.has_more is False

    g = svc.get_global_context(uid)
    assert _ids(g) == {gmem}, _ids(g)
    assert g.total == 1 and g.has_more is False
