"""Flag-on proof: compile -> dedup -> recall, through the real code path.

Real Qdrant and the real embedder, ``DEDUP_SEMANTIC_TOMBSTONE_ENABLED`` on. A
fresh user holds two near-identical PRIVATE memories (a private memory never
reaches the vault, so the vault has no entry for this user: the shape the
acceptance stack hits). ``compile_date`` for an empty date must still run the
post-compile dedup, which tombstones the older copy, and then:

* the embedder scores the pair above the dedup threshold (asserted, so a
  too-low similarity fails loudly instead of looking like a dedup bug);
* recall and listings hide the tombstone, the audit listing keeps it;
* deriving the older text again revives it.

Uses the integration fixtures (``stores``, ``llm_key``, ``flags``,
``unique_ns``) over a throwaway collection dropped afterwards.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import math
import time
import uuid

import pytest


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns):
    from qdrant_client import QdrantClient

    flags(
        qdrant_collection=unique_ns.qdrant_collection,
        qdrant_url=stores.qdrant_url,
        dedup_semantic_tombstone_enabled=True,
    )
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    return service


def _store(svc, uid, text, project):
    out = svc.store_raw(
        content=text, user_id=uid, category="convention", scope="project",
        project_id=project, visibility="private", add_to_graph=False,
    )
    return out[0].id


def _recalled(svc, uid, query, project):
    return {m.id for m in svc.search(query, user_id=uid, project_id=project, limit=20)}


def test_compile_dedup_tombstones_the_older_copy_and_recall_hides_it(svc, unique_ns, tmp_path):
    from config import settings
    from extensions.conversation_compiler.compile import compile_date
    from extensions.conversation_compiler.obsidian_writer import ObsidianWriter

    uid, project = unique_ns.user_id, unique_ns.project_id
    tag = uuid.uuid4().hex[:8]
    older = f"zq-{tag} The nightly report job runs at 02:00 UTC on weekdays"
    newer = f"{older}."

    old_id = _store(svc, uid, older, project)
    time.sleep(1.1)
    new_id = _store(svc, uid, newer, project)

    # The embedder must actually score this pair above the dedup threshold.
    emb = svc._get_memory().embedding_model
    va, vb = emb.embed(older, memory_action="add"), emb.embed(newer, memory_action="add")
    cos = sum(x * y for x, y in zip(va, vb)) / (
        math.sqrt(sum(x * x for x in va)) * math.sqrt(sum(y * y for y in vb))
    )
    assert cos >= settings.dedup_similarity_threshold, (
        f"the pair scores {cos:.4f}, below dedup_similarity_threshold={settings.dedup_similarity_threshold}"
    )
    assert {old_id, new_id} <= _recalled(svc, uid, f"zq-{tag} nightly report", project)

    # compile an EMPTY vault date, as the REST route does for this user
    writer = ObsidianWriter(vault_path=tmp_path / "vault")
    result = asyncio.run(compile_date("2026-11-16", svc, writer, user_id=uid))
    assert result.entries_compiled == 0
    assert result.dedup_triggered is True

    rows = {r["id"]: r["payload"] for r in svc._scroll_all_user_memories(uid)}
    meta_old = rows[old_id].get("metadata") or {}
    assert meta_old.get("dream_tombstoned") is True, meta_old
    assert meta_old.get("superseded_by") == new_id
    assert meta_old.get("superseded_reason") == "semantic_dedup"
    assert not (rows[new_id].get("metadata") or {}).get("dream_tombstoned")

    ids = _recalled(svc, uid, f"zq-{tag} nightly report", project)
    assert new_id in ids and old_id not in ids, ids
    listed = {m.id for m in svc.list_memories(user_id=uid, project_id=project, limit=100)}
    assert new_id in listed and old_id not in listed
    audit = {m.id for m in svc.list_memories(user_id=uid, project_id=project, limit=100, include_tombstoned=True)}
    assert old_id in audit

    # reversible: deriving the older fact again revives it
    _store(svc, uid, older, project)
    revived = {r["id"]: r["payload"] for r in svc._scroll_all_user_memories(uid)}[old_id]
    assert (revived.get("metadata") or {}).get("dream_tombstoned") is False
    assert old_id in _recalled(svc, uid, f"zq-{tag} nightly report", project)
