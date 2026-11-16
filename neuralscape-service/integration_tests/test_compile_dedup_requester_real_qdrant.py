"""Flag-on proof: an explicit compile dedups the requester on a day other users wrote to.

Real Qdrant and the real embedder, ``DEDUP_SEMANTIC_TOMBSTONE_ENABLED`` on. The vault's
daily log for the date holds a SHARED entry from ANOTHER user (the vault is one tree that
accumulates every user's shared facts), while the requester holds two near-identical
PRIVATE memories and so owns no entry. ``compile_date`` for the requester must still run
dedup for them: the older copy is tombstoned and recall hides it. Before the fix dedup ran
for the entry owners only, so the requester's pair was left alone.

Uses the integration fixtures (``stores``, ``llm_key``, ``flags``, ``unique_ns``) over a
throwaway collection dropped afterwards. The compile's LLM summary call is stubbed.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.integration


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
    return svc.store_raw(
        content=text, user_id=uid, category="convention", scope="project",
        project_id=project, visibility="private", add_to_graph=False,
    )[0].id


def _recalled(svc, uid, query, project):
    return {m.id for m in svc.search(query, user_id=uid, project_id=project, limit=20)}


def test_compile_on_a_day_with_other_users_entries_still_dedups_the_requester(svc, unique_ns, tmp_path):
    from extensions.conversation_compiler.compile import compile_date
    from extensions.conversation_compiler.obsidian_writer import ObsidianWriter

    pytest.importorskip("fcntl")
    uid, project = unique_ns.user_id, unique_ns.project_id
    tag = uuid.uuid4().hex[:8]
    older = f"zq-{tag} The nightly report job runs at 02:00 UTC on weekdays"
    old_id = _store(svc, uid, older, project)
    time.sleep(1.1)
    new_id = _store(svc, uid, f"{older}.", project)
    assert {old_id, new_id} <= _recalled(svc, uid, f"zq-{tag} nightly report", project)

    writer = ObsidianWriter(vault_path=tmp_path / "vault")
    writer.append_daily_log(
        "2026-11-16",
        [{
            "time": "10:00", "category": "decision", "content": "Someone else chose Qdrant",
            "session_id": "s", "user_id": f"other-{tag}",
        }],
    )

    with patch(
        "extensions.conversation_compiler.compile._async_call_gemini",
        new=AsyncMock(return_value="summary"),
    ):
        result = asyncio.run(compile_date("2026-11-16", svc, writer, user_id=uid))
    assert result.entries_compiled == 1 and result.dedup_triggered is True

    rows = {r["id"]: r["payload"] for r in svc._scroll_all_user_memories(uid)}
    assert (rows[old_id].get("metadata") or {}).get("dream_tombstoned") is True, rows[old_id]
    assert not (rows[new_id].get("metadata") or {}).get("dream_tombstoned")
    ids = _recalled(svc, uid, f"zq-{tag} nightly report", project)
    assert new_id in ids and old_id not in ids, ids
