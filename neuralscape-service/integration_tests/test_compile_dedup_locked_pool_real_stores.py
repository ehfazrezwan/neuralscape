"""Post-compile dedup reports a pool a sweep holds, on real Redis and real Qdrant.

A sweep marks a pool busy with ``dreaming:lock:{pool}``. While that key is held, the dedup an
explicit compile triggers skips the pool; the compile result must say so (reason, per-phase
counter, ``dedup_triggered`` false) and leave the pool's near-duplicate pair alone. Once the
lock is released the same compile deduplicates the pair and reports nothing skipped. Fails if
the compile drops the held-pool counter again, or if dedup stops honouring the lock.

Uses the integration fixtures (``stores``, ``llm_key``, ``flags``, ``unique_ns``) over a
throwaway Qdrant collection and a pool key unique to the test user. The compile's LLM summary
call is stubbed.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.integration

DATE = "2026-11-17"


@pytest.fixture
def redis_client(stores, unique_ns, monkeypatch):
    import redis as redis_lib

    client = redis_lib.Redis.from_url(stores.redis_url, socket_timeout=5)
    client.ping()
    monkeypatch.setattr("extensions.dreaming.sweep._get_redis", lambda: client)
    pools: list[str] = []
    unique_ns.on_cleanup(lambda: [client.delete(f"dreaming:lock:{p}") for p in pools])
    client.nsit_pools = pools  # test-owned bookkeeping: pools whose lock key must be cleared
    return client


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns, redis_client):
    from qdrant_client import QdrantClient

    flags(
        qdrant_collection=unique_ns.qdrant_collection,
        qdrant_url=stores.qdrant_url,
        redis_url=stores.redis_url,
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


def _rows(svc, uid):
    return {r["id"]: r["payload"] for r in svc._scroll_all_user_memories(uid)}


def _tombstoned(rows, mid) -> bool:
    return mid not in rows or bool((rows[mid].get("metadata") or {}).get("dream_tombstoned"))


def _compile(svc, writer, uid):
    from extensions.conversation_compiler.compile import compile_date

    with patch(
        "extensions.conversation_compiler.compile._async_call_gemini",
        new=AsyncMock(return_value="summary"),
    ):
        return asyncio.run(compile_date(DATE, svc, writer, user_id=uid))


def test_compile_reports_a_pool_a_sweep_holds_and_dedups_it_once_released(
    svc, redis_client, unique_ns, tmp_path, monkeypatch
):
    from config import settings
    from extensions.conversation_compiler.obsidian_writer import ObsidianWriter
    from extensions.dreaming import gate
    from extensions.dreaming.config import dreaming_settings
    from memory.delete import _dream_pool_of

    pytest.importorskip("fcntl")
    # dreaming off: the semantic phase runs, and dedup takes the pool lock whenever Redis answers
    monkeypatch.setattr(dreaming_settings, "enabled", False)

    uid, project = unique_ns.user_id, unique_ns.project_id
    older = f"zq-{uuid.uuid4().hex[:8]} The nightly report job runs at 02:00 UTC on weekdays"
    old_id = _store(svc, uid, older, project)
    time.sleep(1.1)
    new_id = _store(svc, uid, f"{older}.", project)

    embed = svc._memory.embedding_model.embed
    va, vb = embed(older, memory_action="add"), embed(f"{older}.", memory_action="add")
    cosine = sum(x * y for x, y in zip(va, vb)) / (
        (sum(x * x for x in va) ** 0.5) * (sum(y * y for y in vb) ** 0.5)
    )
    assert cosine >= settings.dedup_similarity_threshold, cosine  # else "untouched" proves nothing

    rows = _rows(svc, uid)
    assert not _tombstoned(rows, old_id) and not _tombstoned(rows, new_id)
    pool = _dream_pool_of(rows[old_id])
    assert pool and pool == _dream_pool_of(rows[new_id])
    redis_client.nsit_pools.append(pool)

    writer = ObsidianWriter(vault_path=tmp_path / "vault")
    writer.append_daily_log(
        DATE,
        [{"time": "10:00", "category": "decision", "content": "Chose Qdrant",
          "session_id": "s", "user_id": uid}],
    )

    # a sweep holds the pool, exactly as the dream sweep takes it
    sweep_token = gate.acquire_lock(redis_client, pool)
    assert sweep_token, "the pool lock must be free before the test starts"

    held = _compile(svc, writer, uid)

    rows = _rows(svc, uid)
    seen = held.model_dump(exclude={"articles"})
    assert not _tombstoned(rows, old_id) and not _tombstoned(rows, new_id), seen
    assert held.dedup_semantic_skipped_locked_pools == 1, seen
    assert held.dedup_triggered is False and held.dedup_skipped_reason == "dream lock held or lost", seen
    assert held.dedup_skipped == 0, seen  # the lock store answered: that counter keeps its meaning
    assert redis_client.get(f"dreaming:lock:{pool}").decode() == sweep_token, "the sweep's lock is untouched"

    gate.release_lock(redis_client, pool, sweep_token)
    assert redis_client.get(f"dreaming:lock:{pool}") is None

    done = _compile(svc, writer, uid)

    rows = _rows(svc, uid)
    seen = done.model_dump(exclude={"articles"})
    assert done.dedup_triggered is True and done.dedup_skipped_reason is None, seen
    assert (
        done.dedup_exact_skipped_locked_pools,
        done.dedup_semantic_skipped_locked_pools,
        done.dedup_lock_lost_pools,
    ) == (0, 0, 0), seen
    assert _tombstoned(rows, old_id) and not _tombstoned(rows, new_id), seen
    assert redis_client.get(f"dreaming:lock:{pool}") is None, "dedup released the lock it took"
