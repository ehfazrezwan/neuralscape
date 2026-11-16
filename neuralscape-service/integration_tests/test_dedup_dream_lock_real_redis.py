"""Dedup respects the dream sweep's pool lock, on real Qdrant and real Redis.

A sweep marks a pool busy with ``dreaming:lock:{pool}``. While that key is held,
``dedup_memories`` must not delete or tombstone a row of the pool, in either phase (exact
hash and semantic), and must report the pool as skipped; once the lock is released the same duplicates are handled. This test fails if
dedup stops honouring the lock (wrong key, no check) or if the lock's compare-and-renew /
compare-and-delete scripts misbehave on a real server.

Uses the integration fixtures (``stores``, ``llm_key``, ``flags``, ``unique_ns``) over a
throwaway Qdrant collection and a pool key unique to the test user.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import time
import uuid

import pytest

pytestmark = pytest.mark.integration


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


def _gone_or_tombstoned(rows, mid) -> bool:
    return mid not in rows or bool((rows[mid].get("metadata") or {}).get("dream_tombstoned"))


@pytest.mark.parametrize("tombstone", [False, True], ids=["hard-delete", "tombstone"])
def test_dedup_skips_a_pool_a_sweep_holds_and_handles_it_once_released(
    svc, flags, redis_client, unique_ns, monkeypatch, tombstone
):
    from config import settings
    from extensions.dreaming import gate
    from extensions.dreaming.config import dreaming_settings
    from memory.delete import _dream_pool_of

    # dreaming off: phase 2 runs (a forced sweep runs with dreaming off too); the lock is
    # taken whenever Redis answers.
    monkeypatch.setattr(dreaming_settings, "enabled", False)
    flags(dedup_semantic_tombstone_enabled=tombstone)

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
    assert not _gone_or_tombstoned(rows, old_id) and not _gone_or_tombstoned(rows, new_id)
    pool = _dream_pool_of(rows[old_id])
    assert pool and pool == _dream_pool_of(rows[new_id])
    redis_client.nsit_pools.append(pool)

    # a sweep holds the pool, exactly as the dream sweep takes it
    sweep_token = gate.acquire_lock(redis_client, pool)
    assert sweep_token, "the pool lock must be free before the test starts"

    held = svc.dedup_memories(uid)

    rows = _rows(svc, uid)
    assert not _gone_or_tombstoned(rows, old_id) and not _gone_or_tombstoned(rows, new_id), held
    assert held["semantic_skipped_locked_pools"] == 1 and held["semantic_duplicates_removed"] == 0, held
    assert redis_client.get(f"dreaming:lock:{pool}").decode() == sweep_token, "the sweep's lock is untouched"

    gate.release_lock(redis_client, pool, sweep_token)
    assert redis_client.get(f"dreaming:lock:{pool}") is None

    done = svc.dedup_memories(uid)

    rows = _rows(svc, uid)
    assert done["semantic_skipped_locked_pools"] == 0 and done["semantic_duplicates_removed"] == 1, done
    assert _gone_or_tombstoned(rows, old_id) and not _gone_or_tombstoned(rows, new_id)
    assert (old_id in rows) is tombstone  # tombstone mode keeps the row, hard delete removes it
    assert redis_client.get(f"dreaming:lock:{pool}") is None, "dedup released the lock it took"


def test_exact_phase_skips_a_pool_a_sweep_holds_and_handles_it_once_released(
    svc, flags, redis_client, unique_ns
):
    """The exact-hash phase honours the same lock: two rows with one hash, one hard delete."""
    from extensions.dreaming import gate
    from memory.delete import _dream_pool_of

    uid, project = unique_ns.user_id, unique_ns.project_id
    text = f"zq-{uuid.uuid4().hex[:8]} The release train leaves on the first Monday of the month"
    old_id = _store(svc, uid, text, project)
    time.sleep(1.1)
    new_id = _store(svc, uid, f"{text} (second copy, different words)", project)

    # make the two rows exact duplicates: one hash (the write path would have merged them)
    old_hash = _rows(svc, uid)[old_id]["hash"]
    svc._memory.vector_store.client.set_payload(
        collection_name=unique_ns.qdrant_collection, payload={"hash": old_hash}, points=[new_id]
    )

    rows = _rows(svc, uid)
    assert rows[old_id]["hash"] == rows[new_id]["hash"]
    pool = _dream_pool_of(rows[old_id])
    assert pool and pool == _dream_pool_of(rows[new_id])
    redis_client.nsit_pools.append(pool)

    sweep_token = gate.acquire_lock(redis_client, pool)
    assert sweep_token, "the pool lock must be free before the test starts"

    held = svc.dedup_memories(uid, semantic=False)

    rows = _rows(svc, uid)
    assert old_id in rows and new_id in rows, held
    assert held["exact_duplicates_removed"] == 0 and held["exact_skipped_locked_pools"] == 1, held
    assert redis_client.get(f"dreaming:lock:{pool}").decode() == sweep_token, "the sweep's lock is untouched"

    gate.release_lock(redis_client, pool, sweep_token)

    done = svc.dedup_memories(uid, semantic=False)

    rows = _rows(svc, uid)
    assert done["exact_duplicates_removed"] == 1 and done["exact_skipped_locked_pools"] == 0, done
    assert old_id not in rows and new_id in rows  # the older copy goes, the newest stays
    assert redis_client.get(f"dreaming:lock:{pool}") is None, "dedup released the lock it took"


def test_lock_scripts_on_a_real_server(redis_client, unique_ns):
    from extensions.dreaming import gate

    pool = f"nsit-{unique_ns.user_id}"
    other = f"{pool}-other"
    redis_client.nsit_pools.extend([pool, other])
    key = f"dreaming:lock:{pool}"

    token = gate.acquire_lock(redis_client, pool, ttl_ms=60_000)
    assert token
    assert gate.acquire_lock(redis_client, pool) is None  # held

    assert gate.refresh_lock(redis_client, pool, token, ttl_ms=600_000) is True
    assert 60_000 < redis_client.pttl(key) <= 600_000  # renewed, not the original 60 s
    ttl_before = redis_client.pttl(key)
    assert gate.refresh_lock(redis_client, pool, "stranger", ttl_ms=5_000) is False
    assert redis_client.pttl(key) >= ttl_before - 2_000  # a stranger's refresh changed nothing
    assert gate.refresh_lock(redis_client, other, token, ttl_ms=5_000) is False  # token of another key

    gate.release_lock(redis_client, pool, "stranger")
    assert redis_client.get(key).decode() == token
    gate.release_lock(redis_client, pool, token)
    assert redis_client.get(key) is None
    assert gate.refresh_lock(redis_client, pool, token, ttl_ms=5_000) is False  # released: no longer ours
