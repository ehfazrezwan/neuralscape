"""Real-Qdrant qualification of reversible semantic dedup.

Real Qdrant + the real embedder. A fresh, unique user stores two near-duplicate
facts (the older one reinforced) and an unrelated fact; ``dedup_memories`` then
must:

1. tombstone the OLDER near-duplicate (the row stays on disk, flagged
   ``dream_tombstoned`` with ``superseded_by`` / ``superseded_reason`` /
   ``superseded_at``) and keep the newer one live;
2. hide the tombstone from recall and listings while the survivor is found;
3. transfer ``times_derived`` to the survivor;
4. revive the tombstone when its fact is derived again (``store_raw`` of the
   same text);
5. leave the unrelated fact untouched, and never touch a copy in another
   project (partition rule).

Both behaviours of ``DEDUP_SEMANTIC_TOMBSTONE_ENABLED`` are covered: ON (tombstone,
the first test) and OFF (hard delete, the last).

Uses the integration fixtures (``stores``, ``llm_key``, ``flags``, ``unique_ns``): all
rows live in a throwaway collection that is dropped afterwards, so nothing is
cleaned user-wide in a shared collection. On the gate (``NS_INTEGRATION_REQUIRED=1``)
missing stores or key fail the test. Needs Neo4j (the service initialises its graph
store), Qdrant, and GOOGLE_API_KEY (embeddings).
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import time
import uuid

import pytest


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns):
    """A MemoryService over a throwaway Qdrant collection (dropped at teardown).

    Paraphrase similarity varies with the embedding model; the test is about
    what happens to a pair that IS a near-duplicate, so the threshold is a
    loose 0.85 for this test only.
    """
    from qdrant_client import QdrantClient

    flags(
        qdrant_collection=unique_ns.qdrant_collection,
        qdrant_url=stores.qdrant_url,
        dedup_similarity_threshold=0.85,
    )
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()  # real mem0 over the throwaway collection; raises if unreachable
    return service


@pytest.fixture
def user(unique_ns):
    return unique_ns.user_id


def _store(svc, uid, text, project="proj-a"):
    out = svc.store_raw(
        content=text, user_id=uid, category="convention", scope="project",
        project_id=project, visibility="private", add_to_graph=False,
    )
    return out[0].id


def _row(svc, uid, mid):
    rows = {r["id"]: r for r in svc._scroll_all_user_memories(uid)}
    return rows[mid]["payload"]


def _meta(svc, uid, mid):
    return _row(svc, uid, mid).get("metadata") or {}


def _recalled(svc, uid, query, project="proj-a"):
    return {m.id for m in svc.search(query, user_id=uid, project_id=project, limit=20)}


def test_semantic_dedup_tombstones_reversibly_and_revives(flags, svc, user):
    flags(dedup_semantic_tombstone_enabled=True)  # opt-in behaviour: the flag-ON proof
    tag = uuid.uuid4().hex[:6]
    older = f"The deploy window for the zq{tag} billing service is Friday at 5pm."
    newer = f"The deploy window for the zq{tag} billing service is Friday 5pm."
    other = f"Quillfeather{tag} prefers lapsang souchong tea in the morning."

    old_id = _store(svc, user, older)
    time.sleep(1.1)  # distinct created_at
    new_id = _store(svc, user, newer)
    other_id = _store(svc, user, other)
    elsewhere_id = _store(svc, user, older, project="proj-b")  # same text, other partition
    assert len({old_id, new_id, other_id, elsewhere_id}) == 4, "setup must create four distinct rows"
    svc._bump_times_derived(old_id, 2)  # the older row was re-derived before

    result = svc.dedup_memories(user)

    # 1. tombstoned, not deleted
    assert result["semantic_duplicates_removed"] >= 1, result
    assert result["semantic_mode"] == "tombstone"
    meta = _meta(svc, user, old_id)  # still on disk
    assert meta["dream_tombstoned"] is True
    assert meta["superseded_by"] == new_id
    assert meta["superseded_reason"] == "semantic_dedup"
    assert meta["superseded_at"]
    assert _row(svc, user, old_id)["data"] == older
    assert not _meta(svc, user, new_id).get("dream_tombstoned")
    # untouched: the unrelated fact and the other-partition copy
    assert not _meta(svc, user, other_id).get("dream_tombstoned")
    assert not _meta(svc, user, elsewhere_id).get("dream_tombstoned")

    # 2. recall and listings hide the tombstone
    ids = _recalled(svc, user, f"deploy window zq{tag} billing")
    assert new_id in ids and old_id not in ids, ids
    listed = {m.id for m in svc.list_memories(user_id=user, project_id="proj-a", limit=100)}
    assert new_id in listed and old_id not in listed
    assert old_id in {m.id for m in svc.list_memories(user_id=user, project_id="proj-a", limit=100, include_tombstoned=True)}

    # 3. reinforcement moved to the survivor (1 own + 2 bumped + 1 old = at least 3)
    assert int(_meta(svc, user, new_id).get("times_derived", 1)) >= 3

    # a second run does not re-merge or delete anything
    again = svc.dedup_memories(user)
    assert again["semantic_duplicates_removed"] == 0
    assert {r["id"] for r in svc._scroll_all_user_memories(user)} >= {old_id, new_id, other_id, elsewhere_id}

    # 4. deriving the older fact again revives it
    _store(svc, user, older)
    assert _meta(svc, user, old_id).get("dream_tombstoned") is False
    assert old_id in _recalled(svc, user, f"deploy window zq{tag} billing")


def test_semantic_dedup_flag_off_hard_deletes_the_older_row(flags, svc, user):
    """Contrast: with the flag off (the default) the older near-duplicate is deleted."""
    flags(dedup_semantic_tombstone_enabled=False)
    tag = uuid.uuid4().hex[:6]
    older = f"The deploy window for the zq{tag} billing service is Friday at 5pm."
    newer = f"The deploy window for the zq{tag} billing service is Friday 5pm."
    old_id = _store(svc, user, older)
    time.sleep(1.1)
    new_id = _store(svc, user, newer)

    result = svc.dedup_memories(user)

    assert result["semantic_mode"] == "delete"
    assert result["semantic_duplicates_removed"] >= 1, result
    ids = {r["id"] for r in svc._scroll_all_user_memories(user)}
    assert old_id not in ids and new_id in ids


def test_dedup_summary_counts_declines_and_names_the_merged_pair(flags, svc, user, caplog):
    """The run reports every decline reason (zero here) and logs the merged pair's ids."""
    import logging

    flags(dedup_semantic_tombstone_enabled=True)
    tag = uuid.uuid4().hex[:6]
    older = f"The deploy window for the zq{tag} billing service is Friday at 5pm."
    newer = f"The deploy window for the zq{tag} billing service is Friday 5pm."
    old_id = _store(svc, user, older)
    time.sleep(1.1)
    new_id = _store(svc, user, newer)

    with caplog.at_level(logging.INFO):
        result = svc.dedup_memories(user)

    assert result["semantic_duplicates_removed"] >= 1, result
    declined = result["semantic_declined"]
    assert declined["search_failed"] == 0 and declined["merge_failed"] == 0, declined
    assert f"older={old_id} newer={new_id}" in caplog.text
