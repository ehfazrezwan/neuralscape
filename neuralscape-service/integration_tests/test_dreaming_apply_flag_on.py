"""Dreaming deep-consolidate apply primitives, flag ON, real Qdrant/Redis/embedder.

No consolidation LLM is involved: the actions are handed to
``consolidate.apply_actions`` directly, so these assertions are exact. The
real Gemini embedder is used (rows are stored through ``store_raw``).

Run (from neuralscape-service/):
    uv run pytest integration_tests/test_dreaming_apply_flag_on.py -q
Env: NEO4J_URI NEO4J_USER NEO4J_PASSWORD QDRANT_URL REDIS_URL GOOGLE_API_KEY
(NS_INTEGRATION_REQUIRED=1 makes a missing store/key a failure.)
"""

from __future__ import annotations

import uuid

import pytest

from extensions.dreaming import consolidate
from integration_tests._dreaming_support import dream_env  # noqa: F401  (fixture)

pytestmark = pytest.mark.integration


def _tok() -> str:
    # one rare lexical token per test: BM25-only hits can't come from anywhere else
    return "zq" + uuid.uuid4().hex[:10]


def _named_vectors(env, mid: str) -> dict:
    pt = env.point(mid, vectors=True)
    assert pt is not None, f"{mid} missing"
    vec = pt.vector
    assert isinstance(vec, dict), (
        "point stores a bare dense vector (no named vectors): BM25 slot absent "
        f"from the throwaway collection? got {type(vec).__name__}"
    )
    return vec


@pytest.mark.asyncio
async def test_merge_keeps_bm25_vector_and_provenance(dream_env):
    env = dream_env
    tok = _tok()
    old_tok = _tok()
    keep = env.store(f"User prefers Python for backend services {old_tok}")
    gone = env.store("The user likes Python for server-side work")
    merged = f"User prefers Python for backend services, especially with {tok} tooling"
    assert env.lexical_ids(tok) == [], "control: token must not match before the rewrite"
    assert keep in env.lexical_ids(old_tok), "control: old token matches before the rewrite"
    before = _named_vectors(env, keep)
    assert before.get("bm25") is not None, "write path must store a bm25 sparse vector"

    result = await consolidate.apply_actions(
        env.service, env.batch([keep, gone]),
        [{"type": "merge", "memory_ids": [keep, gone], "survivor_id": keep,
          "content": merged, "confidence": 0.9}],
        dry_run=False,
    )
    assert not result.errors, result.errors

    # survivor: rewritten, reversible, hash follows content, provenance + counter
    from memory_service import content_hash

    p = env.payload(keep)
    meta = p["metadata"]
    assert p["data"] == merged
    assert p["hash"] == content_hash(merged)
    assert meta["dream_prev_content"] == f"User prefers Python for backend services {old_tok}"
    assert gone in meta["derived_from"]
    assert meta["times_derived"] >= 2
    assert not meta.get("dream_tombstoned")
    # loser: tombstoned, pointing at the survivor
    lmeta = env.meta(gone)
    assert lmeta["dream_tombstoned"] is True and lmeta["superseded_by"] == keep

    # THE regression: the rewritten row still has dense + bm25 vectors ...
    after = _named_vectors(env, keep)
    assert len(after[""]) == len(before[""]), "dense vector lost or resized"
    assert after.get("bm25") is not None and after["bm25"].indices, "bm25 sparse vector dropped"
    # ... and a keyword-only (BM25) query finds it by the NEW text
    assert keep in env.lexical_ids(tok), "keyword-only query must find the rewritten row"
    assert keep not in env.lexical_ids(old_tok), "words only in the OLD text must stop matching"
    # recall excludes the tombstoned loser, includes the survivor
    ids = {h.id for h in env.service.search(query="Python backend services", user_id=env.user, limit=10)}
    assert gone not in ids
    assert keep in ids


@pytest.mark.asyncio
async def test_rewrite_and_reframe_keep_bm25_and_are_reversible(dream_env):
    env = dream_env
    t1, t2, o1 = _tok(), _tok(), _tok()
    rw = env.store(f"User works at OldCorp {o1} on the platform team", "personal_fact")
    rf = env.store("User plans to ship the beta in early July 2026", "task_context")
    old_rf = env.payload(rf)["data"]

    result = await consolidate.apply_actions(
        env.service, env.batch([rw, rf]),
        [{"type": "rewrite", "memory_ids": [rw], "confidence": 0.9,
          "content": f"User works at NewCorp ({t1}) on the platform team"},
         {"type": "temporal_reframe", "memory_ids": [rf], "confidence": 0.9,
          "content": f"User planned to ship the beta in early July 2026 ({t2}); date has passed"}],
        dry_run=False,
    )
    assert not result.errors, result.errors

    assert env.payload(rw)["data"].startswith("User works at NewCorp")
    assert env.meta(rw)["dream_prev_content"] == f"User works at OldCorp {o1} on the platform team"
    assert rw not in env.lexical_ids(o1), "old-text keyword still matches the rewritten row"
    assert env.meta(rf)["dream_temporal_reframed"] is True
    assert env.meta(rf)["dream_prev_content"] == old_rf
    for mid, tok in ((rw, t1), (rf, t2)):
        assert _named_vectors(env, mid).get("bm25") is not None, f"{mid} lost bm25"
        assert mid in env.lexical_ids(tok), f"keyword-only query misses rewritten {mid}"


@pytest.mark.asyncio
async def test_invalidate_and_prune_are_tombstones_excluded_from_recall(dream_env):
    env = dream_env
    tok = _tok()
    old = env.store(f"The demo service runs on a single node only {tok}", "decision")
    new = env.store("The demo service now runs on a managed cluster", "decision")
    noise = env.store(f"ok got it {tok}", "task_context")

    result = await consolidate.apply_actions(
        env.service, env.batch([old, new, noise]),
        [{"type": "invalidate", "memory_ids": [old], "superseded_by_id": new, "confidence": 0.95},
         {"type": "prune", "memory_ids": [noise], "confidence": 0.95}],
        dry_run=False,
    )
    assert not result.errors, result.errors
    om = env.meta(old)
    assert om["dream_tombstoned"] is True and om["superseded_by"] == new and om["superseded_at"]
    nm = env.meta(noise)
    assert nm["dream_tombstoned"] is True and nm["dream_pruned"] is True
    # rows are retained (reversible), recall excludes them
    assert env.payload(old) and env.payload(noise)
    got = {h.id for h in env.service.search(query=f"demo service node {tok}", user_id=env.user, limit=10)}
    assert old not in got and noise not in got
    assert new in got


@pytest.mark.asyncio
async def test_secret_prune_gate_hard_deletes_only_above_confidence(dream_env):
    env = dream_env
    sure = env.store("The staging API key is sk-e2e-FAKE-abc123456789 for testing", "task_context")
    unsure = env.store("maybe a credential: hunter2-fake-value in the notes", "task_context")
    actions = [
        {"type": "prune", "memory_ids": [sure], "confidence": 0.99, "contains_secret": True},
        {"type": "prune", "memory_ids": [unsure], "confidence": 0.2, "contains_secret": True},
    ]
    to_apply, to_report = consolidate.split_by_posture(actions, auto_apply_confidence=0.85)
    assert len(to_apply) == 2 and not to_report
    result = await consolidate.apply_actions(
        env.service, env.batch([sure, unsure]), to_apply, dry_run=False
    )
    assert not result.errors, result.errors
    assert env.payload(sure) == {}, "gate-cleared secret must be hard-deleted from Qdrant"
    um = env.meta(unsure)  # below-gate claim: reversible tombstone + recorded claim
    assert um["dream_tombstoned"] is True
    assert um["dream_secret_claim"]["contains_secret"] is True


@pytest.mark.asyncio
async def test_dry_run_writes_nothing(dream_env):
    env = dream_env
    a = env.store("User drinks green tea in the morning")
    before = env.payload(a)
    result = await consolidate.apply_actions(
        env.service, env.batch([a]),
        [{"type": "rewrite", "memory_ids": [a], "content": "changed", "confidence": 0.9},
         {"type": "prune", "memory_ids": [a], "confidence": 0.99}],
        dry_run=True,
    )
    assert len(result.applied) == 2 and all(x.get("dry_run") for x in result.applied)
    assert env.payload(a) == before


def test_rewrite_does_not_lose_a_concurrent_times_derived_bump(dream_env):
    """Real Qdrant: a writer bumps times_derived between the rewrite's read and
    its write. The nested-key patch must leave the bump intact."""
    env = dream_env
    mid = env.store("User keeps notes in a plain markdown vault")
    client = env.service._memory.vector_store.client
    real_retrieve = client.retrieve
    state = {"done": False}

    def racing_retrieve(*a, **kw):
        out = real_retrieve(*a, **kw)
        if not state["done"]:
            state["done"] = True
            client.set_payload(
                collection_name=env.collection, payload={"times_derived": 9},
                points=[mid], key="metadata",
            )
        return out

    client.retrieve = racing_retrieve
    try:
        consolidate._rewrite_content(env.service, mid, "User keeps notes in a markdown vault (rewritten)")
    finally:
        client.retrieve = real_retrieve
    assert state["done"]
    meta = env.meta(mid)
    assert meta["times_derived"] == 9, "concurrent bump was clobbered by the rewrite"
    assert meta["dream_prev_content"] == "User keeps notes in a plain markdown vault"
    assert env.payload(mid)["data"].endswith("(rewritten)")
