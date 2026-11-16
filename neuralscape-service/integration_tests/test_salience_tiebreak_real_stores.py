"""Flag-on proof for the recall salience tie-breaker (``DREAMING_SALIENCE_RECALL_K``).

Atlas node ``read.salience-tiebreak``. Real Qdrant (a throwaway collection),
real Redis (the dreaming recall traces) and the real embedder; no mocks.

What it shows, through the service's own write and read paths:

1. Two memories that are a near tie for a query are stored with ``store_raw``.
   With ``salience_recall_k = 0`` (the default) the order is the native one,
   a search never touches Redis and never logs a recall trace.
2. With ``salience_recall_k > 0`` (and dreaming itself still off, so the
   setting alone is what turns trace logging on) the memory that LOST the
   native tie is recalled repeatedly through ``MemoryService.search``. Its
   recall traces and salience-dynamics state accumulate in Redis; its twin has
   none.
3. A fresh search now ranks it above its twin, with its score lifted by the
   bounded ``1 + k*log1p(signal)`` factor, and the twin's score unchanged.
4. Back at ``k = 0`` the order returns to the native one even though Redis
   still holds the strength state, and again Redis is never touched.

It fails when the tie-break is broken: if ``k`` is ignored the order never
flips (step 3), if traces are not logged the strength signal never appears
(step 2), and if ``k = 0`` still reads Redis the spy in step 1/4 trips.

Needs Neo4j (the service initialises its graph store), Qdrant, Redis and
``GOOGLE_API_KEY`` (embeddings). On the gate (``NS_INTEGRATION_REQUIRED=1``)
missing infrastructure fails the test instead of skipping it.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import math
import uuid

import pytest

# Large step so a handful of distinct recalls saturate the strength cap
# (default 5.0): signal = cap - 1 = 4, lift = 1 + 0.1 * log1p(4) ~= 1.161.
STRENGTH_DELTA = 1.0
K = 0.1  # the hard ceiling the setting allows
TWIN_CATEGORIES = ("convention", "architecture")  # distinct so a category filter can single one out


@pytest.fixture
def traces(stores, monkeypatch, unique_ns):
    """The dreaming traces module, bound to the test Redis, with call spies.

    ``calls["redis"]`` counts ``_get_redis`` (every read or write of the trace
    store goes through it); ``calls["submit"]`` counts trace writes handed to
    the daemon thread. ``flush()`` waits for the single trace worker thread.
    """
    import redis as redis_lib

    from extensions.dreaming import traces as tr

    client = redis_lib.Redis.from_url(stores.redis_url, socket_timeout=5, socket_connect_timeout=5)
    monkeypatch.setattr(tr, "_redis_client", client)

    calls = {"redis": 0, "submit": 0}
    real_get_redis = tr._get_redis
    real_submit = tr._executor.submit

    def counted_get_redis():
        calls["redis"] += 1
        return real_get_redis()

    def counted_submit(*args, **kwargs):
        calls["submit"] += 1
        return real_submit(*args, **kwargs)

    monkeypatch.setattr(tr, "_get_redis", counted_get_redis)
    monkeypatch.setattr(tr._executor, "submit", counted_submit)

    touched: list[str] = []

    def cleanup():
        # No daemon-thread trace may land after the deletes below.
        real_submit(lambda: None).result(timeout=20)
        if not touched:
            return
        pipe = client.pipeline(transaction=False)
        for mid in touched:
            pipe.hdel(tr._COUNT_KEY, mid)
            pipe.hdel(tr._LAST_KEY, mid)
            pipe.hdel(tr._DYN_KEY, mid)
            pipe.delete(tr._QUERY_KEY.format(mid=mid))
        pipe.execute()

    unique_ns.on_cleanup(cleanup)

    class Handle:
        module = tr
        redis = client
        counts = calls

        @staticmethod
        def track(*memory_ids):
            touched.extend(memory_ids)

        @staticmethod
        def flush():
            # One worker thread, FIFO: when this no-op ran, every earlier trace did.
            real_submit(lambda: None).result(timeout=20)

    return Handle


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns):
    """A MemoryService over a throwaway Qdrant collection (dropped at teardown)."""
    from qdrant_client import QdrantClient

    flags(qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url, redis_url=stores.redis_url)
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()  # real mem0 over the throwaway collection; raises if unreachable
    return service


@pytest.fixture
def dreaming(monkeypatch):
    """The dreaming settings singleton at its defaults, mutable per test.

    ``dreaming_settings`` is its own object (env prefix ``DREAMING_``), not a
    field of ``config.settings``, so the ``flags`` fixture does not reach it.
    ``enabled`` stays False throughout: the tie-breaker setting alone must be
    enough to log traces and to read them back.
    """
    from extensions.dreaming.config import dreaming_settings

    monkeypatch.setattr(dreaming_settings, "enabled", False)
    monkeypatch.setattr(dreaming_settings, "dynamics_enabled", True)
    monkeypatch.setattr(dreaming_settings, "dynamics_strength_delta", STRENGTH_DELTA)
    monkeypatch.setattr(dreaming_settings, "salience_recall_k", 0.0)
    return dreaming_settings


def _order(results, *ids):
    """The given ids in the order the search returned them (others ignored)."""
    wanted = set(ids)
    return [r.id for r in results if r.id in wanted]


def test_salience_tiebreak_flips_a_near_tie_with_the_flag_on_and_is_inert_at_zero(svc, traces, dreaming, unique_ns):
    user, project = unique_ns.user_id, unique_ns.project_id
    tag = uuid.uuid4().hex[:6]
    texts = (
        f"The zq{tag} ingest service retries a failed upload three times with exponential backoff (first variant).",
        f"The zq{tag} ingest service retries a failed upload three times with exponential backoff (second variant).",
    )

    # Real write path; the twins live in two categories so a category filter
    # can return exactly one of them.
    ids = []
    for text, category in zip(texts, TWIN_CATEGORIES):
        out = svc.store_raw(
            content=text, user_id=user, category=category, scope="project",
            project_id=project, visibility="private", add_to_graph=False,
        )
        ids.append(out[0].id)
    assert len(set(ids)) == 2, "setup must create two distinct rows"
    traces.track(*ids)

    final_query = f"how does the zq{tag} ingest service retry a failed upload"

    # ── 1. k = 0 (default): native order, Redis never touched ──────────────
    assert dreaming.salience_recall_k == 0.0
    native = svc.search(final_query, user_id=user, project_id=project, limit=10)
    native_order = _order(native, *ids)
    assert set(native_order) == set(ids), f"both twins must be recalled: {[r.id for r in native]}"
    winner, loser = native_order
    native_score = {r.id: r.score for r in native if r.id in ids}
    traces.flush()
    assert traces.counts == {"redis": 0, "submit": 0}, (
        "salience_recall_k=0 must leave the search path Redis-free and trace-free, "
        f"got {traces.counts}"
    )
    assert traces.redis.hget(traces.module._COUNT_KEY, loser) is None
    # No score-gap precondition: the hybrid sort uses fused order keys, not the
    # displayed score. The twins differ by one word, and step 3's flip (which can
    # only happen when the boost beats the native gap) is the proof.

    # ── 2. k > 0: recall the loser repeatedly; its traces accumulate ───────
    dreaming.salience_recall_k = K
    queries = [
        f"zq{tag} ingest upload retry backoff",
        f"what happens when a zq{tag} upload fails",
        f"zq{tag} ingest service exponential backoff policy",
        f"how many times does zq{tag} retry an upload",
        f"zq{tag} failed upload handling",
        f"retry behaviour of the zq{tag} ingest service",
    ]
    for query in queries:
        only_loser = svc.search(
            query, user_id=user, project_id=project,
            categories=[TWIN_CATEGORIES[ids.index(loser)]], limit=5,
        )
        assert [r.id for r in only_loser] == [loser], "the category filter must return the loser alone"
    traces.flush()

    agg = traces.module.read_aggregates(traces.redis, [winner, loser])
    assert agg[loser]["recall_count"] == len(queries), agg
    assert agg[loser]["unique_query_count"] == len(queries), agg
    assert agg[winner]["recall_count"] == 0, "the twin was never recalled, it must have no trace"
    signals = traces.module.get_strength_signals([winner, loser])
    assert signals.get(loser, 0.0) > 0.0, "recalls with k>0 (dreaming off) must record salience dynamics"
    assert signals.get(winner, 0.0) == 0.0

    # ── 3. a fresh search ranks the recalled twin first ────────────────────
    boosted = svc.search(final_query, user_id=user, project_id=project, limit=10)
    assert _order(boosted, *ids) == [loser, winner], (
        f"native order was {native_order}; with k={K} and recall history the "
        f"recalled twin must overtake its near-tied twin, got {_order(boosted, *ids)}"
    )
    score = {r.id: r.score for r in boosted if r.id in ids}
    lift = 1.0 + K * math.log1p(signals[loser])
    assert score[loser] == pytest.approx(native_score[loser] * lift, rel=1e-3)
    assert score[winner] == pytest.approx(native_score[winner], rel=1e-3), "an unrecalled memory is never boosted"
    assert lift <= 1.0 + K * math.log1p(4.0) + 1e-9, "the boost stays bounded by the strength cap"
    traces.flush()

    # ── 4. back to k = 0: native order again, Redis untouched ──────────────
    dreaming.salience_recall_k = 0.0
    traces.counts.update(redis=0, submit=0)
    again = svc.search(final_query, user_id=user, project_id=project, limit=10)
    assert _order(again, *ids) == [winner, loser], "with k=0 the recall history must not change the order"
    traces.flush()
    assert traces.counts == {"redis": 0, "submit": 0}, (
        f"salience_recall_k=0 touched the trace store again: {traces.counts}"
    )
    # ... even though the strength state is still sitting in Redis.
    assert traces.redis.hget(traces.module._DYN_KEY, loser) is not None
