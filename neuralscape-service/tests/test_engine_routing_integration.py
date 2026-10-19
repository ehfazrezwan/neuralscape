"""Engine switch against REAL Qdrant and pgvector: generation 1 (Qdrant) -> dual write -> catch-up ->
flip (generation 2, pgvector) -> writes continue -> rollback (generation 3, Qdrant) -> roll forward again,
with exact-state parity asserted at every step.

Skipped unless ENGINE_TEST_QDRANT_URL and ENGINE_TEST_PG_DSN point at disposable services.  Data is
synthetic; embeddings are a deterministic hash.  Retrieval ranking is not compared here (see the
pgvector parity test); this test is about exact state: ids, revisions, text, labels, deletes and fences.
"""

import os
import json
import random
import threading
import uuid

import pytest

QDRANT_URL = os.environ.get("ENGINE_TEST_QDRANT_URL")
PG_DSN = os.environ.get("ENGINE_TEST_PG_DSN")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (QDRANT_URL and PG_DSN), reason="ENGINE_TEST_QDRANT_URL and ENGINE_TEST_PG_DSN are not set"),
]

from contracts_errors import InternalErrorCode  # noqa: E402
from engine_adapters_current import QdrantVectorAdapter, hashed_bag_embedder  # noqa: E402
from engine_adapters_pgvector import PgVectorAdapter  # noqa: E402
from engine_ports import EngineRecord, OperationContext, PortFailure, TenantFilter  # noqa: E402
from engine_routing import Backend, CanonicalState, EngineRouter, RoutingError  # noqa: E402

DIMS = 64
embed = hashed_bag_embedder(DIMS)
TENANTS = ("tenant-a", "tenant-b")


def ctx(tenant):
    return OperationContext(tenant_id=tenant, correlation_id="c", deadline_ms=10_000)  # the adapters honor it


class Canonical:
    """The coordinator's source of truth; writes are mirrored here only when accepted."""

    def __init__(self):
        self.rows = {}  # (tenant, id) -> dict(revision, removed, text, labels)
        self.lock = threading.Lock()

    def accepted_apply(self, r):
        with self.lock:
            self.rows[(r.tenant_id, r.record_id)] = dict(revision=r.revision, removed=False, text=r.text, labels=tuple(r.labels))

    def accepted_remove(self, tenant, rid, revision):
        with self.lock:
            self.rows[(tenant, rid)] = dict(revision=revision, removed=True, text="", labels=())

    def states(self):
        with self.lock:
            snapshot = dict(self.rows)
        return iter(CanonicalState(t, i, v["revision"], v["removed"], v["text"], v["labels"]) for (t, i), v in snapshot.items())

    def expected(self):
        return {k: (None if v["removed"] else (v["revision"], v["text"], v["labels"])) for k, v in self.rows.items()}


class Driver:
    """Issues writes through the router, mirroring accepted ones into the canonical store."""

    def __init__(self, router, canonical):
        self.router, self.canonical = router, canonical
        self.outcomes = []

    def apply(self, tenant, rid, revision, text, labels=()):
        record = EngineRecord(record_id=rid, revision=revision, tenant_id=tenant, text=text, labels=tuple(labels))
        try:
            self.router.apply(self.router.route(), ctx(tenant), record)
            self.canonical.accepted_apply(record)
            self.outcomes.append("ok")
        except PortFailure as failure:
            self.outcomes.append(failure.code.value)

    def remove(self, tenant, rid, revision):
        try:
            self.router.remove(self.router.route(), ctx(tenant), rid, revision)
            self.canonical.accepted_remove(tenant, rid, revision)
            self.outcomes.append("ok")
        except PortFailure as failure:
            self.outcomes.append(failure.code.value)


def observed(adapter, canonical):
    out = {}
    for (tenant, rid) in canonical.rows:
        row = adapter.get(ctx(tenant), rid)
        out[(tenant, rid)] = None if row is None else (row.revision, row.text, row.labels)
    return out


def fence_probes(adapter, canonical):
    """Every removed id must refuse an older revision (the tombstone fence), on this store."""
    refused = 0
    for (tenant, rid), row in canonical.rows.items():
        if row["removed"] and row["revision"] >= 2:
            with pytest.raises(PortFailure) as raised:
                adapter.apply(ctx(tenant), EngineRecord(record_id=rid, revision=row["revision"] - 1, tenant_id=tenant, text="older"))
            assert raised.value.code is InternalErrorCode.STALE_REVISION, (tenant, rid)
            refused += 1
    return refused


@pytest.fixture
def stack():
    from mem0.vector_stores.qdrant import Qdrant

    store = Qdrant(collection_name="ns_route_" + uuid.uuid4().hex[:10], embedding_model_dims=DIMS, url=QDRANT_URL)
    qdrant = QdrantVectorAdapter(store, embed, implementation_id="qdrant")
    schema = "ns_route_" + uuid.uuid4().hex[:10]
    pg = PgVectorAdapter(PG_DSN, embed=embed, dimensions=DIMS, schema=schema, implementation_id="pgvector")
    pg.ensure_schema(create_extension=True)
    yield qdrant, pg
    try:
        store.client.delete_collection(store.collection_name)
    except Exception:
        pass
    import psycopg

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def seed(driver):
    rnd = random.Random(5)
    words = [f"w{i}" for i in range(40)]
    for tenant in TENANTS:
        for i in range(20):
            driver.apply(tenant, f"r{i}", 1, " ".join(rnd.choices(words, k=6)), ("x",) if i % 3 == 0 else ())
    for i in range(0, 20, 4):
        driver.apply("tenant-a", f"r{i}", 2, f"updated {i} " + " ".join(rnd.choices(words, k=3)))
    for i in range(1, 20, 5):
        driver.remove("tenant-a", f"r{i}", 3)
    driver.apply("tenant-a", "r0", 1, "stale write")                 # refused: stale
    driver.apply("tenant-b", "r1", 1, "different payload")           # refused: same-revision conflict


def test_engine_switch_with_exact_state_parity_at_every_step(stack):
    qdrant, pg = stack
    router = EngineRouter(Backend("qdrant", qdrant))
    canonical = Canonical()
    driver = Driver(router, canonical)
    assert router.generation == 1

    # generation 1: corpus through Qdrant only
    seed(driver)
    assert "stale_revision" in driver.outcomes and "idempotency_conflict" in driver.outcomes
    assert observed(qdrant, canonical) == canonical.expected()
    assert observed(pg, canonical) != canonical.expected()          # pgvector knows nothing yet

    # dual-write window: new traffic reaches both, old state does not
    router.begin_dual_write(Backend("pgvector", pg))
    driver.apply("tenant-a", "r30", 1, "written during the window", ("x",))
    driver.apply("tenant-b", "r2", 2, "updated during the window")
    driver.remove("tenant-b", "r3", 2)
    assert pg.get(ctx("tenant-a"), "r30") is not None and pg.get(ctx("tenant-a"), "r2") is None
    assert router.lag() > 0
    with pytest.raises(RoutingError, match="lagging"):
        router.flip()

    # catch-up from canonical: exact-state parity before the flip
    report = router.catch_up(canonical)
    assert not report.conflicts and router.lag() == 0
    assert observed(qdrant, canonical) == canonical.expected() == observed(pg, canonical)
    assert fence_probes(qdrant, canonical) == fence_probes(pg, canonical) > 3

    # flip to generation 2 (pgvector); a token routed under generation 1 is refused
    old_token = router.route()
    assert router.flip(parity_check=lambda old, new: observed(old, canonical) == observed(new, canonical)) == 2
    with pytest.raises(PortFailure) as stale:
        router.apply(old_token, ctx("tenant-a"), EngineRecord(record_id="r40", revision=1, tenant_id="tenant-a", text="late"))
    assert stale.value.code is InternalErrorCode.STALE_REVISION
    assert qdrant.get(ctx("tenant-a"), "r40") is None and pg.get(ctx("tenant-a"), "r40") is None

    # writes continue on generation 2
    driver.apply("tenant-a", "r5", 2, "post-switch update", ("y",))
    driver.apply("tenant-a", "r41", 1, "post-switch create")
    driver.remove("tenant-b", "r6", 2)
    driver.apply("tenant-b", "r6", 1, "resurrection attempt")        # refused: stale
    assert observed(pg, canonical) == canonical.expected()
    assert observed(qdrant, canonical) != canonical.expected()      # the old store is behind, as designed
    assert qdrant.get(ctx("tenant-a"), "r41") is None
    assert router.active_label == "pgvector"

    # rollback to generation 3 (Qdrant): post-switch writes are replayed, fences included
    assert router.rollback(canonical) == 3
    assert router.active_label == "qdrant" and router.previous_label == "pgvector"
    assert observed(qdrant, canonical) == canonical.expected() == observed(pg, canonical)
    assert fence_probes(qdrant, canonical) == fence_probes(pg, canonical)
    with pytest.raises(PortFailure) as stale_again:
        router.apply(router.route()._replace(generation=2), ctx("tenant-a"), EngineRecord(record_id="r42", revision=1, tenant_id="tenant-a", text="x"))
    assert stale_again.value.code is InternalErrorCode.STALE_REVISION

    # traffic on generation 3, then roll forward again (generation 4, pgvector)
    driver.apply("tenant-a", "r43", 1, "written while rolled back")
    driver.apply("tenant-a", "r43", 2, "updated while rolled back")
    assert router.rollback(canonical) == 4
    assert router.active_label == "pgvector"
    assert observed(pg, canonical) == canonical.expected() == observed(qdrant, canonical)

    # retrieval after the switches honors tenant and authorization filters on the serving store
    live = {rid for (t, rid), v in canonical.expected().items() if t == "tenant-a" and v is not None}
    allowed = tuple(sorted(live)[::2])
    hits = router.port().search(ctx("tenant-a"), "w1 w2 updated", TenantFilter(tenant_id="tenant-a", authorized_record_ids=allowed), 6)
    assert hits and {h.record_id for h in hits} <= set(allowed)
    print("outcomes:", {o: driver.outcomes.count(o) for o in set(driver.outcomes)}, "generation:", router.generation)


def test_writers_racing_the_flip_on_real_stores_lose_nothing(stack):
    qdrant, pg = stack
    for _ in range(3):
        router = EngineRouter(Backend("qdrant", qdrant))
        canonical = Canonical()
        driver = Driver(router, canonical)
        driver.apply("tenant-a", "seed", 1, "seed text")
        router.begin_dual_write(Backend("pgvector", pg))
        router.catch_up(canonical)
        barrier = threading.Barrier(7)
        refused = []

        def writer(i):
            token = router.route()
            barrier.wait()
            record = EngineRecord(record_id=f"race-{uuid.uuid4().hex[:6]}-{i}", revision=1, tenant_id="tenant-a", text=f"race {i}")
            try:
                router.apply(token, ctx("tenant-a"), record)
                canonical.accepted_apply(record)
            except PortFailure as failure:
                assert failure.code is InternalErrorCode.STALE_REVISION
                refused.append(record.record_id)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(6)]
        [t.start() for t in threads]
        barrier.wait()
        router.flip()
        [t.join() for t in threads]
        assert observed(pg, canonical) == canonical.expected()          # every accepted write is on the new store
        for rid in refused:
            assert pg.get(ctx("tenant-a"), rid) is None and qdrant.get(ctx("tenant-a"), rid) is None
        router.rollback(canonical)
        assert observed(qdrant, canonical) == canonical.expected() == observed(pg, canonical)
        # fresh adapters for the next round
        qdrant._store.client.delete_collection(qdrant._store.collection_name)
        qdrant._store.create_col(DIMS, False)
        import psycopg

        with psycopg.connect(PG_DSN, autocommit=True) as conn:
            conn.execute(f"TRUNCATE {pg._schema}.records")


# ---- erasure reaches every retained generation and release destroys it (raw store scans) ----


MARKER = "ERASEME-SECRET-PHRASE"


def raw_qdrant_text(adapter) -> str:
    client, name = adapter._store.client, adapter._store.collection_name
    try:
        points, _ = client.scroll(name, limit=1000, with_payload=True, with_vectors=False)
    except Exception:
        return ""  # the collection is gone
    return json.dumps([p.payload for p in points], default=str)


def raw_pg_text(adapter) -> str:
    import psycopg

    try:
        with psycopg.connect(PG_DSN) as conn:
            rows = conn.execute(f"SELECT record_id, revision, removed, body, labels FROM {adapter._schema}.records").fetchall()
    except Exception:
        return ""  # the schema is gone
    return json.dumps(rows, default=str)


def _flip_with_marker(qdrant, pg):
    from engine_routing import InMemoryJournal

    journal = InMemoryJournal()
    router = EngineRouter(Backend("qdrant", qdrant), journal=journal)
    canonical = Canonical()
    driver = Driver(router, canonical)
    driver.apply("tenant-a", "victim", 1, f"{MARKER} private note", ("x",))
    driver.apply("tenant-a", "keeper", 1, "an unrelated note")
    router.begin_dual_write(Backend("pgvector", pg))
    router.catch_up(canonical)
    router.flip()
    return router, canonical, driver, journal


def test_erased_text_is_absent_from_every_generation_and_the_journal_after_a_flip(stack):
    qdrant, pg = stack
    router, canonical, driver, journal = _flip_with_marker(qdrant, pg)
    assert MARKER in raw_qdrant_text(qdrant) and MARKER in raw_pg_text(pg)  # both generations hold it before the erasure
    driver.remove("tenant-a", "victim", 2)  # erasure after the switch
    assert MARKER not in raw_pg_text(pg)  # the active generation
    assert MARKER not in raw_qdrant_text(qdrant)  # the previous generation (this was the retained copy)
    assert MARKER not in repr(journal.entries_after(0))
    assert router.previous_purge_pending == 0
    assert "an unrelated note" in raw_pg_text(pg) and "an unrelated note" in raw_qdrant_text(qdrant)  # nothing else was touched
    fence_probes(qdrant, canonical); fence_probes(pg, canonical)  # the tombstone fences hold on both


def test_an_erasure_the_previous_generation_missed_is_flagged_and_repaired_by_rollback(stack):
    qdrant, pg = stack
    router, canonical, driver, journal = _flip_with_marker(qdrant, pg)
    real_remove = qdrant.remove
    qdrant.remove = lambda c, rid, rev: (_ for _ in ()).throw(RuntimeError("previous generation unreachable"))
    driver.remove("tenant-a", "victim", 2)  # the caller is not failed
    assert router.previous_purge_pending == 1 and MARKER in raw_qdrant_text(qdrant)  # visible, not silent
    with pytest.raises(RoutingError, match="unpurged"):
        router.release_rollback_target()
    qdrant.remove = real_remove
    router.rollback(canonical)  # repaired from canonical state
    assert MARKER not in raw_qdrant_text(qdrant) and MARKER not in raw_pg_text(pg)


def test_release_destroys_the_previous_generation_on_a_real_store_and_records_it(stack):
    qdrant, pg = stack
    router, canonical, driver, journal = _flip_with_marker(qdrant, pg)
    audit = router.release_rollback_target()
    assert audit["destroyed"] is True and audit["label"] == "qdrant"
    with pytest.raises(Exception):
        qdrant._store.client.get_collection(qdrant._store.collection_name)  # the old generation no longer exists
    assert MARKER in raw_pg_text(pg)  # the serving generation is untouched
    assert router.release_log == [audit] and router.previous_label is None


def test_a_rolled_back_generation_can_be_destroyed_too(stack):
    import psycopg

    qdrant, pg = stack
    router, canonical, driver, journal = _flip_with_marker(qdrant, pg)
    router.rollback(canonical)  # qdrant serves again; pgvector is the rollback target
    audit = router.release_rollback_target()
    assert audit["label"] == "pgvector" and audit["destroyed"] is True
    with psycopg.connect(PG_DSN) as conn:
        found = conn.execute("SELECT count(*) FROM information_schema.schemata WHERE schema_name = %s", (pg._schema,)).fetchone()[0]
    assert found == 0 and MARKER in raw_qdrant_text(qdrant)


def test_a_second_flip_destroys_the_generation_it_would_orphan(stack):
    from mem0.vector_stores.qdrant import Qdrant

    qdrant, pg = stack
    router, canonical, driver, journal = _flip_with_marker(qdrant, pg)
    third_store = Qdrant(collection_name="ns_route_" + uuid.uuid4().hex[:10], embedding_model_dims=DIMS, url=QDRANT_URL)
    third = QdrantVectorAdapter(third_store, embed, implementation_id="qdrant-3")
    try:
        router.begin_dual_write(Backend("qdrant-3", third))
        router.catch_up(canonical)
        router.flip()  # the first generation (qdrant) is superseded: it must be destroyed, not orphaned
        assert router.release_log[-1]["reason"] == "superseded" and router.release_log[-1]["label"] == "qdrant"
        assert raw_qdrant_text(qdrant) == ""  # its collection is gone
        assert router.previous_label == "pgvector"
    finally:
        third_store.client.delete_collection(third_store.collection_name)
