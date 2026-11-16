"""pgvector adapter against a real PostgreSQL with the vector extension.

Needs ``ENGINE_TEST_PG_DSN`` (or ``NS_TEST_PG_DSN``) to point at a DISPOSABLE database whose
role may create schemas (and, if needed, the ``vector`` extension); fails instead of skipping
when ``NS_INTEGRATION_REQUIRED`` is set (see ``engine_env``).  Every test
uses its own random schema and drops it afterwards.  The parity test also
needs ``ENGINE_TEST_QDRANT_URL`` (or ``QDRANT_URL``).  Data is synthetic; embeddings are a
deterministic hash; no model is called.
"""

import random
import threading
import uuid

import pytest

from engine_env import PG_DSN, QDRANT_URL, require_env

pytestmark = [pytest.mark.integration]


@pytest.fixture(autouse=True, scope="module")
def _stores_configured():
    require_env(PostgreSQL=PG_DSN)

from engine_adapters_current import QdrantVectorAdapter, hashed_bag_embedder  # noqa: E402
from engine_adapters_pgvector import PgVectorAdapter, SchemaVersionError  # noqa: E402
from contracts_engines import CapabilityRequirement, EngineAdapterKind  # noqa: E402
from engine_ports import (  # noqa: E402
    AdapterRegistry,
    EngineRecord,
    OPERATIONS_BY_KIND,
    OperationContext,
    PortFailure,
    TenantFilter,
    run_vector_store_conformance,
)

DIMS = 64
embed = hashed_bag_embedder(DIMS)
_SCHEMAS: list[str] = []


def _admin(sql):
    import psycopg

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute(sql)


@pytest.fixture(autouse=True, scope="module")
def _drop_schemas():
    yield
    for schema in _SCHEMAS:
        try:
            _admin(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        except Exception:
            pass


def make(schema=None, dims=DIMS):
    schema = schema or "ns_t_" + uuid.uuid4().hex[:12]
    _SCHEMAS.append(schema)
    adapter = PgVectorAdapter(PG_DSN, embed=embed, dimensions=dims, schema=schema)
    adapter.ensure_schema(create_extension=True)
    return adapter


def ctx(tenant="tenant-a"):
    # The adapter honors the deadline (statement and lock timeouts), so it must fit a real call.
    return OperationContext(tenant_id=tenant, correlation_id="c", deadline_ms=10_000)


def rec(rid, revision=1, text="red apple", tenant="tenant-a", labels=()):
    return EngineRecord(record_id=rid, revision=revision, tenant_id=tenant, text=text, labels=tuple(labels))


def test_full_vector_conformance_on_real_postgres():
    assert run_vector_store_conformance(make) == ()


def test_ensure_schema_is_idempotent_and_records_version_and_dimensions():
    a = make()
    assert a.ensure_schema() == 1 and a.ensure_schema() == 1
    import psycopg

    with psycopg.connect(PG_DSN) as conn:
        rows = conn.execute(f"SELECT version, dimensions FROM {a._schema}.schema_version").fetchall()
    assert rows == [(1, DIMS)]


def test_schema_refuses_dimension_change_and_newer_versions():
    a = make()
    with pytest.raises(SchemaVersionError, match="dimensions"):
        PgVectorAdapter(PG_DSN, embed=lambda t: [0.0] * 8, dimensions=8, schema=a._schema).ensure_schema()
    _admin(f"INSERT INTO {a._schema}.schema_version (version, dimensions) VALUES (99, {DIMS})")
    with pytest.raises(SchemaVersionError, match="newer"):
        a.ensure_schema()


def test_concurrent_first_time_setup_is_safe():
    schema = "ns_t_" + uuid.uuid4().hex[:12]
    _SCHEMAS.append(schema)
    errors = []

    def setup():
        try:
            PgVectorAdapter(PG_DSN, embed=embed, dimensions=DIMS, schema=schema).ensure_schema(create_extension=True)
        except Exception as error:  # noqa: BLE001
            errors.append(repr(error))

    threads = [threading.Thread(target=setup) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    import psycopg

    with psycopg.connect(PG_DSN) as conn:
        assert conn.execute(f"SELECT count(*) FROM {schema}.schema_version").fetchone()[0] == 1


def test_concurrent_writers_never_break_the_revision_fence():
    for _ in range(12):
        a = make()
        revisions = list(range(1, 9))
        random.shuffle(revisions)
        barrier = threading.Barrier(len(revisions))
        unexpected = []

        def write(revision):
            barrier.wait()
            try:
                a.apply(ctx(), rec("r", revision, f"text {revision}"))
            except PortFailure:
                pass
            except Exception as error:  # noqa: BLE001
                unexpected.append(repr(error))

        threads = [threading.Thread(target=write, args=(r,)) for r in revisions]
        [t.start() for t in threads]
        [t.join() for t in threads]
        stored = a.get(ctx(), "r")
        assert unexpected == [] and stored.revision == 8 and stored.text == "text 8"


def test_apply_racing_a_removal_never_resurrects_an_older_revision():
    for _ in range(20):
        a = make()
        a.apply(ctx(), rec("r", 1, "first"))
        barrier = threading.Barrier(2)
        errors = []

        def remover():
            barrier.wait()
            try:
                a.remove(ctx(), "r", 5)
            except PortFailure:
                pass
            except Exception as error:  # noqa: BLE001
                errors.append(repr(error))

        def writer():
            barrier.wait()
            try:
                a.apply(ctx(), rec("r", 3, "older"))
            except PortFailure:
                pass
            except Exception as error:  # noqa: BLE001
                errors.append(repr(error))

        threads = [threading.Thread(target=remover), threading.Thread(target=writer)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert errors == [] and a.get(ctx(), "r") is None


def test_tombstone_keeps_only_identity_and_revision():
    a = make()
    a.apply(ctx(), rec("r", 1, "secret words", labels=("pii",)))
    a.remove(ctx(), "r", 4)
    import psycopg

    with psycopg.connect(PG_DSN) as conn:
        row = conn.execute(
            f"SELECT revision, removed, body, labels, embedding IS NULL FROM {a._schema}.records"
        ).fetchone()
    assert row == (4, True, "", [], True)
    with pytest.raises(PortFailure) as stale:
        a.apply(ctx(), rec("r", 2, "again"))
    assert stale.value.code.value == "stale_revision"
    a.apply(ctx(), rec("r", 5, "reborn"))
    assert a.get(ctx(), "r").text == "reborn"


def test_hostile_identifiers_round_trip_as_data():
    a = make()
    nasty = "r'\"); DROP SCHEMA public CASCADE;--"
    a.apply(ctx(), rec(nasty, 1, "q' OR 1=1 --", labels=("l'1",)))
    assert a.get(ctx(), nasty).text == "q' OR 1=1 --"
    flt = TenantFilter(tenant_id="tenant-a", labels_all=("l'1",), authorized_record_ids=(nasty,))
    assert [h.record_id for h in a.search(ctx(), "OR", flt, 5)] == [nasty]


def test_authorization_and_tenant_filters_apply_before_the_limit_in_a_large_corpus():
    a = make()
    rng = random.Random(3)
    words = [f"w{i}" for i in range(50)]
    for i in range(300):
        a.apply(ctx("tenant-a"), rec(f"a{i}", 1, " ".join(rng.choices(words, k=6)), tenant="tenant-a"))
        a.apply(ctx("tenant-b"), rec(f"b{i}", 1, " ".join(rng.choices(words, k=6)), tenant="tenant-b"))
    allowed = tuple(f"a{i}" for i in range(0, 300, 37))
    hits = a.search(ctx(), "w1 w2 w3", TenantFilter(tenant_id="tenant-a", authorized_record_ids=allowed), 5)
    assert len(hits) == 5 and all(h.record_id in allowed for h in hits)
    everything = a.search(ctx(), "w1", TenantFilter(tenant_id="tenant-a"), 1000)
    assert len(everything) == 300 and all(h.record_id.startswith("a") for h in everything)


def test_registry_activation_needs_a_ready_schema():
    schema = "ns_t_" + uuid.uuid4().hex[:12]
    _SCHEMAS.append(schema)
    not_ready = PgVectorAdapter(PG_DSN, embed=embed, dimensions=DIMS, schema=schema)
    reqs = tuple(
        CapabilityRequirement(
            operation=n, contract_schema_version="candidate-v1", require_configured=True, require_healthy=True,
            qualification_profile_reference=None, qualification_profile_version=None,
        )
        for n in OPERATIONS_BY_KIND[EngineAdapterKind.VECTOR_STORE]
    )
    registry = AdapterRegistry()
    adapter_id = registry.register(not_ready, EngineAdapterKind.VECTOR_STORE)
    from engine_ports import ActivationRejected

    with pytest.raises(ActivationRejected):
        registry.activate(adapter_id, reqs)
    not_ready.ensure_schema(create_extension=True)
    assert registry.activate(adapter_id, reqs).adapter_id == adapter_id


# ---- parity with the current Qdrant adapter ---------------------------------


def _qdrant_adapter():
    from mem0.vector_stores.qdrant import Qdrant

    store = Qdrant(collection_name="ns_parity_" + uuid.uuid4().hex[:10], embedding_model_dims=DIMS, url=QDRANT_URL)
    return QdrantVectorAdapter(store, embed), store


def _script():
    """One deterministic operation script: applies, updates, stale/conflict, removals, re-applies."""
    rng = random.Random(11)
    words = [f"w{i}" for i in range(60)]
    ops = []
    for tenant in ("tenant-a", "tenant-b"):
        for i in range(25):
            ops.append(("apply", tenant, f"r{i}", 1, " ".join(rng.choices(words, k=7)), ("x",) if i % 3 == 0 else ()))
    for i in range(0, 25, 4):
        ops.append(("apply", "tenant-a", f"r{i}", 2, " ".join(rng.choices(words, k=7)), ()))   # update
        ops.append(("apply", "tenant-a", f"r{i}", 1, "stale write", ()))                      # stale
        ops.append(("apply", "tenant-a", f"r{i}", 2, "different payload", ()))                # conflict
    for i in range(0, 25, 5):
        ops.append(("remove", "tenant-a", f"r{i}", 6, None, ()))
        ops.append(("apply", "tenant-a", f"r{i}", 3, "older after removal", ()))              # resurrection attempt
        ops.append(("remove", "tenant-a", f"r{i}", 7, None, ()))                              # remove again
    ops.append(("remove", "tenant-b", "r1", 1, None, ()))
    ops.append(("remove", "tenant-b", "never-existed", 1, None, ()))
    ops.append(("apply", "tenant-b", "r0", 9, "revived", ()))
    return ops


def _run_script(adapter, ops):
    outcomes = []
    for kind, tenant, rid, revision, text, labels in ops:
        c = ctx(tenant)
        try:
            if kind == "apply":
                adapter.apply(c, rec(rid, revision, text, tenant, labels))
            else:
                adapter.remove(c, rid, revision)
            outcomes.append("ok")
        except PortFailure as failure:
            outcomes.append(failure.code.value)
    return outcomes


def _state(adapter, tenants=("tenant-a", "tenant-b"), count=26):
    state = {}
    for tenant in tenants:
        for rid in [f"r{i}" for i in range(count)] + ["never-existed"]:
            row = adapter.get(ctx(tenant), rid)
            state[(tenant, rid)] = None if row is None else (row.revision, row.text, row.labels)
    return state


def test_exact_state_parity_with_the_current_qdrant_adapter():
    require_env(Qdrant=QDRANT_URL)
    qdrant, store = _qdrant_adapter()
    try:
        postgres = make()
        ops = _script()
        q_out, p_out = _run_script(qdrant, ops), _run_script(postgres, ops)
        # 1. Exact state: per-operation outcomes and final records (ids, revisions, tenants, deletes).
        diffs = [(i, ops[i][:4], q, p) for i, (q, p) in enumerate(zip(q_out, p_out)) if q != p]
        assert diffs == [], diffs
        assert _state(qdrant) == _state(postgres)
        assert sum(1 for o in p_out if o == "stale_revision") > 5 and "idempotency_conflict" in p_out
        # 2. Retrieval is compared separately: only invariants, never identical ranking.
        overlaps = []
        for tenant in ("tenant-a", "tenant-b"):
            live = {k[1] for k, v in _state(postgres).items() if k[0] == tenant and v is not None}
            allowed = tuple(sorted(live)[::2])
            for query in ("w1 w2", "w10 w11 w12", "stale write"):
                flt = TenantFilter(tenant_id=tenant, authorized_record_ids=allowed)
                q_hits = [h.record_id for h in qdrant.search(ctx(tenant), query, flt, 8)]
                p_hits = [h.record_id for h in postgres.search(ctx(tenant), query, flt, 8)]
                assert q_hits and p_hits, "retrieval returned nothing; the invariants below would be vacuous"
                for hits in (q_hits, p_hits):
                    assert set(hits) <= set(allowed) and set(hits) <= live
                overlaps.append(len(set(q_hits) & set(p_hits)) / max(len(set(q_hits) | set(p_hits)), 1))
        print("retrieval top-8 overlap (informational):", [round(o, 2) for o in overlaps])
    finally:
        try:
            store.client.delete_collection(store.collection_name)
        except Exception:
            pass


# ── removal fence before a first apply, and content-free typed errors ─────────────────


def _code(call):
    try:
        call()
    except PortFailure as failure:
        return failure.code
    return "ok"


def test_removal_of_a_never_applied_record_fences_any_later_apply_at_or_below_it():
    from contracts_errors import InternalErrorCode as E

    a = make()
    # the erasure (revision 3) is processed before the record's write (an out-of-order stage)
    assert _code(lambda: a.remove(ctx(), "r-late", 3)) is E.MISSING_RESOURCE
    for revision in (1, 2):
        assert _code(lambda: a.apply(ctx(), rec("r-late", revision, "erased content"))) is E.STALE_REVISION
    assert _code(lambda: a.apply(ctx(), rec("r-late", 3, "erased content"))) is E.IDEMPOTENCY_CONFLICT
    assert a.get(ctx(), "r-late") is None
    assert a.search(ctx(), "erased content", TenantFilter(tenant_id="tenant-a"), 10) == ()
    a.apply(ctx(), rec("r-late", 4, "a newer write"))  # a newer canonical revision is accepted
    assert a.get(ctx(), "r-late").text == "a newer write"
    # the fence is per tenant and holds no content
    a.apply(ctx("tenant-b"), rec("r-late", 1, "tenant b's own record", tenant="tenant-b"))
    assert a.get(ctx("tenant-b"), "r-late").revision == 1


def test_a_repeated_removal_raises_the_fence_to_the_newest_revision():
    from contracts_errors import InternalErrorCode as E

    a = make()
    a.apply(ctx(), rec("r1", 1, "text"))
    a.remove(ctx(), "r1", 2)
    assert _code(lambda: a.remove(ctx(), "r1", 5)) is E.MISSING_RESOURCE  # already removed
    assert _code(lambda: a.apply(ctx(), rec("r1", 4, "text"))) is E.STALE_REVISION
    assert _code(lambda: a.remove(ctx(), "r1", 3)) is E.MISSING_RESOURCE  # a lower repeat changes nothing
    assert _code(lambda: a.apply(ctx(), rec("r1", 5, "text"))) is E.IDEMPOTENCY_CONFLICT
    a.apply(ctx(), rec("r1", 6, "back"))
    assert a.get(ctx(), "r1").text == "back"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_embedding_is_a_typed_error(bad):
    from contracts_errors import InternalErrorCode as E

    a = make()
    a._embed = lambda text: [bad] * DIMS
    assert _code(lambda: a.apply(ctx(), rec("r1", 1))) is E.INVALID_SCHEMA
    assert _code(lambda: a.search(ctx(), "q", TenantFilter(tenant_id="tenant-a"), 5)) is E.INVALID_SCHEMA


def test_a_nul_byte_in_text_or_labels_is_a_typed_error_not_a_driver_error():
    from contracts_errors import InternalErrorCode as E

    a = make()
    assert _code(lambda: a.apply(ctx(), rec("r1", 1, "nul\x00byte"))) is E.INVALID_SCHEMA
    assert _code(lambda: a.apply(ctx(), rec("r2", 1, "fine", labels=("bad\x00label",)))) is E.INVALID_SCHEMA
    assert _code(lambda: a.search(ctx(), "fine", TenantFilter(tenant_id="tenant-a", labels_all=("x\x00",)), 5)) is E.INVALID_SCHEMA
    assert a.get(ctx(), "r1") is None and a.get(ctx(), "r2") is None


def test_an_embedder_failure_is_typed_and_never_carries_its_message():
    from contracts_errors import InternalErrorCode as E

    a = make()

    def failing(text):
        raise RuntimeError("provider said: api key sk-CANARY-123 rejected")

    a._embed = failing
    for call in (lambda: a.apply(ctx(), rec("r1", 1)), lambda: a.search(ctx(), "q", TenantFilter(tenant_id="tenant-a"), 5)):
        with pytest.raises(PortFailure) as raised:
            call()
        assert raised.value.code is E.UNAVAILABLE_DEPENDENCY
        assert "CANARY" not in str(raised.value) and "CANARY" not in raised.value.detail
        assert raised.value.__cause__ is None and raised.value.__suppress_context__


def test_a_database_failure_is_typed_and_content_free():
    from contracts_errors import InternalErrorCode as E

    a = make()

    def broken():
        raise __import__("psycopg").OperationalError("connection to server at db.invalid failed: password=hunter2")

    a._connect = broken
    for call in (
        lambda: a.apply(ctx(), rec("r1", 1)), lambda: a.remove(ctx(), "r1", 1), lambda: a.get(ctx(), "r1"),
        lambda: a.search(ctx(), "q", TenantFilter(tenant_id="tenant-a"), 5),
    ):
        with pytest.raises(PortFailure) as raised:
            call()
        assert raised.value.code is E.UNAVAILABLE_DEPENDENCY
        assert "hunter2" not in str(raised.value) and "hunter2" not in raised.value.detail
        assert raised.value.__cause__ is None and raised.value.__suppress_context__


# ---- every store call is bounded by the operation deadline


def _run_bounded(fn, wait_s):
    import threading
    import time

    out = {}

    def run():
        started = time.monotonic()
        try:
            out["result"] = fn()
        except BaseException as error:  # noqa: BLE001
            out["error"] = error
        out["elapsed"] = time.monotonic() - started

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(wait_s)
    return None if thread.is_alive() else out


def test_a_call_behind_a_held_record_lock_ends_at_its_deadline():
    import psycopg

    from contracts_errors import InternalErrorCode as E

    a = make()
    a.apply(ctx(), rec("locked", 1))
    holder = psycopg.connect(PG_DSN)
    holder.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{a._schema}\x1ftenant-a\x1flocked",))
    try:
        short = OperationContext(tenant_id="tenant-a", correlation_id="c", deadline_ms=500)
        out = _run_bounded(lambda: a.apply(short, rec("locked", 2)), 8)
    finally:
        holder.rollback()
        holder.close()
    assert out is not None, "the call was still blocked behind the lock 8 s later (500 ms deadline)"
    assert isinstance(out.get("error"), PortFailure) and out["error"].code is E.UNAVAILABLE_DEPENDENCY
    assert out["elapsed"] < 3, out["elapsed"]
    assert a.get(ctx(), "locked").revision == 1  # nothing was written


def test_a_server_that_never_answers_is_bounded_by_the_connect_timeout():
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    try:
        silent = PgVectorAdapter(f"postgresql://u:p@127.0.0.1:{listener.getsockname()[1]}/db", embed=embed,
                                 dimensions=DIMS, schema="ns_silent")
        short = OperationContext(tenant_id="tenant-a", correlation_id="c", deadline_ms=500)
        out = _run_bounded(lambda: silent.get(short, "x"), 8)
    finally:
        listener.close()
    assert out is not None, "the call was still connecting 8 s later"
    assert isinstance(out.get("error"), PortFailure), out  # libpq's two-second connect-timeout floor applies


def test_destroying_a_busy_generation_fails_instead_of_waiting_forever(monkeypatch):
    import psycopg

    import engine_adapters_pgvector

    monkeypatch.setattr(engine_adapters_pgvector, "DESTROY_LOCK_TIMEOUT_MS", 300)
    a = make()
    a.apply(ctx(), rec("busy", 1))
    holder = psycopg.connect(PG_DSN)
    holder.execute(f"SELECT * FROM {a._schema}.records")  # an open transaction holds a lock on the table
    try:
        out = _run_bounded(a.destroy, 8)
    finally:
        holder.rollback()
        holder.close()
    assert out is not None and "error" in out, "destroy waited for the busy schema"
    assert a.get(ctx(), "busy") is not None  # the generation is intact


def test_a_search_never_returns_more_than_the_hit_bound(monkeypatch):
    import engine_adapters_pgvector

    monkeypatch.setattr(engine_adapters_pgvector, "MAX_SEARCH_LIMIT", 50, raising=False)
    a = make()
    for i in range(120):
        a.apply(ctx(), rec(f"r{i}", 1, text=f"common word{i}"))
    hits = a.search(ctx(), "common", TenantFilter(tenant_id="tenant-a"), 10**9)
    assert len(hits) == 50  # bounded, not the whole tenant
    assert len(a.search(ctx(), "common", TenantFilter(tenant_id="tenant-a"), 7)) == 7  # a smaller limit holds


@pytest.mark.parametrize("deadline_ms", [2**31 - 1, 2**31, 2**40, 2**53 - 1])
def test_a_deadline_beyond_the_servers_timeout_range_still_works(deadline_ms):
    """``deadline_ms`` accepts any safe integer; PostgreSQL timeouts are 32-bit milliseconds."""

    a = make()
    long = OperationContext(tenant_id="tenant-a", correlation_id="c", deadline_ms=deadline_ms)
    a.apply(long, rec("far", 1))
    assert a.get(long, "far").revision == 1
    assert [h.record_id for h in a.search(long, "red", TenantFilter(tenant_id="tenant-a"), 5)] == ["far"]
    a.remove(long, "far", 2)
    assert a.get(long, "far") is None
