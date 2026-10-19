"""Explicit gap matrix for the direct-store substitution (pgvector) against the current Qdrant path.

Both adapters run against REAL stores.  Every row of the vector-store obligation is classified as ``same``
(measured identical), ``differs`` (measured, both acceptable, reported), ``gap_both`` (neither engine provides
it through the port; the canonical source or the legacy path covers it) or ``gap_routed`` (provided by the
current stack outside the port, so lost when recall is routed).  Each measured verdict is asserted, so a
behavior change fails here; the matrix is written as JSON when ``ENGINE_GAP_MATRIX_OUT`` names a file.

Skipped unless ``ENGINE_TEST_PG_DSN`` and ``ENGINE_TEST_QDRANT_URL`` point at disposable services.
"""

from __future__ import annotations

import json
import os

import pytest

from tests.test_engine_adapters_pgvector_integration import (  # noqa: F401
    DIMS,
    PG_DSN,
    QDRANT_URL,
    _admin,
    _qdrant_adapter,
    ctx,
    make,
    rec,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (PG_DSN and QDRANT_URL), reason="ENGINE_TEST_PG_DSN and ENGINE_TEST_QDRANT_URL are not set"),
]

from contracts_engines import CapabilityHealth  # noqa: E402
from engine_ports import PortFailure, TenantFilter  # noqa: E402


@pytest.fixture
def pair():
    qdrant, store = _qdrant_adapter()
    postgres = make()
    yield qdrant, store, postgres
    try:
        store.client.delete_collection(store.collection_name)
    except Exception:
        pass
    _admin(f"DROP SCHEMA IF EXISTS {postgres._schema} CASCADE")


def _outcome(fn):
    try:
        fn()
        return "ok"
    except PortFailure as failure:
        return failure.code.value
    except Exception as error:  # noqa: BLE001
        return type(error).__name__


def test_the_substitution_gap_matrix(pair):
    qdrant, store, postgres = pair
    rows = []

    def row(obligation, verdict, qdrant_result, pgvector_result, note=""):
        rows.append({"obligation": obligation, "verdict": verdict, "current_qdrant": qdrant_result,
                     "pgvector": pgvector_result, "note": note})

    # 1. Tombstones keep only identity and revision (no text, labels or meaningful vector) on both.
    for adapter in (qdrant, postgres):
        adapter.apply(ctx(), rec("t1", 1, "secret words here", labels=("pii",)))
        adapter.remove(ctx(), "t1", 3)
    points, _ = store.client.scroll(collection_name=store.collection_name, with_payload=True, with_vectors=False, limit=50)
    q_residue = [p.payload for p in points if p.payload.get("canonical_id") == "t1"]
    import psycopg

    with psycopg.connect(PG_DSN) as conn:
        p_residue = conn.execute(f"SELECT body, labels FROM {postgres._schema}.records WHERE record_id = 't1'").fetchall()
    q_clean = all("secret" not in json.dumps(p) and "pii" not in json.dumps(p) for p in q_residue)
    p_clean = all(body == "" and not labels for body, labels in p_residue)
    assert q_clean and p_clean
    row("remove leaves a content-free tombstone", "same", "content-free", "content-free")

    # 2. Stale and conflicting applies are refused identically.
    stale = (_outcome(lambda: qdrant.apply(ctx(), rec("t1", 2, "older"))),
             _outcome(lambda: postgres.apply(ctx(), rec("t1", 2, "older"))))
    assert stale == ("stale_revision", "stale_revision")
    row("revision fence after a tombstone", "same", stale[0], stale[1])

    # 2b. A remove carrying an older revision is refused too: on a live record as a stale revision, on a
    # tombstone as a missing record (there is nothing left to remove), identically on both.
    for adapter in (qdrant, postgres):
        adapter.apply(ctx(), rec("live", 5, "live record"))
    stale_remove = tuple(
        (_outcome(lambda a=adapter: a.remove(ctx(), "t1", 2)), _outcome(lambda a=adapter: a.remove(ctx(), "live", 4)))
        for adapter in (qdrant, postgres)
    )
    assert stale_remove == (("missing_resource", "stale_revision"),) * 2, stale_remove
    assert qdrant.get(ctx(), "live").revision == postgres.get(ctx(), "live").revision == 5  # nothing removed
    row("older-revision remove refused (on a tombstone; on a live record)", "same",
        list(stale_remove[0]), list(stale_remove[1]))

    # 3. Authorization and tenant filters apply before the limit; inaccessible distractors never return.
    for adapter in (qdrant, postgres):
        for i in range(40):
            adapter.apply(ctx(), rec(f"d{i}", 1, "blue whale song"))
            adapter.apply(ctx("tenant-b"), rec(f"d{i}", 1, "blue whale song", tenant="tenant-b"))
        adapter.apply(ctx(), rec("allowed", 1, "a quiet blue whale"))
    flt = TenantFilter(tenant_id="tenant-a", authorized_record_ids=("allowed",))
    q_hits = [h.record_id for h in qdrant.search(ctx(), "blue whale", flt, 1)]
    p_hits = [h.record_id for h in postgres.search(ctx(), "blue whale", flt, 1)]
    assert q_hits == p_hits == ["allowed"]
    row("authorized filter before limit (40 inaccessible distractors)", "same", q_hits, p_hits)

    # 4. Ranking is measured, never required to match.
    open_flt = TenantFilter(tenant_id="tenant-a")
    q_rank = [h.record_id for h in qdrant.search(ctx(), "blue whale", open_flt, 10)]
    p_rank = [h.record_id for h in postgres.search(ctx(), "blue whale", open_flt, 10)]
    overlap = len(set(q_rank) & set(p_rank)) / max(len(set(q_rank) | set(p_rank)), 1)
    row("dense ranking for the same state", "differs" if q_rank != p_rank else "same",
        q_rank[:3], p_rank[:3], f"top-10 overlap {overlap:.2f}; compared separately from exact state")

    # 5. Hostile identifiers round-trip as data on both.
    hostile = "x'); DROP TABLE records;--\x1f‮"
    for adapter in (qdrant, postgres):
        adapter.apply(ctx(), rec(hostile, 1, "hostile id"))
    assert qdrant.get(ctx(), hostile).record_id == postgres.get(ctx(), hostile).record_id == hostile
    row("hostile record identifiers", "same", "round-trip", "round-trip")

    # 6. A wrong embedding dimension is refused by both (types differ).
    q_dim = _outcome(lambda: qdrant.__class__(store, lambda text: [0.1] * (DIMS + 1)).apply(ctx(), rec("dim", 1, "x")))
    from engine_adapters_pgvector import PgVectorAdapter

    p_dim = _outcome(lambda: PgVectorAdapter(PG_DSN, embed=lambda text: [0.1] * (DIMS + 1), dimensions=DIMS,
                                             schema=postgres._schema).apply(ctx(), rec("dim", 1, "x")))
    assert (q_dim, p_dim) == ("UnexpectedResponse", "invalid_schema"), (q_dim, p_dim)
    row("embedding dimension mismatch refused", "differs", q_dim, p_dim,
        "both refuse; pgvector raises the typed contract error, the Qdrant client raises its own")

    # 7. Receipts name an engine and index version but no embedding model identity.
    q_receipt = qdrant.apply(ctx(), rec("r1", 1, "receipt"))
    p_receipt = postgres.apply(ctx(), rec("r1", 1, "receipt"))
    assert q_receipt.index_version == q_receipt.engine_version and p_receipt.index_version == p_receipt.engine_version
    row("receipt index/model identity", "gap_both", q_receipt.index_version, p_receipt.index_version,
        "index_version repeats the adapter version; the embedding model/version is not recorded")

    # 8. Health reflects a missing store: both adapters probe it (measured last: it breaks the Qdrant store).
    assert qdrant.manifest().operations[0].health is CapabilityHealth.HEALTHY
    assert postgres.manifest().operations[0].health is CapabilityHealth.HEALTHY
    broken_pg = PgVectorAdapter(PG_DSN, embed=lambda text: [0.0] * DIMS, dimensions=DIMS, schema="ns_absent_schema")
    store.client.delete_collection(store.collection_name)
    q_health = qdrant.manifest().operations[0].health
    p_health = broken_pg.manifest().operations[0].health
    assert (q_health, p_health) == (CapabilityHealth.UNHEALTHY, CapabilityHealth.UNHEALTHY)
    row("health when the store is missing", "same", q_health.value, p_health.value,
        "the Qdrant adapter probes its collection; pgvector probes its schema and extension")

    # Port-level and recall-path rows (no runtime difference to measure; stated from the port and the code).
    row("stable scan / cursor", "gap_both", "not in port", "not in port",
        "catch-up and rollback read the canonical source instead of scanning the engine")
    row("sparse / lexical query", "gap_both", "not in port", "not in port",
        "the current stack's lexical leg stays on the legacy path")
    row("graph and lexical legs, reinforcement boost, shared-pool weave", "gap_routed", "legacy path only",
        "not available", "routed recall serves dense vector hits hydrated through the caller's batch read")
    row("exact revisioned apply/remove/get and per-operation outcomes", "same", "measured",
        "measured", "test_exact_state_parity_with_the_current_qdrant_adapter (same run)")

    assert all(r["verdict"] in {"same", "differs", "gap_both", "gap_routed"} for r in rows)
    out = os.environ.get("ENGINE_GAP_MATRIX_OUT")
    if out:
        with open(out, "w") as handle:
            json.dump({"rows": rows}, handle, indent=2, default=str)
    print(json.dumps(rows, indent=1, default=str))
