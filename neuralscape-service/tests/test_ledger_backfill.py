"""Legacy backfill: one canonical revision-1 reference per row, read-only, idempotent, resumable."""

from __future__ import annotations

import pytest

from ledger.backfill import (
    Backfill,
    LegacyRow,
    baseline_marker,
    content_checksum,
    is_backfill_baseline,
    single_tenant_resolver,
)
from ledger.sqlite import SqliteLedger
from write_producer import SqlitePayloadStore

TENANT = "tenant-default"


class FakeScanner:
    """Pages over fixed rows in id order; exposes no write method, and can die mid-pass."""

    def __init__(self, rows: list[LegacyRow]) -> None:
        self.rows = sorted(rows, key=lambda r: r.id)
        self.scans: list[tuple[str | None, int]] = []
        self.die_on_scan: int | None = None

    def scan(self, cursor, limit):
        self.scans.append((cursor, limit))
        if self.die_on_scan is not None and len(self.scans) == self.die_on_scan:
            raise RuntimeError("interrupted")
        start = 0 if cursor is None else next(i for i, r in enumerate(self.rows) if r.id == cursor)
        page = self.rows[start:start + limit]
        nxt = self.rows[start + limit].id if start + limit < len(self.rows) else None
        return page, nxt


def row(i: int, owner: str | None = "alice", content: str | None = None) -> LegacyRow:
    return LegacyRow(id=f"mem-{i:03d}", owner_user_id=owner, content=content or f"fact number {i}")


@pytest.fixture
def env(tmp_path):
    ledger = SqliteLedger(str(tmp_path / "ledger.db"))
    kv = SqlitePayloadStore(str(tmp_path / "kv.db"))
    yield ledger, kv
    kv.close()
    ledger.close()


def make(env, rows, resolver=None, page=4, name="backfill"):
    ledger, kv = env
    scanner = FakeScanner(rows)
    return Backfill(ledger, scanner, resolver or single_tenant_resolver(TENANT), kv, page_size=page, name=name), scanner


def test_every_row_gets_exactly_one_revision_one_reference_and_the_pass_is_idempotent(env) -> None:
    ledger, _ = env
    backfill, scanner = make(env, [row(i) for i in range(10)])
    first = backfill.run()
    assert first.done and first.counts.scanned == 10 and first.counts.registered == 10
    for i in range(10):
        ref = ledger.get_object_ref(TENANT, f"mem-{i:03d}", 1)
        assert ref is not None and ref.content_revision == 1 and is_backfill_baseline(ref)
        assert ref.checksum == content_checksum(f"fact number {i}")
        assert ref.source_refs == (baseline_marker(TENANT),)
        assert ledger.get_object_ref(TENANT, f"mem-{i:03d}", 2) is None
    second = backfill.run()  # a finished pass starts a fresh one; nothing new to register
    assert second.counts.scanned == 10 and second.counts.registered == 0 and second.counts.already_canonical == 10
    assert backfill.reconcile().ok


def test_the_scan_is_read_only_against_the_legacy_store(env) -> None:
    backfill, scanner = make(env, [row(i) for i in range(6)])
    backfill.run()
    backfill.reconcile()
    assert not [name for name in dir(scanner) if name.startswith(("put", "delete", "upsert", "write"))]
    assert all(limit == 4 for _, limit in scanner.scans)  # only bounded page reads were issued


def test_an_interruption_resumes_from_the_checkpoint_and_ends_where_a_clean_pass_ends(env, tmp_path) -> None:
    ledger, kv = env
    rows = [row(i) for i in range(11)]
    backfill, scanner = make(env, rows)
    scanner.die_on_scan = 3  # two pages are checkpointed, the third read dies
    with pytest.raises(RuntimeError):
        backfill.run()
    assert backfill.report().cursor == rows[8].id and not backfill.report().done
    resumed = Backfill(ledger, FakeScanner(rows), single_tenant_resolver(TENANT), kv, page_size=4)  # a new process
    report = resumed.run()
    assert report.done and report.counts.scanned == 11 and report.counts.registered == 11
    assert resumed.reconcile().ok


def test_a_bounded_run_makes_progress_and_a_later_run_finishes(env) -> None:
    backfill, _ = make(env, [row(i) for i in range(9)])
    partial = backfill.run(max_pages=1)
    assert not partial.done and partial.counts.scanned == 4
    assert backfill.run().counts.registered == 9


def test_owners_that_cannot_be_mapped_are_quarantined_and_reported_never_guessed(env) -> None:
    ledger, _ = env
    rows = [row(1), row(2, owner="ghost"), row(3, owner=None), row(4, owner="bob")]
    resolver = lambda owner: {"alice": "tenant-a", "bob": "tenant-b"}.get(owner)
    backfill, _ = make(env, rows, resolver)
    report = backfill.run()
    assert report.counts.quarantined == 2 and report.counts.registered == 2
    assert {q["row_id"] for q in report.quarantined} == {"mem-002", "mem-003"}
    assert ledger.get_object_ref("tenant-a", "mem-001", 1) and ledger.get_object_ref("tenant-b", "mem-004", 1)
    assert not ledger.has_object_refs(TENANT, "mem-002") and not ledger.has_object_refs("tenant-a", "mem-002")
    reconcile = backfill.reconcile()
    assert reconcile.ok and reconcile.quarantined == 2  # accounted for, not missing


def test_an_erased_record_never_gets_a_reference_back(env) -> None:
    ledger, _ = env
    ledger.tombstone(TENANT, "mem-002", 1, "user_delete")
    backfill, _ = make(env, [row(1), row(2), row(3)])
    report = backfill.run()
    assert report.counts.erased_skipped == 1 and report.counts.registered == 2
    assert not ledger.has_object_refs(TENANT, "mem-002")
    assert backfill.reconcile().ok


def test_a_row_the_write_path_registered_first_is_left_alone(env) -> None:
    ledger, _ = env
    from write_producer import MemoryRevisions

    MemoryRevisions(ledger).register_baseline(TENANT, "mem-001", content_checksum("fact number 1"), "intent-x")
    backfill, _ = make(env, [row(1), row(2)])
    report = backfill.run()
    assert report.counts.already_canonical == 1 and report.counts.registered == 1
    assert not is_backfill_baseline(ledger.get_object_ref(TENANT, "mem-001", 1))  # still the write path's own


def test_reconcile_names_a_row_that_appeared_after_the_pass(env) -> None:
    backfill, scanner = make(env, [row(i) for i in range(5)])
    backfill.run()
    scanner.rows.append(row(99))
    report = backfill.reconcile()
    assert not report.ok and report.missing == ["mem-099"]
    backfill.run()
    assert backfill.reconcile().ok


def test_reconcile_flags_a_row_that_is_scanned_twice(env) -> None:
    backfill, scanner = make(env, [row(1), row(2)])
    scanner.rows.append(row(2))  # a duplicate id in the legacy store
    backfill.run()
    assert backfill.reconcile().duplicate_rows == ["mem-002"]


def test_the_write_path_accepts_a_baseline_the_backfill_registered_between_store_and_register(tmp_path) -> None:
    """A row written by the ledger path while the pass runs is registered by whichever side is first."""

    from tests.test_v1_durable_write import PAYLOAD, FakeService, run, worker

    ledger = SqliteLedger(str(tmp_path / "l.db"))
    payloads = SqlitePayloadStore(str(tmp_path / "p.db"))
    from write_producer import DEFAULT_TENANT, DurableWriteProducer

    producer = DurableWriteProducer(ledger, payloads)
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    service = FakeService()
    memory_id = "mem-" + str(abs(hash(PAYLOAD["content"])) % 10**8)
    # the backfill saw the stored row and registered it before the stage's own registration
    original = service.store_raw

    def store_then_backfill(**kw):
        result = original(**kw)
        Backfill(ledger, FakeScanner([LegacyRow(memory_id, "user-1", PAYLOAD["content"])]),
                 single_tenant_resolver(DEFAULT_TENANT), payloads, page_size=5).run()
        return result

    service.store_raw = store_then_backfill
    run(worker((ledger, payloads, producer, None), service).drain())
    assert ledger.get_intent(DEFAULT_TENANT, accepted.intent_id).status.value == "applied"
    assert is_backfill_baseline(ledger.get_object_ref(DEFAULT_TENANT, memory_id, 1))
    assert {r.attempt for r in ledger.receipts(DEFAULT_TENANT, accepted.intent_id)} == {1}  # no failed attempt and retry
    payloads.close()
    ledger.close()
