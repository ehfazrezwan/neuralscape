"""Read-only legacy backfill: one canonical object reference per existing memory row.

The scan reads the legacy stores through a :class:`LegacyScanner` and never writes to them.  For
every row it registers revision 1 in the ledger, attributed to the tenant of the row's owner as the
identity migration mapping resolves it.  Rows whose owner cannot be mapped to exactly one tenant are
quarantined and reported, never guessed, and rows the ledger has already erased are never given a
reference back.

The reference is the same immutable content-digest baseline the write path registers for a new
memory, with one lineage handle that says it was not produced by a ledger write
(``baseline_unregistered``).  It carries no run identity, so repeating the pass registers nothing
new, and a row the write path registered first is left as it is.

The pass is resumable: a checkpoint (cursor plus counts) is appended after each page, and every
registration is idempotent, so an interruption at any point only repeats work.  Row ordering comes
from the scanner (the vector store's point order), so a row written while the pass runs is either
seen by it or seen by the next pass; ``reconcile`` is the proof that nothing is missing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from contracts_references import ReferenceHandle
from ledger.cutover import KeyValueStore, SequencedLog
from ledger.port import LedgerPort, ObjectRef, ObjectRefConflict, ResurrectionDenied

BASELINE_MARKER_ID = "baseline_unregistered"
BASELINE_RESOLVER = "ns:legacy-backfill"
QUARANTINE_DETAIL_LIMIT = 1000


def baseline_marker(tenant_id: str) -> ReferenceHandle:
    return ReferenceHandle(kind="artifact", id=BASELINE_MARKER_ID, tenant_id=tenant_id, resolver=BASELINE_RESOLVER)


def is_backfill_baseline(ref: ObjectRef | None) -> bool:
    return ref is not None and any(h.id == BASELINE_MARKER_ID and h.resolver == BASELINE_RESOLVER for h in ref.source_refs)


def content_checksum(content: str) -> str:
    """The write path's baseline checksum for the same content."""

    from write_producer import request_digest

    return request_digest({"content": content})


@dataclass(frozen=True)
class LegacyRow:
    id: str
    owner_user_id: str | None
    content: str
    visibility: str = "private"
    project_id: str | None = None
    dream_tombstoned: bool = False


class LegacyScanner(Protocol):
    def scan(self, cursor: str | None, limit: int) -> tuple[list[LegacyRow], str | None]:
        """Rows after ``cursor`` in a stable order, and the next cursor (``None`` at the end)."""


OwnerTenantResolver = Callable[[str], str | None]


class QdrantScanner:
    """Scroll the memory collection through the existing vector-store client (read-only)."""

    def __init__(self, client: Any, collection: str, *, id_field: str | None = None) -> None:
        # ``id_field`` names a payload key holding the memory id when it is not the point id
        self._client, self._collection, self._id_field = client, collection, id_field

    def scan(self, cursor: str | None, limit: int) -> tuple[list[LegacyRow], str | None]:
        points, next_offset = self._client.scroll(
            collection_name=self._collection,
            limit=limit,
            offset=cursor,
            with_payload=True,
            with_vectors=False,
        )
        rows = []
        for point in points:
            payload = point.payload or {}
            meta = payload.get("metadata") or {}
            rows.append(
                LegacyRow(
                    id=str(payload.get(self._id_field) if self._id_field and payload.get(self._id_field) else point.id),
                    owner_user_id=payload.get("user_id") or meta.get("owner_user_id") or None,
                    content=payload.get("data") or payload.get("memory") or "",
                    visibility=meta.get("visibility") or payload.get("visibility") or "private",
                    project_id=meta.get("project_id") or None,
                    dream_tombstoned=bool(meta.get("dream_tombstoned")),
                )
            )
        return rows, (None if next_offset is None else str(next_offset))


def authority_owner_resolver(authority: Any, bindings: Any) -> OwnerTenantResolver:
    """Owner -> tenant through the identity binding and the authority store.

    A legacy user id with no binding, or whose subject has no active record, maps to ``None`` and is
    quarantined (the migration mapping never invents an owner).
    """

    from identity_binding import resolve_binding
    from write_producer import LOCAL_ISSUER

    def resolve(owner_user_id: str) -> str | None:
        subject = resolve_binding(bindings, LOCAL_ISSUER, owner_user_id)
        record = None if subject is None else authority.subject_record(subject)
        return None if record is None else record.tenant_id

    return resolve


def single_tenant_resolver(tenant_id: str) -> OwnerTenantResolver:
    """Ungoverned deployments: every owner belongs to the one default tenant."""

    return lambda owner_user_id: tenant_id


@dataclass
class BackfillCounts:
    scanned: int = 0
    registered: int = 0
    already_canonical: int = 0
    quarantined: int = 0
    erased_skipped: int = 0
    dream_tombstoned: int = 0
    rebaselined: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class BackfillReport:
    counts: BackfillCounts
    cursor: str | None
    done: bool
    quarantined: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts.as_dict(),
            "cursor": self.cursor,
            "done": self.done,
            "quarantined": self.quarantined,
        }


@dataclass
class ReconcileReport:
    """Every legacy row has a canonical reference for its current content (or is accounted for)."""

    scanned: int = 0
    canonical: int = 0
    quarantined: int = 0
    erased: int = 0
    missing: list[str] = field(default_factory=list)
    wrong_tenant: list[str] = field(default_factory=list)
    duplicate_rows: list[str] = field(default_factory=list)
    drifted: list[str] = field(default_factory=list)  # content changed after the reference was registered

    @property
    def ok(self) -> bool:
        return not (self.missing or self.wrong_tenant or self.duplicate_rows or self.drifted) and (
            self.scanned == self.canonical + self.quarantined + self.erased
        )

    def summary(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "canonical": self.canonical,
            "quarantined": self.quarantined,
            "erased": self.erased,
            "missing": len(self.missing),
            "wrong_tenant": len(self.wrong_tenant),
            "duplicate_rows": len(self.duplicate_rows),
            "drifted": len(self.drifted),
        }


class Backfill:
    def __init__(
        self,
        ledger: LedgerPort,
        scanner: LegacyScanner,
        owner_tenant: OwnerTenantResolver,
        store: KeyValueStore,
        *,
        page_size: int = 200,
        name: str = "backfill",
        extra_tenants: Iterable[str] = (),
    ) -> None:
        self._ledger, self._scanner, self._owner_tenant = ledger, scanner, owner_tenant
        self._log = SequencedLog(store, name)
        self._page = page_size
        self._extra_tenants = tuple(extra_tenants)

    # ---------------------------------------------------------------- one row
    def _register(self, row: LegacyRow, counts: BackfillCounts, quarantined: list[dict[str, str]]) -> None:
        counts.scanned += 1
        if row.dream_tombstoned:
            counts.dream_tombstoned += 1
        tenant = self._owner_tenant(row.owner_user_id) if row.owner_user_id else None
        if tenant is None:
            counts.quarantined += 1
            quarantined.append({"row_id": row.id, "owner": row.owner_user_id or "", "reason": "owner_not_mapped"})
            return
        if self._ledger.get_tombstone(tenant, row.id) is not None:
            counts.erased_skipped += 1  # never give an erased record a reference back
            return
        revision, latest = self._latest_ref(tenant, row.id)
        checksum = content_checksum(row.content)
        if latest is not None:
            if (revision == 1 or is_backfill_baseline(latest)) and latest.checksum != checksum:
                # the legacy row was edited after a baseline (revision 1 holds only a baseline checksum,
                # whoever registered it): the ledger never saw this content
                self._put(tenant, row, revision + 1, checksum, counts, "rebaselined")
            else:
                counts.already_canonical += 1
            return
        self._put(tenant, row, 1, checksum, counts, "registered")

    def _put(self, tenant: str, row: LegacyRow, revision: int, checksum: str, counts: BackfillCounts, kind: str) -> None:
        ref = ObjectRef(
            tenant_id=tenant,
            record_id=row.id,
            content_revision=revision,
            policy_epoch=0,
            source_refs=(baseline_marker(tenant),),
            checksum=checksum,
        )
        try:
            created = self._ledger.put_object_ref(ref)
        except ObjectRefConflict:
            created = False  # the write path registered it between the read and the put
        except ResurrectionDenied:
            counts.erased_skipped += 1  # erased between the tombstone check and the registration: benign
            return
        if created:
            setattr(counts, kind, getattr(counts, kind) + 1)
        else:
            counts.already_canonical += 1

    def _latest_ref(self, tenant: str, record_id: str) -> tuple[int, ObjectRef | None]:
        revision, latest = 0, None
        while True:
            found = self._ledger.get_object_ref(tenant, record_id, revision + 1)
            if found is None:
                return revision, latest
            revision, latest = revision + 1, found

    # -------------------------------------------------------------------- run
    def _load(self) -> tuple[int, str | None, BackfillCounts, list[dict[str, str]], bool]:
        n, record = self._log.latest()
        if record is None:
            return 0, None, BackfillCounts(), [], False
        counts = BackfillCounts(**record["counts"])
        return n, record.get("cursor"), counts, list(record.get("quarantined", [])), bool(record.get("done"))

    def report(self) -> BackfillReport:
        _, cursor, counts, quarantined, done = self._load()
        return BackfillReport(counts, cursor, done, quarantined)

    def run(self, max_pages: int | None = None) -> BackfillReport:
        """Continue the pass from its checkpoint; start a fresh pass when the last one finished."""

        n, cursor, counts, quarantined, done = self._load()
        if done:
            n, cursor, counts, quarantined, done = n, None, BackfillCounts(), [], False
        pages = 0
        while max_pages is None or pages < max_pages:
            rows, next_cursor = self._scanner.scan(cursor, self._page)
            for row in rows:
                self._register(row, counts, quarantined)
            pages += 1
            cursor, done = next_cursor, next_cursor is None
            n = self._log.append(n, {
                "cursor": cursor,
                "counts": counts.as_dict(),
                "quarantined": quarantined[:QUARANTINE_DETAIL_LIMIT],
                "done": done,
            })
            if done:
                break
        return BackfillReport(counts, cursor, done, quarantined[:QUARANTINE_DETAIL_LIMIT])

    # -------------------------------------------------------------- reconcile
    def reconcile(self) -> ReconcileReport:
        """Re-scan the legacy rows and check each against the ledger (read-only on both sides).

        Beyond existence it compares content: a row whose latest reference is revision 1 or a
        backfill baseline must carry the checksum of the row's current content, otherwise the row was
        changed behind the ledger and is reported as drifted.  Stray references for a record in a
        tenant other than its owner's are found among every tenant the scan attributed rows to (and
        any ``extra_tenants``), so the check needs no list of tenants from the caller.
        """

        report = ReconcileReport()
        seen: set[str] = set()
        tenants: set[str] = set(self._extra_tenants)
        cursor: str | None = None
        while True:
            rows, cursor = self._scanner.scan(cursor, self._page)
            for row in rows:
                report.scanned += 1
                if row.id in seen:
                    report.duplicate_rows.append(row.id)
                    continue
                seen.add(row.id)
                tenant = self._owner_tenant(row.owner_user_id) if row.owner_user_id else None
                if tenant is None:
                    report.quarantined += 1
                    continue
                tenants.add(tenant)
                if self._ledger.get_tombstone(tenant, row.id) is not None:
                    report.erased += 1
                    continue
                revision, latest = self._latest_ref(tenant, row.id)
                if latest is None:
                    report.missing.append(row.id)
                    continue
                report.canonical += 1
                if (revision == 1 or is_backfill_baseline(latest)) and latest.checksum != content_checksum(row.content):
                    report.drifted.append(row.id)
            if cursor is None:
                break
        self._find_stray_references(report, tenants)
        return report

    def _find_stray_references(self, report: ReconcileReport, tenants: set[str]) -> None:
        cursor: str | None = None
        while True:
            rows, cursor = self._scanner.scan(cursor, self._page)
            for row in rows:
                owner = self._owner_tenant(row.owner_user_id) if row.owner_user_id else None
                for other in tenants:
                    if other != owner and self._ledger.has_object_refs(other, row.id) and row.id not in report.wrong_tenant:
                        report.wrong_tenant.append(row.id)  # also catches a reference for an unmapped (quarantined) owner
            if cursor is None:
                break


def digest_of_ids(ids: Iterable[str]) -> str:
    """A stable digest of a set of row ids (for comparing a scan with another system)."""

    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
