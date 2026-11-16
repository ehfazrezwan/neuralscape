"""Holds, owner-override erasure, audit trail and erased-reference resolution."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Callable

import pytest

from contracts_lifecycle import IntentStatus, ProcessingStage, StageStatus
from contracts_references import ReferenceHandle, SourceVersion
from ledger.dispatch import Applied, StageContext, StageFailure, StageWorker
from ledger.erasure import (
    _PROOF as ERASURE_PROOF,
    ErasureAuthority,
    erase_owner_override,
    guard_ordinary_delete,
    issue_erasure_token,
    resolve_reference,
)
from ledger.conformance import make_intent, ref
from ledger.port import (
    AuthorityDenied,
    HoldActive,
    IdempotencyConflict,
    LedgerError,
    ResurrectionDenied,
    VerifiedErasure,
    snapshot_digest,
)
from ledger.stages import stage_set
from tests.test_ledger_dispatch import External, Handler, SqliteStore, all_handlers

C, V, G, M = ProcessingStage.CANONICAL, ProcessingStage.VECTOR, ProcessingStage.GRAPH, ProcessingStage.MARKDOWN
T = "tenant-1"
RECORD = SourceVersion(record_id="rec-1", content_revision=5, policy_epoch=2)


class Owner:
    """Fake authority store: allows exactly the registered (principal, record, witness)."""

    def __init__(self, *allowed: tuple[str, str, str], raises: bool = False) -> None:
        self.allowed, self.raises, self.calls = set(allowed), raises, 0

    def verify_owner_erasure(self, authority: ErasureAuthority, *, record_id: str) -> bool:
        self.calls += 1
        if self.raises:
            raise RuntimeError("authority store unreachable: secret-detail")
        return (authority.principal_id, record_id, authority.witness_id) in self.allowed

    def hold(self, authority, body):  # a fake verifier that "holds" trivially: it has no store to lock
        return body()


def authority(principal: str = "owner-1", witness: str = "w-1", tenant: str = T) -> ErasureAuthority:
    return ErasureAuthority(tenant, principal, "cred-1", witness, ERASURE_PROOF)  # arbitrary witnesses probe the verifier


def erase(ledger, verifier, *, key="k1", digest="sha256:e", intent_id="erase-1", who=None, record=RECORD):
    return erase_owner_override(
        ledger, verifier, who or authority(), record=record, intent_id=intent_id,
        idempotency_key=key, request_digest=digest, correlation_id="corr-e",
    )


def good_owner() -> Owner:
    return Owner(("owner-1", "rec-1", "w-1"))


class SqliteErasureStore(SqliteStore):
    def sql_error(self, sql: str) -> BaseException | None:
        raw = sqlite3.connect(self.path, isolation_level=None)
        try:
            raw.execute(sql)
            return None
        except sqlite3.Error as exc:
            return exc
        finally:
            raw.close()


# --------------------------------------------------------------------------- scenarios
def scenario_holds_block_ordinary_delete(store) -> None:
    ledger = store.open()
    ledger.accept(make_intent("a"), [ref("rec-1")])
    assert ledger.place_hold(T, "hold-1", "rec-1", "legal", "counsel", "cred-c") is True
    assert ledger.place_hold(T, "hold-1", "rec-1", "legal", "counsel", "cred-c") is False  # idempotent
    with pytest.raises(LedgerError):
        ledger.place_hold(T, "hold-1", "rec-1", "other-reason", "counsel", "cred-c")
    ledger.place_hold(T, "hold-2", "rec-1", "audit", "auditor", "cred-a")
    assert [h.hold_id for h in ledger.active_holds(T, "rec-1")] == ["hold-1", "hold-2"]
    assert ledger.active_holds("tenant-2", "rec-1") == ()
    with pytest.raises(HoldActive) as blocked:
        guard_ordinary_delete(ledger, T, "rec-1")
    assert blocked.value.hold_ids == ("hold-1", "hold-2")
    with pytest.raises(HoldActive):  # the ledger repeats the check inside its transaction
        ledger.tombstone(T, "rec-1", 5, "admin_delete")
    assert ledger.get_tombstone(T, "rec-1") is None and ledger.get_object_ref(T, "rec-1", 1) is not None
    assert ledger.release_hold(T, "hold-1", "counsel", "case closed") is True
    assert ledger.release_hold(T, "hold-1", "counsel", "case closed") is False
    with pytest.raises(HoldActive):
        ledger.tombstone(T, "rec-1", 5, "admin_delete")  # hold-2 still active
    ledger.release_hold(T, "hold-2", "auditor", "done")
    assert ledger.tombstone(T, "rec-1", 5, "admin_delete") is True
    released = ledger.get_hold(T, "hold-1")
    assert released.released_by == "counsel" and not released.active
    with pytest.raises(ResurrectionDenied):
        ledger.place_hold(T, "hold-3", "rec-1", "late", "x", "y")
    with pytest.raises(LedgerError):
        ledger.release_hold(T, "no-such-hold", "x", "y")
    with pytest.raises(ValueError):  # owner-erasure reasons are reserved for the audited path
        ledger.tombstone(T, "rec-2", 1, "owner_erasure")


def scenario_owner_override_erases_held_record_with_audit(store) -> None:
    ledger = store.open()
    ledger.accept(make_intent("a"), [ref("rec-1", sources=(ReferenceHandle(kind="source", id="s", tenant_id=T, resolver="r"),))])
    ledger.place_hold(T, "hold-1", "rec-1", "legal", "counsel", "cred-c")
    ledger.place_hold(T, "hold-2", "rec-1", "audit", "auditor", "cred-a")
    verifier = good_owner()
    result = erase(ledger, verifier)
    assert result.created and result.overridden_hold_ids == ("hold-1", "hold-2") and verifier.calls == 1
    tomb = ledger.get_tombstone(T, "rec-1")
    assert tomb.reason_code == "owner_erasure_hold_override" and tomb.content_revision == 5
    assert ledger.get_object_ref(T, "rec-1", 1) is None and not ledger.has_object_refs(T, "rec-1")
    assert ledger.active_holds(T, "rec-1") == ()
    assert ledger.get_hold(T, "hold-1").release_reason == "erased"
    (audit,) = ledger.audit_records(T, "rec-1")
    assert (audit.actor_id, audit.credential_id, audit.authority_witness_id) == ("owner-1", "cred-1", "w-1")
    assert audit.overridden_hold_ids == ("hold-1", "hold-2") and audit.intent_id == "erase-1"
    assert audit.reason_code == "owner_erasure_hold_override" and audit.recorded_at.year >= 2026
    assert audit.content_revision == 5 and audit.record_id == "rec-1"
    # the audit is content-free: only ids, hashes-free metadata and times
    assert set(vars(audit)) == {
        "tenant_id", "audit_id", "record_id", "content_revision", "actor_id", "credential_id",
        "authority_witness_id", "overridden_hold_ids", "reason_code", "intent_id", "recorded_at",
    }
    stored = ledger.get_intent(T, "erase-1")
    assert stored.status is IntentStatus.PROCESSING  # canonical applied; projections pending
    receipts = {r.stage: r.status for r in ledger.receipts(T, "erase-1")}
    assert receipts == {C: StageStatus.APPLIED}
    assert [s.stage for s in stored.intent.stage_requirements] == [C, V, G, M]  # every projection required
    with pytest.raises(ResurrectionDenied):
        ledger.put_object_ref(ref("rec-1", 6))
    assert ledger.audit_records("tenant-2") == ()


def scenario_erasure_without_holds_and_over_existing_tombstone(store) -> None:
    ledger = store.open()
    ledger.accept(make_intent("a"), [ref("rec-1")])
    first = erase(ledger, good_owner())
    assert first.overridden_hold_ids == () and ledger.get_tombstone(T, "rec-1").reason_code == "owner_erasure"
    # an ordinary delete fenced the record earlier: the owner erasure still audits and keeps the fence
    ledger.tombstone(T, "rec-2", 1, "admin_delete")
    other = SourceVersion(record_id="rec-2", content_revision=1, policy_epoch=1)
    erase(ledger, Owner(("owner-1", "rec-2", "w-1")), key="k2", intent_id="erase-2", record=other)
    assert ledger.get_tombstone(T, "rec-2").reason_code == "admin_delete"
    assert [a.record_id for a in ledger.audit_records(T)] == ["erase:erase-1" and "rec-1", "rec-2"]


def scenario_authority_is_required_and_fails_closed(store) -> None:
    ledger = store.open()
    ledger.accept(make_intent("a"), [ref("rec-1")])
    ledger.place_hold(T, "hold-1", "rec-1", "legal", "counsel", "cred-c")
    for verifier, who in (
        (Owner(), authority()),  # not the owner
        (Owner(("owner-1", "rec-1", "other-witness")), authority()),  # wrong witness
        (Owner(("owner-1", "rec-1", "w-1"), raises=True), authority()),  # authority store down
        (good_owner(), authority(witness="")),  # missing witness
    ):
        with pytest.raises(AuthorityDenied) as denied:
            erase(ledger, verifier, who=who)
        assert "secret-detail" not in str(denied.value)
    forged_token = VerifiedErasure(T, "rec-1", 5, "owner-1", "cred-1", "w-1", "owner_erasure")  # no MAC
    from ledger.erasure import build_erase_intent as _build

    with pytest.raises(AuthorityDenied):  # a token cannot be forged by constructing it
        ledger.erase_record(forged_token, _build(authority(), record=RECORD, intent_id="forged", idempotency_key="kf",
                                                 request_digest="d", correlation_id="c"))
    assert ledger.get_tombstone(T, "rec-1") is None and ledger.audit_records(T) == ()
    assert ledger.get_intent(T, "erase-1") is None and ledger.active_holds(T, "rec-1")
    # a verified token must describe the intent it is used with
    token = issue_erasure_token(good_owner(), authority(), record_id="rec-1", content_revision=5, intent_id="x")
    from ledger.erasure import build_erase_intent

    mismatch = build_erase_intent(authority(principal="intruder"), record=RECORD, intent_id="x", idempotency_key="kx",
                                  request_digest="d", correlation_id="c")
    with pytest.raises(LedgerError):
        ledger.erase_record(token, mismatch)
    assert ledger.get_tombstone(T, "rec-1") is None


def scenario_erasure_is_idempotent_and_atomic(store) -> None:
    ledger = store.open()
    ledger.accept(make_intent("a"), [ref("rec-1")])
    ledger.place_hold(T, "hold-1", "rec-1", "legal", "counsel", "cred-c")
    first = erase(ledger, good_owner())
    again = erase(ledger, good_owner(), intent_id="erase-retry")  # same key + digest
    assert not again.created and again.overridden_hold_ids == first.overridden_hold_ids == ("hold-1",)
    assert len(ledger.audit_records(T)) == 1 and ledger.get_intent(T, "erase-retry") is None
    with pytest.raises(IdempotencyConflict):
        erase(ledger, good_owner(), digest="sha256:other")
    # concurrent identical erasures: exactly one creator, one audit row
    results: list = []
    stores = [store.open() for _ in range(6)]
    ledger2 = store.open()
    ledger2.accept(make_intent("b"), [ref("rec-9")])
    nine = SourceVersion(record_id="rec-9", content_revision=1, policy_epoch=1)

    def worker(conn) -> None:
        try:
            results.append(erase(conn, Owner(("owner-1", "rec-9", "w-1")), key="race", intent_id="erase-race", record=nine))
        except BaseException as exc:  # noqa: BLE001
            results.append(exc)

    threads = [threading.Thread(target=worker, args=(c,)) for c in stores]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert all(not isinstance(r, BaseException) for r in results), results
    assert sum(r.created for r in results) == 1
    assert len(ledger2.audit_records(T, "rec-9")) == 1


def scenario_audit_is_append_only(store) -> None:
    ledger = store.open()
    ledger.accept(make_intent("a"), [ref("rec-1")])
    erase(ledger, good_owner())
    for statement in (
        "UPDATE erasure_audit SET actor_id='forged'",
        "DELETE FROM erasure_audit",
    ):
        error = store.sql_error(statement)  # even the strongest (owner) connection is refused
        assert error is not None and "append-only" in str(error), (statement, error)
    assert ledger.audit_records(T)[0].actor_id == "owner-1"


def scenario_projections_follow_through_the_dispatcher(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.put_object_ref(ref("rec-1"))
    erase(ledger, good_owner())
    assert ledger.get_intent(T, "erase-1").status is IntentStatus.PROCESSING
    result = StageWorker(store.open(), all_handlers(external), owner="w").run_once()
    assert result.intent_id == "erase-1" and result.status is IntentStatus.APPLIED
    assert result.outcomes[C] == "applied (earlier attempt)"  # the ledger transaction was the canonical effect
    assert external.effects() == 3 and external.calls("canonical") == 0  # vector, graph, markdown only
    assert {r.stage for r in ledger.receipts(T, "erase-1") if r.status is StageStatus.APPLIED} == {C, V, G, M}


def scenario_failed_projection_never_reports_applied_but_keeps_the_fence(store) -> None:
    ledger, external = store.open(), store.external()
    ledger.put_object_ref(ref("rec-1"))
    erase(ledger, good_owner())

    def down(ctx: StageContext) -> None:
        if ctx.stage is V:
            raise StageFailure("vector_down", "vector store unavailable")

    handlers = all_handlers(external, **{V: Handler(external, hook=down)})
    result = StageWorker(store.open(), handlers, owner="w").run_once()
    assert result.status is IntentStatus.PARTIAL  # canonical applied, vector failed: honest, not APPLIED
    assert ledger.get_tombstone(T, "rec-1") is not None
    with pytest.raises(ResurrectionDenied):
        ledger.put_object_ref(ref("rec-1", 9))


def scenario_references_resolve_to_erased_never_content(store) -> None:
    ledger = store.open()
    ledger.accept(make_intent("a"), [ref("rec-1")])
    handle = ReferenceHandle(kind="memory", id="rec-1", tenant_id=T, resolver="resolve_memory")
    seen: list = []

    def allow(h: ReferenceHandle) -> bool:
        seen.append(h)
        return True

    assert resolve_reference(ledger, handle, allow).state == "present"
    erase(ledger, good_owner())
    resolved = resolve_reference(ledger, handle, allow)
    assert resolved.state == "erased" and set(vars(resolved)) == {"state", "handle"}  # no content field
    assert resolve_reference(ledger, handle, lambda h: False).state == "denied"  # nothing leaks without authorization
    assert resolve_reference(ledger, handle, lambda h: (_ for _ in ()).throw(RuntimeError("x"))).state == "denied"
    assert resolve_reference(ledger, handle, lambda h: "yes").state == "denied"  # only a real True authorizes
    other = ReferenceHandle(kind="memory", id="rec-404", tenant_id=T, resolver="resolve_memory")
    assert resolve_reference(ledger, other, allow).state == "unknown"
    assert resolve_reference(ledger, ReferenceHandle(kind="source", id="rec-1", tenant_id=T, resolver="r"), allow).state == "unknown"
    before = len(seen)
    resolve_reference(ledger, handle, allow)
    resolve_reference(ledger, handle, allow)
    assert len(seen) == before + 2  # authorization is evaluated on every resolution
    cross = ReferenceHandle(kind="memory", id="rec-1", tenant_id=T, resolver="x")
    assert resolve_reference(store.open(), cross, lambda h: h.tenant_id == "tenant-2").state == "denied"


def scenario_restore_after_erasure_keeps_the_fence_and_checks_coherence(store) -> None:
    ledger = store.open()
    ledger.accept(make_intent("a"), [ref("rec-1")])
    ledger.place_hold(T, "hold-1", "rec-1", "legal", "counsel", "cred-c")
    ledger.place_hold(T, "hold-keep", "rec-other", "legal", "counsel", "cred-c")
    erase(ledger, good_owner())
    good = ledger.snapshot()
    assert good["counts"]["erasure_audit"] == 1 and good["counts"]["retention_holds"] == 2
    json.dumps(good)

    target_store = store.fresh()
    target_store.open_admin().restore(json.loads(json.dumps(good)), expected_digest=good["digest"])
    target = target_store.open()
    assert target.get_tombstone(T, "rec-1").reason_code == "owner_erasure_hold_override"
    assert [a.overridden_hold_ids for a in target.audit_records(T)] == [("hold-1",)]
    assert [h.hold_id for h in target.active_holds(T, "rec-other")] == ["hold-keep"]
    with pytest.raises(ResurrectionDenied):
        target.put_object_ref(ref("rec-1", 7))
    assert not erase(target, good_owner()).created  # replay after restore does nothing twice

    def rejected(mutate: Callable[[dict], None], message: str) -> None:
        broken = json.loads(json.dumps(good))
        mutate(broken)
        broken["counts"] = {t: len(rows) for t, rows in broken["tables"].items()}
        broken["digest"] = snapshot_digest(broken)  # forger recomputes: only coherence checks can object
        with pytest.raises(LedgerError, match=message):
            store.fresh().open_admin().restore(broken)

    rejected(lambda b: b["tables"]["tombstones"].clear(), "lost its tombstone")
    rejected(lambda b: b["tables"]["erasure_audit"].clear(), "no audit record")
    rejected(lambda b: b["tables"]["retention_holds"][0].update(released_at=None, release_reason=None, released_by=None),
             "active hold on an erased record")
    rejected(lambda b: b["tables"]["erasure_audit"][0].update(intent_id="intent-1"), "audit must reference|erase intent")
    rejected(lambda b: b["tables"]["erasure_audit"][0].update(overridden_hold_ids="{}"), "malformed")
    rejected(lambda b: b["tables"]["tombstones"].append(
        {"tenant_id": T, "record_id": "rec-x", "content_revision": 1, "reason_code": "owner_erasure_hold_override",
         "recorded_at": 1.0}), "no audit record")


SCENARIOS = {name.removeprefix("scenario_"): fn for name, fn in sorted(globals().items()) if name.startswith("scenario_")}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_erasure_scenarios_on_sqlite(name: str, tmp_path: Path) -> None:
    class Store(SqliteErasureStore):
        def fresh(self) -> "Store":
            other = Store(self.root / f"fresh{len(list(self.root.glob('fresh*')))}")
            other.root.mkdir()
            return other

        def external(self) -> External:
            return External(self.root / "external.db")

    store = Store(tmp_path)
    try:
        SCENARIOS[name](store)
    finally:
        store.close()


def test_stage_registry_has_an_erase_set_with_every_projection() -> None:
    erase_set = stage_set("memory.erase")
    assert [s.stage for s in erase_set.stages] == [C, V, G, M]
    assert {s.stage for s in erase_set.stages if s.optional} == {G, M}
    assert erase_set.stages[1].effect_kind == "vector_delete" and erase_set.stages[2].effect_kind == "graph_remove"


# ----------------------------------------------------- review fixes (H-B2-1..H-B2-5)
def test_the_erasure_token_cannot_be_retargeted_or_forged(tmp_path: Path) -> None:
    """H-B2-2: ``dataclasses.replace`` kept the issuer key, so a token minted for record A
    became a valid token for record B; the module-level key also let any code mint one."""

    import dataclasses

    store = SqliteErasureStore(tmp_path)
    ledger = store.open()
    ledger.accept(make_intent("a"), [ref("rec-1"), ref("rec-2")])
    token = issue_erasure_token(good_owner(), authority(), record_id="rec-1", content_revision=5, intent_id="erase-1")
    from ledger.erasure import build_erase_intent

    other = SourceVersion(record_id="rec-2", content_revision=5, policy_epoch=2)
    retargeted = dataclasses.replace(token, record_id="rec-2")
    intent_for_b = build_erase_intent(authority(), record=other, intent_id="erase-b", idempotency_key="kb",
                                      request_digest="d", correlation_id="c")
    with pytest.raises(AuthorityDenied):
        ledger.erase_record(retargeted, intent_for_b)
    for forged in (
        dataclasses.replace(token, actor_id="evil"),
        dataclasses.replace(token, authority_witness_id="other-witness"),
        dataclasses.replace(token, reason_code="owner_erasure_hold_override"),
    ):
        with pytest.raises(AuthorityDenied):
            ledger.erase_record(forged, build_erase_intent(authority(), record=RECORD, intent_id="x", idempotency_key="kx",
                                                           request_digest="d", correlation_id="c"))
    assert ledger.get_tombstone(T, "rec-2") is None and ledger.get_tombstone(T, "rec-1") is None
    assert ledger.audit_records(T) == ()
    # the genuine token still works for exactly its own record
    genuine = build_erase_intent(authority(), record=RECORD, intent_id="erase-1", idempotency_key="k1",
                                 request_digest="sha256:e", correlation_id="corr-e")
    assert ledger.erase_record(token, genuine).created
    store.close()


def test_the_authority_store_verifier_requires_a_current_bound_human_owner(tmp_path: Path) -> None:
    from authority_store_sql import SqliteAuthorityStore
    from identity_binding import bind_identity
    from ledger.erasure import AuthorityStoreVerifier

    authority_store = SqliteAuthorityStore(str(tmp_path / "authority.db"))
    ids = {}
    for name, kind in (("alice", "human"), ("bob", "human"), ("bot", "worker")):
        subject = bind_identity(authority_store, "local", name).subject_id
        authority_store.add_subject(subject, "tenant-1", kind, f"cred-{name}")
        ids[name] = subject
    owners = {"rec-1": ids["alice"]}
    verifier = AuthorityStoreVerifier(
        authority_store, owner_of=lambda tenant, record: owners.get(record), policy_independent=True
    )

    def auth(name, witness=None, tenant="tenant-1", credential=None):
        record = authority_store.subject_record(ids[name])
        epoch = authority_store.current_policy(tenant).policy_epoch if authority_store.current_policy(tenant) else 0
        return ErasureAuthority(tenant, ids[name], credential or f"cred-{name}",
                                witness or f"{record.membership_version}:{epoch}", ERASURE_PROOF)

    authority_store.publish_policy("tenant-1", [])
    assert verifier.verify_owner_erasure(auth("alice"), record_id="rec-1") is True
    assert verifier.verify_owner_erasure(auth("bob"), record_id="rec-1") is False  # not the owner
    assert verifier.verify_owner_erasure(auth("bot"), record_id="rec-1") is False  # a worker is not a human owner
    assert verifier.verify_owner_erasure(auth("alice"), record_id="rec-nobody") is False  # unknown owner: fail closed
    assert verifier.verify_owner_erasure(auth("alice", credential="cred-bob"), record_id="rec-1") is False
    assert verifier.verify_owner_erasure(auth("alice", witness="1:999"), record_id="rec-1") is False  # stale witness
    assert verifier.verify_owner_erasure(auth("alice", tenant="tenant-2"), record_id="rec-1") is False
    assert AuthorityStoreVerifier(authority_store, owner_of=None, policy_independent=True).verify_owner_erasure(auth("alice"), record_id="rec-1") is False
    current = auth("alice")
    authority_store.retire_credential("cred-alice")
    assert verifier.verify_owner_erasure(current, record_id="rec-1") is False  # retired credential
    authority_store.close()


def test_work_accepted_before_an_erasure_does_not_run_after_it(tmp_path: Path) -> None:
    """H-B2-3: an edit accepted for revision 2 before the erasure used to re-create content afterwards."""

    from ledger.stages import build_intent

    store = SqliteErasureStore(tmp_path)
    ledger, external = store.open(), store.external()
    edit = build_intent(
        "memory.edit", intent_id="edit-1", tenant_id=T, actor_id="actor-1", credential_id="cred-1",
        idempotency_key="ke", request_digest="d", correlation_id="c",
        preconditions=(SourceVersion(record_id="rec-1", content_revision=6, policy_epoch=2),),
    )
    ledger.accept(edit, [ref("rec-1")])
    erase(ledger, good_owner())

    result = StageWorker(store.open(), all_handlers(external), owner="late").run_once()
    while result is not None and result.intent_id != "edit-1":
        result = StageWorker(store.open(), all_handlers(external), owner="late").run_once()

    assert result is not None and result.status is IntentStatus.FAILED
    assert result.outcomes[C] == "failed (record_erased)"
    assert external.effects() == 0  # nothing was re-created for the erased record
    store.close()


@pytest.mark.parametrize("reason", ["OWNER_ERASURE", " owner_erasure", "owner-erasure", "Owner_Erasure_x", "ＯＷＮＥＲ_ERASURE"])
def test_reserved_reason_spellings_cannot_be_used_for_an_ordinary_tombstone(tmp_path: Path, reason) -> None:
    store = SqliteErasureStore(tmp_path)
    ledger = store.open()
    with pytest.raises(ValueError):
        ledger.tombstone(T, "rec-x", 1, reason)
    assert ledger.get_tombstone(T, "rec-x") is None
    store.close()
