"""Publication fence: current authority and current source, never the payload."""

from __future__ import annotations

import threading
import time

import pytest

from authority_store_sql import SqliteAuthorityStore, WitnessStale
from identity_binding import NamespaceKey
from policy_enforcement import pool_handle
from publication_fence import (
    Admission,
    CurrentSource,
    PublicationFence,
)
from tests.test_authority_store_sql import Clock, stmt

TENANT = "tenant-a"
PRIVATE = NamespaceKey(TENANT, "private", "sub_a", None, None)
SHARED = NamespaceKey(TENANT, "shared", None, None, None)


class Sources:
    def __init__(self, namespace=PRIVATE, revision=3, visibility=None) -> None:
        self.rows = {"memory-1": (revision, namespace, visibility or namespace.pool_kind)}
        self.calls = 0
        self.on_call = None

    def current(self, tenant_id, record_id):
        self.calls += 1
        if self.on_call:
            self.on_call()
        row = self.rows.get(record_id)
        return None if row is None else CurrentSource(*row)


def handle(namespace: NamespaceKey):
    return pool_handle(*namespace)


@pytest.fixture
def world(tmp_path):
    store = SqliteAuthorityStore(str(tmp_path / "a.db"), clock=Clock())
    store.add_subject("sub_a", TENANT, "human", "cred-a")
    store.add_subject("sub_other", "tenant-b", "human", "cred-other")
    policy = store.publish_policy(TENANT, [stmt("g1", "sub_a", "background_derive", handle(PRIVATE))])
    sources = Sources()
    fence = PublicationFence(store, sources)
    admission = Admission(TENANT, "sub_a", "cred-a", 1, policy.policy_epoch, "memory-1", 3, PRIVATE)
    yield store, sources, fence, admission
    store.close()


def reason(decision):
    return (decision.allowed, decision.reason)


def test_current_authority_and_source_allow(world) -> None:
    store, sources, fence, admission = world
    decision = fence.evaluate(admission)
    assert reason(decision) == (True, "explicit_grant") and not decision.epoch_advanced
    assert decision.witness.policy_epoch == 1 and decision.witness.source_revision == 3
    assert decision.witness.namespace.startswith("ns1")


def test_retired_credential_denies_work_admitted_under_it(world) -> None:
    store, _, fence, admission = world
    store.retire_credential("cred-a")
    assert reason(fence.evaluate(admission)) == (False, "credential_retired")
    # Rotating the subject to a new credential does not revive the old one.
    store2 = store
    store2.add_subject("sub_r", TENANT, "human", "cred-r1")
    store2.update_subject("sub_r", credential_id="cred-r2")
    old = Admission(TENANT, "sub_r", "cred-r1", 2, 1, "memory-1", 3, PRIVATE)
    assert reason(fence.evaluate(old)) == (False, "credential_retired")


def test_offboarded_subject_and_membership_change_deny(world) -> None:
    store, _, fence, admission = world
    store.update_subject("sub_a", subject_kind="service_account")
    assert reason(fence.evaluate(admission)) == (False, "membership_changed")
    store.offboard_subject("sub_a")
    assert reason(fence.evaluate(admission))[0] is False


def test_admission_claims_cannot_substitute_for_current_state(world) -> None:
    store, _, fence, admission = world
    # A forged membership version, a foreign tenant and an unknown subject.
    assert reason(fence.evaluate(admission._replace(membership_version=9))) == (False, "membership_changed")
    assert reason(fence.evaluate(admission._replace(tenant_id="tenant-b"))) == (False, "tenant_mismatch")
    assert reason(fence.evaluate(admission._replace(subject_id="sub_other"))) == (False, "credential_subject_mismatch")
    assert reason(fence.evaluate(admission._replace(subject_id="sub_nobody")))[0] is False
    # A claim of a newer epoch than exists is a regression, not a grant.
    assert reason(fence.evaluate(admission._replace(policy_epoch=99))) == (False, "policy_epoch_regressed")


def test_policy_epoch_advance_is_judged_by_the_current_policy(world) -> None:
    store, _, fence, admission = world
    store.publish_policy(TENANT, [stmt("g1", "sub_a", "background_derive", handle(PRIVATE)),
                                  stmt("g2", "sub_a", "read", handle(PRIVATE))])
    decision = fence.evaluate(admission)
    assert (decision.allowed, decision.epoch_advanced) == (True, True)  # still granted now
    store.publish_policy(TENANT, [stmt("g1", "sub_a", "background_derive", handle(PRIVATE), effect="deny")])
    assert reason(fence.evaluate(admission)) == (False, "not_authorized")
    store.publish_policy(TENANT, [stmt("g1", "sub_a", "background_derive", handle(PRIVATE))])
    assert fence.evaluate(admission).allowed
    strict = PublicationFence(store, world[1], strict_epoch=True)
    assert reason(strict.evaluate(admission)) == (False, "policy_epoch_advanced")


def test_namespace_and_visibility_are_taken_from_the_current_row(world) -> None:
    store, sources, fence, admission = world
    # The memory was made shared after admission: the private target is wrong,
    # even though the payload still names it.
    sources.rows["memory-1"] = (3, SHARED, "shared")
    assert reason(fence.evaluate(admission)) == (False, "namespace_changed")
    # Even when the payload is updated to the new namespace, the grant must
    # exist on the CURRENT pool, not the one the job was admitted for.
    shifted = admission._replace(target_namespace=SHARED)
    assert reason(fence.evaluate(shifted)) == (False, "not_authorized")
    store.publish_policy(TENANT, [stmt("g3", "sub_a", "background_derive", handle(SHARED))])
    fresh = shifted._replace(policy_epoch=2)
    assert fence.evaluate(fresh).allowed
    # An inconsistent row (visibility says private, namespace says shared) denies.
    sources.rows["memory-1"] = (3, SHARED, "private")
    assert reason(fence.evaluate(fresh)) == (False, "visibility_mismatch")


def test_source_revision_and_disappearance_deny(world) -> None:
    _, sources, fence, admission = world
    sources.rows["memory-1"] = (4, PRIVATE, "private")
    assert reason(fence.evaluate(admission)) == (False, "source_revision_changed")
    del sources.rows["memory-1"]
    assert reason(fence.evaluate(admission)) == (False, "source_gone")


def test_publish_runs_the_write_inside_the_verified_transaction(world) -> None:
    store, sources, fence, admission = world
    with store._txn() as c:
        store._x(c, "CREATE TABLE published(v TEXT)")
    decision, result = fence.publish(
        admission, lambda c: store._x(c, "INSERT INTO published VALUES('derived')") and "ok"
    )
    assert decision.allowed and result == "ok"
    with store._read_txn() as c:
        assert store._x(c, "SELECT COUNT(*) FROM published").fetchone()[0] == 1
    # A failing write rolls back and propagates.
    with pytest.raises(RuntimeError):
        def boom(c):
            store._x(c, "INSERT INTO published VALUES('lost')")
            raise RuntimeError("write failed")
        fence.publish(admission, boom)
    with store._read_txn() as c:
        assert store._x(c, "SELECT COUNT(*) FROM published").fetchone()[0] == 1


def test_denied_publication_never_calls_the_write(world) -> None:
    store, _, fence, admission = world
    store.retire_credential("cred-a")
    called = []
    decision, result = fence.publish(admission, lambda c: called.append(1))
    assert (decision.allowed, decision.reason, result, called) == (False, "credential_retired", None, [])


def test_revocation_between_evaluate_and_hold_is_caught_atomically(world) -> None:
    store, sources, fence, admission = world
    # Retire the credential after the pre-check has passed, before the hold.
    state = {"n": 0}

    def revoke_after_first_resolution():
        state["n"] += 1
        if state["n"] == 1:
            store.retire_credential("cred-a")

    sources.on_call = revoke_after_first_resolution
    called = []
    decision, result = fence.publish(admission, lambda c: called.append(1))
    assert (decision.allowed, decision.reason, result, called) == (False, "credential_retired", None, [])


def test_source_change_inside_the_hold_is_caught(world) -> None:
    store, sources, fence, admission = world
    state = {"n": 0}

    def change_on_second_resolution():
        state["n"] += 1
        if state["n"] == 2:
            sources.rows["memory-1"] = (4, PRIVATE, "private")

    sources.on_call = change_on_second_resolution
    called = []
    decision, result = fence.publish(admission, lambda c: called.append(1))
    assert (decision.allowed, decision.reason, result, called) == (False, "source_changed", None, [])


def test_concurrent_retirement_waits_for_a_held_publication(tmp_path) -> None:
    path = str(tmp_path / "shared.db")
    publisher, revoker = SqliteAuthorityStore(path), SqliteAuthorityStore(path)
    publisher.add_subject("sub_a", TENANT, "human", "cred-a")
    policy = publisher.publish_policy(TENANT, [stmt("g1", "sub_a", "background_derive", handle(PRIVATE))])
    fence = PublicationFence(publisher, Sources())
    admission = Admission(TENANT, "sub_a", "cred-a", 1, policy.policy_epoch, "memory-1", 3, PRIVATE)
    inside, done = threading.Event(), threading.Event()

    def revoke():
        inside.wait()
        revoker.retire_credential("cred-a")
        done.set()

    thread = threading.Thread(target=revoke)
    thread.start()

    def write(connection):
        inside.set()
        time.sleep(0.4)
        assert not done.is_set(), "retirement committed while publication was held"
        return "published"

    decision, result = fence.publish(admission, write)
    thread.join(10)
    assert (decision.allowed, result, done.is_set()) == (True, "published", True)
    # After the retirement the next publication is denied.
    assert fence.publish(admission, lambda c: "again")[0].reason == "credential_retired"
    publisher.close()
    revoker.close()


def test_witness_stale_is_the_only_stale_signal() -> None:
    assert issubclass(WitnessStale, Exception) and WitnessStale("x").reason == "x"
