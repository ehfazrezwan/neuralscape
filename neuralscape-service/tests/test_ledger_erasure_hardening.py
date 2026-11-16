"""Erasure authority hardening from review: copy-proof authority, issued principals, fail-closed commit, lifetimes."""

from __future__ import annotations

import copy
import dataclasses
import math
import pickle
import time

import pytest

from contracts_policy import PolicyStatement, PrincipalContext
from ledger.erasure import (
    AuthorityStoreVerifier,
    ErasureAuthority,
    erase_owner_override,
    issue_erasure_token,
    mark_verified_principal,
)
from ledger.port import AuthorityDenied, LedgerError
from tests.test_ledger_erasure_contract import RECORD, T, World


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    yield w
    w.close()


def issued(world: World) -> PrincipalContext:
    return mark_verified_principal(world.principal())


def build(world: World) -> ErasureAuthority:
    return ErasureAuthority.from_principal(issued(world), policy_epoch=world.authority.current_policy(T).policy_epoch)


# ---- a genuine authority cannot be re-aimed
def test_a_genuine_authority_cannot_be_copied_replaced_or_unpickled_into_another_identity(world) -> None:
    authority = build(world)
    with pytest.raises(TypeError):
        dataclasses.replace(authority, principal_id="someone-else")  # not a dataclass any more
    for attempt in (copy.copy, copy.deepcopy, lambda a: pickle.loads(pickle.dumps(a))):
        with pytest.raises(TypeError):
            attempt(authority)
    for name in ("principal_id", "credential_id", "tenant_id", "witness_id"):
        with pytest.raises(AttributeError):
            setattr(authority, name, "x")
    with pytest.raises(AttributeError):
        authority._principal_id = "x"
    assert (authority.tenant_id, authority.principal_id) == (T, world.owner)


def test_the_authority_does_not_print_its_witness(world) -> None:
    assert "witness" not in repr(build(world)).lower() and "1:" not in repr(build(world))


# ---- only principals the request layer issued
def test_a_principal_shaped_object_the_enforcement_layer_never_issued_is_refused(world) -> None:
    record = world.authority.subject_record(world.owner)
    forged = PrincipalContext(
        schema_version="candidate-v1", tenant_id=T, subject_id=world.owner, subject_kind="human",
        credential_id="cred-alice", membership_version=record.membership_version, device_id=None, delegation=None,
    )
    with pytest.raises(TypeError, match="issued"):
        ErasureAuthority.from_principal(forged, policy_epoch=1)
    assert ErasureAuthority.from_principal(mark_verified_principal(forged), policy_epoch=1).principal_id == world.owner


def test_an_equal_copy_of_an_issued_principal_is_not_itself_issued(world) -> None:
    original = issued(world)
    twin = original.model_copy()  # equal by value, a different object
    with pytest.raises(TypeError, match="issued"):
        ErasureAuthority.from_principal(twin, policy_epoch=1)


# ---- fail closed without the held commit
class BareVerifier:
    """A verifier that says yes but cannot keep the authority held through the commit."""

    def verify_owner_erasure(self, authority, *, record_id):
        return True


def test_a_verifier_without_a_held_commit_is_refused_and_nothing_is_erased(world) -> None:
    with pytest.raises(AuthorityDenied, match="cannot hold"):
        erase_owner_override(world.ledger, BareVerifier(), build(world), record=RECORD, intent_id="erase-1",
                             idempotency_key="k1", request_digest="d", correlation_id="c")
    assert world.ledger.get_tombstone(T, "rec-1") is None and world.ledger.get_intent(T, "erase-1") is None


def test_a_ledger_fault_during_the_held_commit_is_not_reported_as_a_change_of_authority(world) -> None:
    class Faulty:
        def __getattr__(self, name):
            return getattr(world.ledger, name)

        def erase_record(self, token, intent):
            raise ValueError("ledger exploded: row mismatch")

    with pytest.raises(ValueError, match="ledger exploded"):
        erase_owner_override(Faulty(), world.verifier, build(world), record=RECORD, intent_id="erase-1",
                             idempotency_key="k1", request_digest="d", correlation_id="c")
    with pytest.raises(LedgerError, match="mismatch"):
        class Mismatch(Faulty):
            def erase_record(self, token, intent):
                raise LedgerError("token mismatch")

        erase_owner_override(Mismatch(), world.verifier, build(world), record=RECORD, intent_id="erase-2",
                             idempotency_key="k2", request_digest="d", correlation_id="c")


def test_the_held_check_compares_the_presented_credential_with_the_subjects(world) -> None:
    """Another active subject's credential, presented for this subject, must not run the body."""

    from identity_binding import bind_identity

    bob = bind_identity(world.authority, "local", "bob").subject_id
    world.authority.add_subject(bob, T, "human", "cred-bob")  # active, so only the comparison can refuse it
    record = world.authority.subject_record(world.owner)
    mismatched = mark_verified_principal(PrincipalContext(
        schema_version="candidate-v1", tenant_id=T, subject_id=world.owner, subject_kind="human",
        credential_id="cred-bob", membership_version=record.membership_version, device_id=None, delegation=None,
    ))
    epoch = world.authority.current_policy(T).policy_epoch
    forged = ErasureAuthority.from_principal(mismatched, policy_epoch=epoch)
    ran = []
    with pytest.raises(AuthorityDenied):
        world.verifier.hold(forged, lambda: ran.append("body"))
    assert ran == []
    assert world.verifier.hold(build(world), lambda: ran.append("body")) is None and ran == ["body"]  # the real pair runs


# ---- token lifetime and one clock
@pytest.mark.parametrize("ttl", [float("nan"), float("inf"), -1.0, 0.0, 1e12, 301.0, True])
def test_an_unreasonable_token_lifetime_is_refused(world, ttl) -> None:
    with pytest.raises(ValueError, match="lifetime"):
        issue_erasure_token(world.verifier, build(world), record_id="rec-1", content_revision=5,
                            intent_id="erase-1", ttl_seconds=ttl)


def test_a_token_that_never_expires_cannot_be_minted_through_the_clock(world) -> None:
    with pytest.raises(ValueError):
        issue_erasure_token(world.verifier, build(world), record_id="rec-1", content_revision=5,
                            intent_id="erase-1", now=lambda: float("nan"))


def test_minting_and_the_ledger_use_one_clock(world, monkeypatch) -> None:
    import ledger.port as port

    real = time.time
    monkeypatch.setattr(port, "erasure_now", lambda: real() - 3600)  # the clock both sides read is an hour behind wall time
    authority = build(world)
    token = issue_erasure_token(world.verifier, authority, record_id="rec-1", content_revision=5, intent_id="erase-1")
    assert math.isclose(token.expires_at, real() - 3600 + 30, abs_tol=5)  # minted on that clock (already past in wall time)...
    from ledger.erasure import build_erase_intent

    intent = build_erase_intent(authority, record=RECORD, intent_id="erase-1", idempotency_key="k", request_digest="d",
                                correlation_id="c")
    assert world.ledger.erase_record(token, intent).created  # ...and accepted by the ledger on the same one


# ---- the evaluator sees only the principal's statements
def test_the_policy_check_evaluates_only_the_principals_statements(tmp_path) -> None:
    import contracts_policy_reference as reference

    w = World(tmp_path)
    try:
        noise = [
            PolicyStatement(id=f"noise-{i}", effect="allow", subject_id=f"member-{i}", credential_id=None,
                            action="read", resource=w.pool)
            for i in range(300)
        ]
        w.authority.publish_policy(T, [
            PolicyStatement(id="allow-delete", effect="allow", subject_id=w.owner, credential_id=None,
                            action="delete", resource=w.pool),
            *noise,
        ])
        seen = []
        real = reference.evaluate_policy

        def spy(*, principal, evaluation, policy):
            seen.append(len(policy.statements))
            return real(principal=principal, evaluation=evaluation, policy=policy)

        reference.evaluate_policy = spy
        try:
            assert w.erase().created
        finally:
            reference.evaluate_policy = real
        assert seen and max(seen) == 1  # the other members' statements never reach the evaluator
    finally:
        w.close()
