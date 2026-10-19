"""Erasure authority contract: who may build it, whether policy applies, and what binds the token."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from authority_store_sql import SqliteAuthorityStore
from contracts_lifecycle import IntentStatus
from contracts_policy import PolicyStatement, PrincipalContext
from contracts_references import SourceVersion
from identity_binding import bind_identity
from ledger.conformance import make_intent, ref
from ledger.erasure import (
    AuthorityStoreVerifier,
    ErasureAuthority,
    build_erase_intent,
    erase_owner_override,
    issue_erasure_token,
    mark_verified_principal,
)
from ledger.port import AuthorityDenied, LedgerError
from policy_enforcement import pool_handle
from tests.test_ledger_dispatch import SqliteStore

T = "tenant-1"
RECORD = SourceVersion(record_id="rec-1", content_revision=5, policy_epoch=2)


class World:
    """A real SQLite authority store with one human owner, a policy snapshot and a ledger holding the record."""

    def __init__(self, root: Path, *, statements=None, policy_independent=False, with_resource=True) -> None:
        self.authority = SqliteAuthorityStore(str(root / "authority.db"))
        self.owner = bind_identity(self.authority, "local", "alice").subject_id
        self.authority.add_subject(self.owner, T, "human", "cred-alice")
        self.pool = pool_handle(T, "private", self.owner)
        allow = [
            PolicyStatement(id=f"allow-{action}", effect="allow", subject_id=self.owner, credential_id=None,
                            action=action, resource=self.pool)
            for action in ("read", "delete")
        ]
        self.authority.publish_policy(T, allow + list(statements or []))
        self.store = SqliteStore(root)
        self.ledger = self.store.open()
        self.ledger.accept(make_intent("a"), [ref("rec-1")])
        self.verifier = AuthorityStoreVerifier(
            self.authority, owner_of=lambda tenant, record: self.owner if record == "rec-1" else None,
            resource_of=(lambda tenant, record: self.pool) if with_resource else None,
            policy_independent=policy_independent,
        )

    def principal(self) -> PrincipalContext:
        record = self.authority.subject_record(self.owner)
        return mark_verified_principal(PrincipalContext(  # the request layer issues the principal
            schema_version="candidate-v1", tenant_id=T, subject_id=self.owner, subject_kind="human",
            credential_id="cred-alice", membership_version=record.membership_version, device_id=None, delegation=None,
        ))

    def authority_for(self) -> ErasureAuthority:
        return ErasureAuthority.from_principal(self.principal(), policy_epoch=self.authority.current_policy(T).policy_epoch)

    def erase(self, **kw):
        return erase_owner_override(
            self.ledger, self.verifier, self.authority_for(), record=kw.pop("record", RECORD), intent_id="erase-1",
            idempotency_key="k1", request_digest="sha256:e", correlation_id="c",
        )

    def close(self) -> None:
        self.authority.close()
        self.store.close()


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    yield w
    w.close()


# ---- who may build an authority
def test_an_authority_cannot_be_built_from_identifiers() -> None:
    with pytest.raises(TypeError, match="from_principal"):
        ErasureAuthority(T, "owner", "cred", "1:1")  # strings from a body, header or query prove nothing
    with pytest.raises(TypeError, match="verified PrincipalContext"):
        ErasureAuthority.from_principal({"tenant_id": T, "subject_id": "owner"}, policy_epoch=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ErasureAuthority.from_principal(object(), policy_epoch=1)  # type: ignore[arg-type]


def test_an_authority_built_from_a_verified_principal_carries_its_witness(world) -> None:
    authority = world.authority_for()
    assert (authority.tenant_id, authority.principal_id, authority.credential_id) == (T, world.owner, "cred-alice")
    epoch = world.authority.current_policy(T).policy_epoch
    assert authority.witness_id == f"{world.principal().membership_version}:{epoch}"
    assert world.erase().created


def test_the_witness_proves_currency_not_identity(world) -> None:
    """A principal forged with the right public ids but another credential is refused: the witness alone is not proof."""

    record = world.authority.subject_record(world.owner)
    forged = PrincipalContext(
        schema_version="candidate-v1", tenant_id=T, subject_id=world.owner, subject_kind="human",
        credential_id="cred-someone-else", membership_version=record.membership_version, device_id=None, delegation=None,
    )
    authority = ErasureAuthority.from_principal(
        mark_verified_principal(forged), policy_epoch=world.authority.current_policy(T).policy_epoch
    )
    with pytest.raises(AuthorityDenied):
        erase_owner_override(world.ledger, world.verifier, authority, record=RECORD, intent_id="erase-1",
                             idempotency_key="k1", request_digest="d", correlation_id="c")
    assert world.ledger.get_tombstone(T, "rec-1") is None


# ---- policy applies to owner erasure
def test_an_explicit_deny_on_delete_stops_the_owner_erasure_by_default(tmp_path) -> None:
    w = World(tmp_path)
    try:
        w.authority.publish_policy(T, [
            PolicyStatement(id="allow-delete", effect="allow", subject_id=w.owner, credential_id=None, action="delete",
                            resource=w.pool),
            PolicyStatement(id="suspend", effect="deny", subject_id=w.owner, credential_id=None, action="delete",
                            resource=w.pool),
        ])
        with pytest.raises(AuthorityDenied):
            w.erase()
        assert w.ledger.get_tombstone(T, "rec-1") is None
    finally:
        w.close()


def test_without_any_delete_grant_policy_mode_refuses(tmp_path) -> None:
    w = World(tmp_path)
    try:
        w.authority.publish_policy(T, [])  # every grant revoked
        with pytest.raises(AuthorityDenied):
            w.erase()
    finally:
        w.close()


def test_a_policy_independent_deployment_must_say_so_and_then_policy_is_not_consulted(tmp_path) -> None:
    w = World(tmp_path, policy_independent=True)
    try:
        w.authority.publish_policy(T, [])  # no grants at all: an owner right independent of administrator policy
        assert w.erase().created
    finally:
        w.close()


def test_policy_mode_without_a_resource_resolver_is_refused_at_construction(tmp_path) -> None:
    authority = SqliteAuthorityStore(str(tmp_path / "a.db"))
    try:
        with pytest.raises(ValueError, match="resource_of"):
            AuthorityStoreVerifier(authority, owner_of=lambda tenant, record: None)
    finally:
        authority.close()


# ---- check and commit are one held step
def test_a_credential_retired_between_the_check_and_the_commit_stops_the_erasure(world) -> None:
    """The verifier's reads are a pre-check; the commit runs under the held authority."""

    original = world.verifier.verify_owner_erasure

    def check_then_revoke(authority, *, record_id):
        allowed = original(authority, record_id=record_id)
        world.authority.retire_credential("cred-alice")  # lands after the check, before the ledger commit
        return allowed

    world.verifier.verify_owner_erasure = check_then_revoke  # type: ignore[method-assign]
    with pytest.raises(AuthorityDenied, match="changed before the erasure committed"):
        world.erase()
    assert world.ledger.get_tombstone(T, "rec-1") is None and world.ledger.audit_records(T) == ()
    assert world.ledger.get_intent(T, "erase-1") is None  # nothing was committed


def test_a_policy_published_between_the_check_and_the_commit_stops_the_erasure(world) -> None:
    original = world.verifier.verify_owner_erasure

    def check_then_publish(authority, *, record_id):
        allowed = original(authority, record_id=record_id)
        world.authority.publish_policy(T, [])  # the epoch advances after the check
        return allowed

    world.verifier.verify_owner_erasure = check_then_publish  # type: ignore[method-assign]
    with pytest.raises(AuthorityDenied):
        world.erase()
    assert world.ledger.get_tombstone(T, "rec-1") is None


def test_a_token_is_bound_to_one_intent_and_expires(world) -> None:
    authority = world.authority_for()
    token = issue_erasure_token(world.verifier, authority, record_id="rec-1", content_revision=5, intent_id="erase-1")
    other = build_erase_intent(authority, record=RECORD, intent_id="erase-2", idempotency_key="k2",
                               request_digest="d", correlation_id="c")
    with pytest.raises(LedgerError):  # the token names erase-1; it cannot commit another intent
        world.ledger.erase_record(token, other)
    expired = issue_erasure_token(world.verifier, authority, record_id="rec-1", content_revision=5,
                                  intent_id="erase-1", ttl_seconds=5.0, now=lambda: time.time() - 60)
    genuine = build_erase_intent(authority, record=RECORD, intent_id="erase-1", idempotency_key="k1",
                                 request_digest="d", correlation_id="c")
    with pytest.raises(AuthorityDenied, match="expired"):
        world.ledger.erase_record(expired, genuine)
    assert world.ledger.get_tombstone(T, "rec-1") is None
    assert world.ledger.erase_record(token, genuine).created  # the live token for the right intent works


# ---- the token matches the intent it commits
def test_a_token_revision_must_equal_the_intent_precondition(world) -> None:
    authority = world.authority_for()
    token = issue_erasure_token(world.verifier, authority, record_id="rec-1", content_revision=1, intent_id="erase-1")
    intent = build_erase_intent(
        authority, record=SourceVersion(record_id="rec-1", content_revision=99, policy_epoch=2),
        intent_id="erase-1", idempotency_key="k1", request_digest="d", correlation_id="c",
    )
    with pytest.raises(LedgerError, match="do not describe the same erasure"):
        world.ledger.erase_record(token, intent)
    assert world.ledger.get_tombstone(T, "rec-1") is None


def test_the_erased_record_is_fenced_at_the_revision_the_intent_named(world) -> None:
    world.erase()
    assert world.ledger.get_tombstone(T, "rec-1").content_revision == RECORD.content_revision
    assert world.ledger.get_intent(T, "erase-1").status in (IntentStatus.ACCEPTED, IntentStatus.PROCESSING)
