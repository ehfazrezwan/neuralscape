"""Scoped credentials, refresh rotation/replay, offboarding fences and service
accounts over the durable authority store (SQLite here; PostgreSQL in the
integration module)."""

from __future__ import annotations

import threading

import pytest

import credentials
from authority_store_sql import (
    CredentialInvalid,
    CredentialLimit,
    RefreshReplay,
    SqliteAuthorityStore,
    UnknownSubject,
)
from contracts_policy import PolicyStatement
from credentials import ACCESS_PREFIX, REFRESH_PREFIX, CredentialService, ScopeError, parse_token
from identity_binding import NamespaceKey
from policy_enforcement import pool_handle
from publication_fence import Admission, CurrentSource, PublicationFence
from tests.test_authority_store_sql import Clock

T = "tenant-a"


def grant(identifier, subject, action, handle=None):
    return PolicyStatement(
        id=identifier, effect="allow", subject_id=subject, credential_id=None,
        action=action, resource=handle or pool_handle(T, "shared", None),
    )


def seed(backend, clock=None):
    """A human with several grants, plus the service over the store."""

    store = backend.open(clock or Clock())
    store.add_subject("sub_h", T, "human", "cred-h")
    store.add_subject("sub_h2", T, "human", "cred-h2")
    store.publish_policy(
        T,
        [grant(f"g{i}", "sub_h", action) for i, action in enumerate(("read", "search", "list", "create", "manage_credentials"))]
        + [grant("x1", "sub_h2", "read")],
    )
    return store, CredentialService(store, clock=store._clock)


# ── conformance cases (shared with the PostgreSQL module) ────────────────


def case_secrets_are_hashed_at_rest_and_tokens_verify_exactly(backend):
    store, svc = seed(backend)
    issued = svc.issue("sub_h", ["read", "search"])
    assert issued.access_token.startswith(ACCESS_PREFIX) and issued.refresh_token.startswith(REFRESH_PREFIX)
    secrets_ = [issued.access_token.split(".", 1)[1], issued.refresh_token.split(".", 1)[1]]
    stored = store.stored_secret_hashes()
    assert len(stored) == 2 and not set(stored) & set(secrets_)
    assert all(len(h) == 64 for h in stored)
    verified = svc.verify_access(issued.access_token)
    assert (verified.subject_id, verified.tenant_id, verified.scope, verified.credential_id) == (
        "sub_h", T, ("read", "search"), issued.credential_id)
    cred_id, secret = issued.access_token[len(ACCESS_PREFIX):].split(".", 1)
    for bad in (
        f"{ACCESS_PREFIX}{cred_id}.{secret}x",
        f"{ACCESS_PREFIX}{cred_id}.",
        f"{ACCESS_PREFIX}cred_unknown.{secret}",
        issued.refresh_token,            # refresh token is not an access token
        "nsc_garbage", "", "x" * 600,
    ):
        assert svc.verify_access(bad) is None
    assert parse_token(issued.access_token, REFRESH_PREFIX) is None


def case_scope_is_a_subset_of_grants_and_can_only_narrow(backend):
    store, svc = seed(backend)
    for bad in ([], ["nonsense"], ["delete"], ["read", "delete"], ["manage_membership"]):
        with pytest.raises(ScopeError):
            svc.issue("sub_h", bad)
    assert svc.issue("sub_h", ["read"]).scope == ("read",)
    with pytest.raises(ScopeError):          # a scoped credential cannot mint beyond its own scope
        svc.issue("sub_h", ["read", "search"], ceiling=("read",))
    assert svc.issue("sub_h", ["read"], ceiling=("read", "search")).scope == ("read",)
    with pytest.raises(UnknownSubject):
        svc.issue("sub_missing", ["read"])
    with pytest.raises(ValueError):
        svc.issue("sub_h", ["read"], access_ttl=0)
    with pytest.raises(ValueError):
        svc.issue("sub_h", ["read"], access_ttl=10**9)


def case_expiry_and_offboarding_stop_a_credential(backend):
    clock = Clock()
    store, svc = seed(backend, clock)
    short = svc.issue("sub_h", ["read"], access_ttl=60)
    assert svc.verify_access(short.access_token)
    clock.now += 61
    assert svc.verify_access(short.access_token) is None
    live = svc.issue("sub_h2", ["read"])
    store.offboard_subject("sub_h2")
    assert svc.verify_access(live.access_token) is None
    with pytest.raises(CredentialInvalid):
        svc.refresh(live.refresh_token)


def case_refresh_is_single_use_and_rotates_the_family(backend):
    store, svc = seed(backend)
    first = svc.issue("sub_h", ["read", "search"])
    second = svc.refresh(first.refresh_token)
    assert second.family_id == first.family_id and second.scope == ("read", "search")
    assert second.access_token != first.access_token and second.refresh_token != first.refresh_token
    assert svc.verify_access(first.access_token) is None      # predecessor access retired
    assert svc.verify_access(second.access_token).credential_id == second.credential_id
    third = svc.refresh(second.refresh_token)
    assert svc.verify_access(third.access_token) and svc.verify_access(second.access_token) is None
    with pytest.raises(CredentialInvalid):
        svc.refresh("nsr_cred_nope.secret")
    with pytest.raises(CredentialInvalid):
        svc.refresh(first.access_token)                        # wrong kind


def case_reusing_a_refresh_token_revokes_the_whole_family(backend):
    store, svc = seed(backend)
    first = svc.issue("sub_h", ["read"])
    other_family = svc.issue("sub_h", ["search"])
    second = svc.refresh(first.refresh_token)
    store._clock.now += svc._refresh_grace + 1                # past the benign-duplicate window
    with pytest.raises(RefreshReplay):
        svc.refresh(first.refresh_token)                      # replay of a consumed token
    assert svc.verify_access(second.access_token) is None     # the successor died with the family
    with pytest.raises(CredentialInvalid):
        svc.refresh(second.refresh_token)
    assert svc.verify_access(other_family.access_token)       # unrelated family untouched
    assert svc.refresh(other_family.refresh_token)


def case_concurrent_redemption_yields_at_most_one_successor(backend):
    store, svc = seed(backend)
    issued = svc.issue("sub_h", ["read"])
    peers = [CredentialService(backend.open(store._clock), clock=store._clock, refresh_grace=0) for _ in range(6)]
    results: list[object] = []
    barrier = threading.Barrier(len(peers))

    def redeem(peer):
        barrier.wait()
        try:
            results.append(peer.refresh(issued.refresh_token))
        except Exception as error:  # noqa: BLE001
            results.append(error)

    threads = [threading.Thread(target=redeem, args=(p,)) for p in peers]
    [t.start() for t in threads]
    [t.join() for t in threads]
    winners = [r for r in results if isinstance(r, credentials.IssuedCredentials)]
    assert len(winners) == 1
    # The losers were replays, so the family is revoked (strict rotation).
    assert svc.verify_access(winners[0].access_token) is None


def case_revocation_and_offboarding_fence_queued_work(backend):
    store, svc = seed(backend)
    store.publish_policy(
        T,
        [grant("g", "sub_h", "background_derive", pool_handle(T, "private", "sub_h")), grant("r", "sub_h", "read")],
    )
    ns = NamespaceKey(T, "private", "sub_h", None, None)

    class Source:
        def current(self, tenant, record):
            return CurrentSource(1, ns, "private")

    fence = PublicationFence(store, Source())
    epoch = store.current_policy(T).policy_epoch
    version = store.subject_record("sub_h").membership_version

    def admitted(credential_id):
        return Admission(T, "sub_h", credential_id, version, epoch, "memory-1", 1, ns)

    # Derivation is done only within a credential's scope, so these carry it.
    scope = ["read", "background_derive"]
    keep, revoke = svc.issue("sub_h", scope), svc.issue("sub_h", scope)
    assert fence.evaluate(admitted(revoke.credential_id)).allowed
    assert svc.revoke(revoke.credential_id, by_subject_id="sub_h") is True
    decision = fence.evaluate(admitted(revoke.credential_id))
    assert (decision.allowed, decision.reason) == (False, "credential_retired")
    # Work queued under the other credential is unaffected until the subject goes.
    assert fence.evaluate(admitted(keep.credential_id)).allowed
    called = []
    outcome, _ = fence.publish(admitted(revoke.credential_id), lambda c: called.append(1))
    assert not outcome.allowed and not called
    # A credential whose scope lacks the derivation action never has work derived for it.
    reader = svc.issue("sub_h", ["read"])
    assert fence.evaluate(admitted(reader.credential_id)).reason == "outside_credential_scope"
    store.offboard_subject("sub_h")
    assert not fence.evaluate(admitted(keep.credential_id)).allowed


def case_only_the_owner_or_controller_may_revoke(backend):
    store, svc = seed(backend)
    issued = svc.issue("sub_h", ["read"])
    assert svc.revoke(issued.credential_id, by_subject_id="sub_h2") is False   # not theirs
    assert svc.revoke("cred_unknown", by_subject_id="sub_h") is False          # same answer
    assert svc.verify_access(issued.access_token)
    assert svc.revoke(issued.credential_id, by_subject_id="sub_h") is True
    assert svc.verify_access(issued.access_token) is None


def case_service_accounts_have_a_human_controller_and_least_authority(backend):
    store, svc = seed(backend)
    with pytest.raises(UnknownSubject):
        svc.create_service_account("sub_missing", "ci")
    account = svc.create_service_account("sub_h", "ci runner")
    info = store.service_account(account)
    assert (info["controller_subject_id"], info["label"], info["status"]) == ("sub_h", "ci runner", "active")
    assert store.subject_record(account) is None                  # no credential yet: unusable
    # A service account cannot control another one, nor can a worker-like subject.
    store.publish_policy(T, [grant("a1", account, "read"), grant("g", "sub_h", "read")])
    first = svc.rotate_service_account(account, "sub_h", ["read"])
    assert store.subject_record(account).subject_kind == "service_account"
    assert svc.verify_access(first.access_token).subject_id == account
    with pytest.raises(ScopeError):                               # holds no 'create'
        svc.rotate_service_account(account, "sub_h", ["create"])
    with pytest.raises(UnknownSubject):                           # not the controller
        svc.rotate_service_account(account, "sub_h2", ["read"])
    with pytest.raises(UnknownSubject):
        svc.disable_service_account(account, "sub_h2")
    with pytest.raises(ValueError):
        store.create_service_account("sub_sa2", T, account, "nested")  # a service account as controller


def case_rotation_without_downtime_allows_two_active_credentials(backend):
    store, svc = seed(backend)
    account = svc.create_service_account("sub_h", "ci")
    store.publish_policy(T, [grant("a1", account, "read"), grant("g", "sub_h", "read")])
    one = svc.rotate_service_account(account, "sub_h", ["read"])
    two = svc.rotate_service_account(account, "sub_h", ["read"])
    assert svc.verify_access(one.access_token) and svc.verify_access(two.access_token)   # overlap
    with pytest.raises(CredentialLimit):
        svc.rotate_service_account(account, "sub_h", ["read"])
    assert store.active_family_count(account) == 2
    assert svc.revoke(one.credential_id, by_subject_id="sub_h") is True              # controller retires the old key
    assert svc.verify_access(one.access_token) is None and svc.verify_access(two.access_token)
    assert store.subject_record(account) is not None                                 # still usable
    three = svc.rotate_service_account(account, "sub_h", ["read"])
    assert svc.verify_access(three.access_token)


def case_disabling_a_service_account_retires_everything(backend):
    store, svc = seed(backend)
    account = svc.create_service_account("sub_h", "ci")
    store.publish_policy(T, [grant("a1", account, "read"), grant("g", "sub_h", "read")])
    issued = svc.rotate_service_account(account, "sub_h", ["read"])
    assert svc.disable_service_account(account, "sub_h") is True
    assert svc.disable_service_account(account, "sub_h") is False      # idempotent
    assert svc.verify_access(issued.access_token) is None
    with pytest.raises(CredentialInvalid):
        svc.refresh(issued.refresh_token)
    assert store.subject_record(account) is None
    with pytest.raises(ValueError):
        svc.rotate_service_account(account, "sub_h", ["read"])
    assert store.service_account(account)["status"] == "disabled"


# ── additional cases ──────────────────────────────────────────────────────


def case_minted_credentials_never_outlive_their_issuer(backend):
    clock = Clock()
    store, svc = seed(backend, clock)
    parent = svc.issue("sub_h", ["read", "manage_credentials"], access_ttl=60, with_refresh=False)
    issuer = svc.verify_access(parent.access_token)
    assert issuer.expires_at == parent.access_expires_at
    child = svc.issue("sub_h", ["read"], access_ttl=900, with_refresh=False, issuer=issuer)
    assert child.access_expires_at == parent.access_expires_at          # capped at the issuer's expiry
    with pytest.raises(ScopeError):                                      # no refresh token from a scoped issuer
        svc.issue("sub_h", ["read"], issuer=issuer, with_refresh=True)
    with pytest.raises(ScopeError):                                      # nothing beyond the issuer's scope
        svc.issue("sub_h", ["read", "search"], with_refresh=False, issuer=issuer)
    permissive = CredentialService(store, clock=clock, allow_scoped_refresh_minting=True)
    grandchild = permissive.issue("sub_h", ["read"], refresh_ttl=10**6, issuer=issuer)
    assert grandchild.refresh_expires_at == parent.access_expires_at     # refresh capped too
    clock.now += 61
    with pytest.raises(ScopeError):                                      # an expired issuer cannot mint
        svc.issue("sub_h", ["read"], with_refresh=False, issuer=issuer)
    # An ordinary (non-scoped) caller is unchanged.
    plain = svc.issue("sub_h", ["read"])
    assert plain.refresh_token and plain.access_expires_at == clock.now + 900


def case_revoking_the_issuing_family_revokes_what_it_minted(backend):
    store, svc = seed(backend)
    parent = svc.issue("sub_h", ["read", "manage_credentials"], with_refresh=False)
    issuer = svc.verify_access(parent.access_token)
    child = svc.issue("sub_h", ["read", "manage_credentials"], with_refresh=False, issuer=issuer)
    grandchild = svc.issue("sub_h", ["read"], with_refresh=False, issuer=svc.verify_access(child.access_token))
    sibling = svc.issue("sub_h", ["read"], with_refresh=False)
    assert all(svc.verify_access(t.access_token) for t in (parent, child, grandchild, sibling))
    assert svc.revoke(child.credential_id, by_subject_id="sub_h")        # revoking a middle family
    assert svc.verify_access(parent.access_token) and svc.verify_access(sibling.access_token)
    assert svc.verify_access(child.access_token) is None and svc.verify_access(grandchild.access_token) is None
    assert svc.revoke(parent.credential_id, by_subject_id="sub_h")
    assert svc.verify_access(parent.access_token) is None and svc.verify_access(sibling.access_token)


def case_a_benign_duplicate_refresh_returns_the_same_successor(backend):
    clock = Clock()
    store, svc = seed(backend, clock)
    first = svc.issue("sub_h", ["read"])
    one = svc.refresh(first.refresh_token)
    again = svc.refresh(first.refresh_token)                              # lost response / two tabs
    assert (again.access_token, again.refresh_token, again.access_expires_at, again.refresh_expires_at) == (
        one.access_token, one.refresh_token, one.access_expires_at, one.refresh_expires_at)
    assert svc.verify_access(one.access_token) and svc.verify_access(first.access_token) is None
    other = svc.issue("sub_h", ["read"])
    assert svc.refresh(other.refresh_token).refresh_token != one.refresh_token   # derived per presented token
    # The duplicate is only idempotent while the successor is unused: once the
    # successor is redeemed, presenting the original again is a replay.
    nxt = svc.refresh(one.refresh_token)
    with pytest.raises(RefreshReplay):
        svc.refresh(first.refresh_token)
    assert svc.verify_access(nxt.access_token) is None                   # the family is revoked


def case_a_late_duplicate_refresh_is_a_replay_and_revokes_the_family(backend):
    clock = Clock()
    store, svc = seed(backend, clock)
    first = svc.issue("sub_h", ["read"])
    one = svc.refresh(first.refresh_token)
    clock.now += svc._refresh_grace + 0.5
    with pytest.raises(RefreshReplay):
        svc.refresh(first.refresh_token)
    assert svc.verify_access(one.access_token) is None
    strict = CredentialService(store, clock=clock, refresh_grace=0)
    other = strict.issue("sub_h", ["read"])
    strict.refresh(other.refresh_token)
    with pytest.raises(RefreshReplay):                                    # grace 0 disables the tolerance
        strict.refresh(other.refresh_token)


def case_concurrent_duplicates_within_the_grace_all_get_the_same_successor(backend):
    store, svc = seed(backend)
    issued = svc.issue("sub_h", ["read"])
    peers = [CredentialService(backend.open(store._clock), clock=store._clock) for _ in range(6)]
    results: list[object] = []
    barrier = threading.Barrier(len(peers))

    def redeem(peer):
        barrier.wait()
        try:
            results.append(peer.refresh(issued.refresh_token))
        except Exception as error:  # noqa: BLE001
            results.append(error)

    threads = [threading.Thread(target=redeem, args=(p,)) for p in peers]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert all(isinstance(r, credentials.IssuedCredentials) for r in results), results
    assert len({(r.access_token, r.refresh_token) for r in results}) == 1
    assert svc.verify_access(results[0].access_token)


def case_offboarding_a_controller_disables_their_service_accounts(backend):
    store, svc = seed(backend)
    account = svc.create_service_account("sub_h", "ci")
    store.publish_policy(T, [grant("a1", account, "read"), grant("g", "sub_h", "read")])
    issued = svc.rotate_service_account(account, "sub_h", ["read"])
    other = svc.create_service_account("sub_h2", "other")
    before = store.service_account(account)["status"]
    store.offboard_subject("sub_h")
    assert before == "active" and store.service_account(account)["status"] == "disabled"
    assert svc.verify_access(issued.access_token) is None
    with pytest.raises(CredentialInvalid):
        svc.refresh(issued.refresh_token)
    with pytest.raises(ValueError):
        svc.rotate_service_account(account, "sub_h", ["read"])
    assert store.service_account(other)["status"] == "active"             # someone else's account untouched


def case_a_malformed_scope_is_a_scope_error(backend):
    store, svc = seed(backend)
    for bad in ([["read"]], [{"a": 1}], [None]):
        with pytest.raises(ScopeError):
            svc.issue("sub_h", bad)


def case_refreshing_a_capped_family_never_extends_its_lifetime(backend):
    clock = Clock()
    store, svc = seed(backend, clock)
    permissive = CredentialService(store, clock=clock, allow_scoped_refresh_minting=True)
    parent = svc.issue("sub_h", ["read", "manage_credentials"], access_ttl=60, with_refresh=False)
    issuer = svc.verify_access(parent.access_token)
    child = permissive.issue("sub_h", ["read"], issuer=issuer, access_ttl=900, refresh_ttl=10**6)
    assert child.access_expires_at == child.refresh_expires_at == parent.access_expires_at
    clock.now += 30
    successor = permissive.refresh(child.refresh_token)
    # The cap travels with the family: the successor cannot outlive the issuer's expiry.
    assert successor.access_expires_at == parent.access_expires_at
    assert successor.refresh_expires_at == parent.access_expires_at
    assert permissive.verify_access(successor.access_token).expires_at == parent.access_expires_at
    again = permissive.refresh(successor.refresh_token)
    assert again.access_expires_at == again.refresh_expires_at == parent.access_expires_at
    clock.now = parent.access_expires_at + 1                                   # past the cap
    assert permissive.verify_access(again.access_token) is None
    with pytest.raises(CredentialInvalid):
        permissive.refresh(again.refresh_token)                                # and no further refresh
    # An uncapped family still gets the full lifetime on refresh.
    plain = svc.issue("sub_h", ["read"])
    assert svc.refresh(plain.refresh_token).access_expires_at == clock.now + 900


class _LookAlike(str):
    """An identifier that claims to equal every other one."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = str.__hash__


def case_identifiers_that_compare_loosely_are_refused(backend):
    store, svc = seed(backend)
    issued = svc.issue("sub_h", ["read"])
    with pytest.raises(ValueError):                  # sub_h2 holds read only; a loose id would collect sub_h's grants
        svc.granted_actions(_LookAlike("sub_h2"))
    with pytest.raises(ValueError):
        svc.issue(_LookAlike("sub_h2"), ["manage_credentials"])
    with pytest.raises(ValueError):                  # would pass as the credential's owner
        svc.revoke(issued.credential_id, by_subject_id=_LookAlike("sub_h2"))
    with pytest.raises(ValueError):                  # the same refusal for a credential that does not exist
        svc.revoke("cred_unknown", by_subject_id=_LookAlike("sub_h2"))
    assert svc.verify_access(issued.access_token)
    assert svc.granted_actions("sub_h2") == frozenset({"read"})


CASES = [v for k, v in sorted(globals().items()) if k.startswith("case_") and callable(v)]


class SqliteBackend:
    def __init__(self, path):
        self.path, self.opened = path, []

    def open(self, clock=None):
        store = SqliteAuthorityStore(self.path, **({"clock": clock} if clock else {}))
        self.opened.append(store)
        return store

    def close(self):
        for store in self.opened:
            store.close()


@pytest.fixture
def sqlite_backend(tmp_path):
    backend = SqliteBackend(str(tmp_path / "credentials.db"))
    yield backend
    backend.close()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.__name__)
def test_sqlite_credential_conformance(case, sqlite_backend) -> None:
    case(sqlite_backend)


def test_credentials_are_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("CREDENTIALS_ENABLED", raising=False)
    assert credentials.enabled() is False and credentials.accepts_scoped_credentials() is False
    monkeypatch.setenv("CREDENTIALS_ENABLED", "true")
    from config import settings

    assert credentials.enabled() is True
    monkeypatch.setattr(settings, "ns_policy_mode", "observe")
    assert credentials.accepts_scoped_credentials() is False   # a scope nothing enforces is refused
    monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
    assert credentials.accepts_scoped_credentials() is True
