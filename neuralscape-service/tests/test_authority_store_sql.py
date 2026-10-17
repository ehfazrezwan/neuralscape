"""Authority and binding store conformance (SQLite here; PostgreSQL in the
integration module) plus an end-to-end enforcement test on the real app."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

import authority_store_sql as store_module
import policy_enforcement
from authority_store import AuthorityUnavailable
from authority_store_sql import (
    PolicyConflict,
    SqliteAuthorityStore,
    UnknownSubject,
    WitnessStale,
    build_authority_store,
)
from config import settings
from contracts_policy import DelegationConstraints, PolicyStatement
from identity_binding import bind_identity
from ledger.port import SchemaError
from policy_enforcement import PolicyEnforcer, pool_handle


class Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def stmt(identifier: str, subject: str, action: str, handle, effect="allow", credential=None):
    return PolicyStatement(
        id=identifier,
        effect=effect,
        subject_id=subject,
        credential_id=credential,
        action=action,
        resource=handle,
    )


def handle(tenant="tenant-a", kind="shared", owner=None, project=None):
    return pool_handle(tenant, kind, owner, project)


# ── conformance cases: (backend) -> None; shared with the PostgreSQL module ──


def case_binding_is_atomic_verbatim_and_race_safe(backend):
    store = backend.open()
    assert store.get("issuer", "Sub") is None
    assert store.put_if_absent("issuer", "Sub", "sub_1") == "sub_1"
    assert store.put_if_absent("issuer", "Sub", "sub_2") == "sub_1"  # loser gets the winner
    assert store.get("issuer", "Sub") == "sub_1"
    assert store.get("issuer", "sub") is None and store.get("Issuer", "Sub") is None

    results = []
    barrier = threading.Barrier(8)

    def racer(index):
        peer = backend.open()
        barrier.wait()
        results.append(bind_identity(peer, "idp", "racer", generate_id=lambda: f"sub_{index}").subject_id)

    threads = [threading.Thread(target=racer, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(set(results)) == 1 and store.get("idp", "racer") == results[0]


def case_subjects_versions_and_tombstones(backend):
    clock = Clock()
    store = backend.open(clock)
    assert store.add_subject("sub_a", "tenant-a", "human", "cred-a1") == 1
    record = store.subject_record("sub_a")
    assert (record.tenant_id, record.subject_kind, record.credential_id, record.membership_version) == (
        "tenant-a", "human", "cred-a1", 1,
    )
    with pytest.raises(ValueError):
        store.add_subject("sub_a", "tenant-b", "human", "cred-x")  # no id reuse
    assert store.update_subject("sub_a", subject_kind="service_account") == 2
    assert store.subject_record("sub_a").subject_kind == "service_account"
    # Rotation: new credential, old one retired, version bumped again.
    assert store.update_subject("sub_a", credential_id="cred-a2") == 3
    assert store.credential_state("cred-a1") == "retired"
    assert store.credential_state("cred-a2") == "active"
    assert store.subject_record("sub_a").credential_id == "cred-a2"
    # Offboarding tombstones: unreadable, version bumped, id and binding persist.
    store.put_if_absent("idp", "person", "sub_a")
    assert store.offboard_subject("sub_a") == 4
    assert store.subject_record("sub_a") is None
    assert store.get("idp", "person") == "sub_a"
    assert store.offboard_subject("sub_a") == 4  # idempotent
    with pytest.raises(ValueError):
        store.add_subject("sub_a", "tenant-a", "human", "cred-new")
    with pytest.raises(ValueError):
        store.update_subject("sub_a", subject_kind="human")
    with pytest.raises(UnknownSubject):
        store.update_subject("sub_missing", subject_kind="human")
    assert store.subject_record("sub_missing") is None


def case_delegation_round_trips(backend):
    store = backend.open()
    ceiling = DelegationConstraints(
        delegated_by_subject_id="sub_h",
        delegated_by_credential_id="cred-h",
        allowed_actions=("read",),
        allowed_resources=(handle(),),
    )
    store.add_subject("sub_d", "tenant-a", "delegated_agent", "cred-d", delegation=ceiling)
    assert store.subject_record("sub_d").delegation == ceiling


def case_credentials_retire_and_expire(backend):
    clock = Clock()
    store = backend.open(clock)
    store.add_subject("sub_a", "tenant-a", "human", "cred-a", expires_at=clock.now + 10)
    assert store.credential_state("cred-a") == "active"
    clock.now += 11
    assert store.credential_state("cred-a") == "expired"
    assert store.subject_record("sub_a") is None
    store.add_subject("sub_b", "tenant-a", "human", "cred-b")
    assert store.retire_credential("cred-b") is True
    assert store.retire_credential("cred-b") is False
    assert store.credential_state("cred-b") == "retired"
    assert store.subject_record("sub_b") is None
    assert store.credential_state("cred-none") == "unknown"
    with pytest.raises(UnknownSubject):
        store.retire_credential("cred-none")


def case_policy_publish_is_atomic_with_epoch_increment(backend):
    store = backend.open()
    store.add_subject("sub_a", "tenant-a", "human", "cred-a")
    store.add_subject("sub_off", "tenant-a", "human", "cred-off")
    store.offboard_subject("sub_off")
    store.add_subject("sub_b", "tenant-b", "human", "cred-b")
    assert store.current_policy("tenant-a") is None
    first = store.publish_policy("tenant-a", [stmt("s1", "sub_a", "read", handle())], expected_epoch=0)
    assert first.policy_epoch == 1
    assert [m.subject_id for m in first.membership_versions] == ["sub_a"]  # active, own tenant only
    second = store.publish_policy(
        "tenant-a",
        [stmt("s2", "sub_a", "list", handle()), stmt("s3", "sub_a", "read", handle(), effect="deny")],
    )
    assert second.policy_epoch == 2
    assert [s.id for s in second.statements] == ["s2", "s3"]  # replaced, order kept
    assert store.current_policy("tenant-a") == second
    assert store.current_policy("tenant-b") is None
    with pytest.raises(PolicyConflict):
        store.publish_policy("tenant-a", [], expected_epoch=1)
    assert store.current_policy("tenant-a").policy_epoch == 2
    # An invalid set (foreign tenant resource, duplicate id) changes nothing.
    with pytest.raises(Exception):
        store.publish_policy("tenant-a", [stmt("x", "sub_a", "read", handle(tenant="tenant-b"))])
    with pytest.raises(Exception):
        store.publish_policy("tenant-a", [stmt("d", "sub_a", "read", handle()), stmt("d", "sub_a", "list", handle())])
    assert store.current_policy("tenant-a") == second
    # Membership version bumps are reflected without a new epoch.
    store.update_subject("sub_a", subject_kind="service_account")
    current = store.current_policy("tenant-a")
    assert current.policy_epoch == 2 and current.membership_versions[0].version == 2


def case_concurrent_publishes_get_distinct_consecutive_epochs(backend):
    store = backend.open()
    store.add_subject("sub_a", "tenant-a", "human", "cred-a")
    epochs = []
    barrier = threading.Barrier(6)

    def publisher(index):
        peer = backend.open()
        barrier.wait()
        epochs.append(peer.publish_policy("tenant-a", [stmt(f"s{index}", "sub_a", "read", handle())]).policy_epoch)

    threads = [threading.Thread(target=publisher, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(epochs) == [1, 2, 3, 4, 5, 6]
    assert store.current_policy("tenant-a").policy_epoch == 6


def case_hold_current_verifies_then_commits_body_atomically(backend):
    store = backend.open()
    store.add_subject("sub_a", "tenant-a", "human", "cred-a")
    policy = store.publish_policy("tenant-a", [stmt("s1", "sub_a", "read", handle())])
    base = dict(
        tenant_id="tenant-a", subject_id="sub_a", credential_id="cred-a",
        membership_version=1, policy_epoch=policy.policy_epoch,
    )
    store.hold_current(**base, body=lambda c: store._x(c, "CREATE TABLE scratch(v TEXT)"))
    assert store.hold_current(**base, body=lambda c: store._x(c, "INSERT INTO scratch VALUES('ok')") and "done") == "done"
    # A raising body rolls back its writes and propagates.
    with pytest.raises(RuntimeError):
        def boom(c):
            store._x(c, "INSERT INTO scratch VALUES('lost')")
            raise RuntimeError("boom")
        store.hold_current(**base, body=boom)
    with store._read_txn() as c:
        assert [r[0] for r in store._x(c, "SELECT v FROM scratch").fetchall()] == ["ok"]

    def stale(**changes):
        called = []
        with pytest.raises(WitnessStale) as caught:
            store.hold_current(**{**base, **changes}, body=lambda c: called.append(1))
        assert not called
        return caught.value.reason

    assert stale(policy_epoch=policy.policy_epoch + 1) == "policy_epoch_changed"
    assert stale(membership_version=2) == "membership_changed"
    assert stale(tenant_id="tenant-b") == "policy_epoch_changed"
    assert stale(subject_id="sub_other") == "subject_not_active"
    assert stale(credential_id="cred-other") == "credential_retired"
    store.add_subject("sub_b", "tenant-a", "human", "cred-b")  # active, but it is sub_b's credential
    assert stale(credential_id="cred-b") == "credential_subject_mismatch"
    store.publish_policy("tenant-a", [])
    assert stale() == "policy_epoch_changed"


def case_tenants_lists_every_tenant_with_a_policy_or_a_subject(backend):
    store = backend.open()
    assert store.tenants() == ()
    store.add_subject("sub_a", "tenant-b", "human", "cred-a")
    store.publish_policy("tenant-a", [])
    assert store.tenants() == ("tenant-a", "tenant-b")  # sorted, each once


def case_hold_current_denies_after_retirement_and_offboarding(backend):
    store = backend.open()
    store.add_subject("sub_a", "tenant-a", "human", "cred-a")
    store.add_subject("sub_b", "tenant-a", "human", "cred-b")
    epoch = store.publish_policy("tenant-a", []).policy_epoch
    store.retire_credential("cred-a")
    with pytest.raises(WitnessStale) as caught:
        store.hold_current(tenant_id="tenant-a", subject_id="sub_a", credential_id="cred-a",
                           membership_version=1, policy_epoch=epoch, body=lambda c: None)
    assert caught.value.reason == "credential_retired"
    store.offboard_subject("sub_b")
    with pytest.raises(WitnessStale) as caught:
        store.hold_current(tenant_id="tenant-a", subject_id="sub_b", credential_id="cred-b",
                           membership_version=1, policy_epoch=epoch, body=lambda c: None)
    assert caught.value.reason in {"subject_not_active", "credential_retired"}


def case_holding_blocks_a_concurrent_revocation_until_commit(backend):
    store, other = backend.open(), backend.open()
    store.add_subject("sub_a", "tenant-a", "human", "cred-a")
    epoch = store.publish_policy("tenant-a", []).policy_epoch
    inside, release, finished = threading.Event(), threading.Event(), threading.Event()

    def revoker():
        inside.wait()
        other.retire_credential("cred-a")
        finished.set()

    thread = threading.Thread(target=revoker)
    thread.start()

    def body(c):
        inside.set()
        time.sleep(0.4)  # give the revoker a chance to (wrongly) get through
        assert not finished.is_set(), "revocation overtook a held publication"
        return "published"

    result = store.hold_current(tenant_id="tenant-a", subject_id="sub_a", credential_id="cred-a",
                                membership_version=1, policy_epoch=epoch, body=body)
    thread.join(10)
    assert result == "published" and finished.is_set()
    assert store.credential_state("cred-a") == "retired"


def case_schema_is_versioned_idempotent_and_drift_checked(backend):
    store = backend.open()
    latest = max(m.version for m in store_module.MIGRATIONS)
    assert store.schema_version() == latest
    assert backend.open().schema_version() == latest  # reopening applies nothing twice
    with store._txn() as c:
        store._x(c, "UPDATE authority_schema_version SET checksum='tampered' WHERE version=1")
    with pytest.raises(SchemaError, match="modified"):
        backend.open()
    with store._txn() as c:
        store._x(c, "UPDATE authority_schema_version SET checksum=? WHERE version=1",
                 (store_module.MIGRATIONS[0].checksum,))
        store._x(c, "INSERT INTO authority_schema_version VALUES (9, 'future', 'x', 0)")
    with pytest.raises(SchemaError, match="newer"):
        backend.open()


def case_storage_failure_is_an_unavailable_authority(backend):
    store = backend.open()
    store.add_subject("sub_a", "tenant-a", "human", "cred-a")
    store.close()
    for call in (
        lambda: store.subject_record("sub_a"),
        lambda: store.current_policy("tenant-a"),
        lambda: store.get("i", "s"),
        lambda: store.put_if_absent("i", "s", "sub_x"),
        lambda: store.publish_policy("tenant-a", []),
    ):
        with pytest.raises(AuthorityUnavailable):
            call()


# ── additional cases (shared with the PostgreSQL module) ────────────────────


def case_fence_requires_the_admitted_credential_to_belong_to_the_admitted_subject(backend):
    from identity_binding import NamespaceKey
    from publication_fence import Admission, CurrentSource, PublicationFence

    store = backend.open()
    store.add_subject("sub_a", "tenant-a", "human", "cred-a")
    store.add_subject("sub_b", "tenant-a", "human", "cred-b")
    ns = NamespaceKey("tenant-a", "private", "sub_a", None, None)
    policy = store.publish_policy("tenant-a", [stmt("g", "sub_a", "background_derive", pool_handle("tenant-a", "private", "sub_a"))])

    class Source:
        def current(self, tenant, record):
            return CurrentSource(1, ns, "private")

    fence = PublicationFence(store, Source())
    honest = Admission("tenant-a", "sub_a", "cred-a", 1, policy.policy_epoch, "m", 1, ns)
    assert fence.evaluate(honest).allowed
    swapped = honest._replace(credential_id="cred-b")           # another subject's ACTIVE credential
    decision = fence.evaluate(swapped)
    assert (decision.allowed, decision.reason) == (False, "credential_subject_mismatch")
    ran = []
    outcome, _ = fence.publish(swapped, lambda c: ran.append(1))
    assert not outcome.allowed and not ran
    # the held transaction checks the binding too, not only the evaluate pre-check
    with pytest.raises(WitnessStale) as caught:
        store.hold_current(tenant_id="tenant-a", subject_id="sub_a", credential_id="cred-b",
                           membership_version=1, policy_epoch=policy.policy_epoch, body=lambda c: ran.append(1))
    assert caught.value.reason == "credential_subject_mismatch" and not ran


def case_duplicate_credential_ids_are_validation_errors_not_outages(backend):
    store = backend.open()
    store.add_subject("sub_a", "tenant-a", "human", "cred-a")
    store.add_subject("sub_b", "tenant-a", "human", "cred-b")
    store.retire_credential("cred-a")
    for call in (
        lambda: store.add_subject("sub_c", "tenant-a", "human", "cred-a"),           # a retired id
        lambda: store.add_subject("sub_d", "tenant-a", "human", "cred-b"),           # an active id
        lambda: store.update_subject("sub_b", credential_id="cred-a"),               # rotate onto a used id
    ):
        with pytest.raises(ValueError):
            call()
    assert store.subject_record("sub_b").credential_id == "cred-b"                  # nothing changed


def case_control_characters_are_rejected_identically_on_both_backends(backend):
    store = backend.open()
    for bad in ("a\x00b", "a\nb", "a\x7fb", "", "x" * 257):
        with pytest.raises(ValueError):
            store.add_subject("sub_" + "z", "tenant-a", "human", bad)
        with pytest.raises(ValueError):
            store.add_subject(bad, "tenant-a", "human", "cred-z")
        with pytest.raises(ValueError):
            store.put_if_absent("issuer", bad, "sub_x")
        assert store.get("issuer", bad) is None                 # no binding can exist: not an outage
    assert store.subject_summary("sub_z") is None


def case_update_subject_applies_a_new_expiry(backend):
    clock = Clock()
    store = backend.open(clock)
    store.add_subject("sub_a", "tenant-a", "human", "cred-a", expires_at=clock.now + 1000)
    store.update_subject("sub_a", expires_at=clock.now + 10)                        # shorten
    clock.now += 11
    assert store.credential_state("cred-a") == "expired" and store.subject_record("sub_a") is None
    store.add_subject("sub_b", "tenant-a", "human", "cred-b", expires_at=clock.now + 10)
    store.update_subject("sub_b", expires_at=clock.now + 1000)                      # extend
    clock.now += 11
    assert store.credential_state("cred-b") == "active"


def case_reads_are_not_serialized_behind_a_running_publication(backend):
    import time as _time

    store, other = backend.open(), backend.open()
    store.add_subject("sub_a", "tenant-a", "human", "cred-a")
    epoch = store.publish_policy("tenant-a", []).policy_epoch
    inside, release = threading.Event(), threading.Event()

    def holder():
        store.hold_current(tenant_id="tenant-a", subject_id="sub_a", credential_id="cred-a",
                           membership_version=1, policy_epoch=epoch,
                           body=lambda c: (inside.set(), release.wait(10)))

    thread = threading.Thread(target=holder)
    thread.start()
    assert inside.wait(10)
    started = _time.monotonic()
    assert store.subject_record("sub_a") is not None            # same store, same process
    assert store.current_policy("tenant-a").policy_epoch == epoch
    assert store.credential_state("cred-a") == "active"
    assert store.get("issuer", "nobody") is None
    elapsed = _time.monotonic() - started
    release.set()
    thread.join(10)
    assert elapsed < 1.5, f"reads waited {elapsed:.1f}s behind the held publication"


class _LookAlike(str):
    """An identifier that claims to equal every other one."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = str.__hash__


class _LooseInt(int):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = int.__hash__


def case_identifiers_that_compare_loosely_are_refused_before_they_decide_anything(backend):
    store = backend.open()
    store.add_subject("sub_a", "tenant-a", "human", "cred-a")
    store.add_subject("sub_b", "tenant-a", "human", "cred-b")
    policy = store.publish_policy(
        "tenant-a",
        [stmt("s1", "sub_a", "read", handle()), stmt("s2", "sub_b", "read", handle()),
         stmt("s3", "sub_a", "manage_membership", handle())],
    )
    with pytest.raises(ValueError):                  # would drop every member's statements
        store.remove_member("tenant-a", _LookAlike("sub_b"))
    with pytest.raises(ValueError):                  # would pass the tenant check for any tenant
        store.remove_member(_LookAlike("tenant-b"), "sub_b")
    for loose in (_LooseInt(99), True):              # would pass the compare-and-set and replace the policy
        with pytest.raises(ValueError):
            store.publish_policy("tenant-a", [stmt("s9", "sub_a", "read", handle())], expected_epoch=loose)
    assert len(store.current_policy("tenant-a").statements) == 3
    assert store.subject_record("sub_b") is not None
    base = dict(
        tenant_id="tenant-a", subject_id="sub_a", credential_id="cred-a",
        membership_version=1, policy_epoch=policy.policy_epoch,
    )
    for changes in (
        {"tenant_id": _LookAlike("tenant-b")}, {"subject_id": _LookAlike("sub_b")},
        {"credential_id": _LookAlike("cred-b")}, {"membership_version": _LooseInt(7)},
        {"policy_epoch": _LooseInt(9)}, {"policy_epoch": True},
    ):
        called = []
        with pytest.raises(ValueError):
            store.hold_current(**{**base, **changes}, body=lambda c: called.append(1))
        assert not called


CASES = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("case_") and callable(value)
]


class SqliteBackend:
    def __init__(self, path: str) -> None:
        self.path = path
        self.opened: list[SqliteAuthorityStore] = []

    def open(self, clock=None) -> SqliteAuthorityStore:
        store = SqliteAuthorityStore(self.path, **({"clock": clock} if clock else {}))
        self.opened.append(store)
        return store

    def close(self) -> None:
        for store in self.opened:
            store.close()


@pytest.fixture
def sqlite_backend(tmp_path):
    backend = SqliteBackend(str(tmp_path / "authority.db"))
    yield backend
    backend.close()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.__name__)
def test_sqlite_conformance(case, sqlite_backend) -> None:
    case(sqlite_backend)


def test_sqlite_state_survives_reopen(sqlite_backend) -> None:
    store = sqlite_backend.open()
    store.add_subject("sub_a", "tenant-a", "human", "cred-a")
    store.put_if_absent("idp", "p", "sub_a")
    store.publish_policy("tenant-a", [stmt("s1", "sub_a", "read", handle())])
    store.close()
    again = sqlite_backend.open()
    assert again.get("idp", "p") == "sub_a"
    assert again.current_policy("tenant-a").policy_epoch == 1
    assert again.subject_record("sub_a").membership_version == 1


def test_sqlite_requires_wal_and_full_sync(sqlite_backend) -> None:
    store = sqlite_backend.open()
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert store._conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL


def test_migrations_are_expand_only_and_planned() -> None:
    assert [m.version for m in store_module.MIGRATIONS] == [1, 2, 3, 4, 5]
    assert store_module.plan_authority_migrations({}) == store_module.MIGRATIONS
    first = {1: store_module.MIGRATIONS[0].checksum}
    assert store_module.plan_authority_migrations(first) == store_module.MIGRATIONS[1:]
    applied = {m.version: m.checksum for m in store_module.MIGRATIONS}
    assert store_module.plan_authority_migrations(applied) == ()


def test_factory_is_disabled_by_default_and_validates_settings(monkeypatch, tmp_path) -> None:
    assert settings.authority_enabled is False
    assert build_authority_store(settings) is None
    monkeypatch.setattr(settings, "authority_enabled", True)
    monkeypatch.setattr(settings, "authority_sqlite_path", "")
    with pytest.raises(ValueError):
        build_authority_store(settings)
    monkeypatch.setattr(settings, "authority_sqlite_path", str(tmp_path / "a.db"))
    built = build_authority_store(settings)
    assert isinstance(built, SqliteAuthorityStore)
    built.close()
    monkeypatch.setattr(settings, "authority_backend", "postgres")
    with pytest.raises(ValueError):
        build_authority_store(settings)


# ── end to end: enforce mode + SQLite authority + the real app ───────────────


def test_enforce_mode_end_to_end_on_the_real_app(sqlite_backend, monkeypatch) -> None:
    import main
    from tokens import issue_user_token

    store = sqlite_backend.open()
    bindings = {}
    for user, tenant in (("alice", "tenant-a"), ("bob", "tenant-b")):
        subject = bind_identity(store, "local", user).subject_id
        store.add_subject(subject, tenant, "human", f"cred-{user}")
        bindings[user] = subject
    store.publish_policy(
        "tenant-a",
        [
            stmt("a1", bindings["alice"], "read", policy_enforcement._handle("settings", "proj-a", "tenant-a")),
            stmt("a2", bindings["alice"], "search", pool_handle("tenant-a", "private", bindings["alice"])),
        ],
    )
    store.publish_policy(
        "tenant-b",
        [stmt("b1", bindings["bob"], "read", policy_enforcement._handle("settings", "proj-b", "tenant-b"))],
    )
    policy_enforcement.configure(PolicyEnforcer(store, store))
    monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", "e2e-secret")
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    token = {u: issue_user_token(u, "e2e-secret", 3600) for u in ("alice", "bob")}
    client = TestClient(main.app, raise_server_exceptions=False)

    class Collect(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    # The app reconfigures the root logger, which drops caplog's handler, so the
    # decision log is collected by a handler on the policy logger itself.
    collector = Collect()
    policy_logger = logging.getLogger("neuralscape.policy")
    policy_logger.addHandler(collector)
    policy_logger.setLevel(logging.INFO)

    def call(user, method, path, **kwargs):
        headers = {"Authorization": f"Bearer {token[user]}"} if user else {}
        collector.records.clear()
        response = client.request(method, path, headers=headers, **kwargs)
        decisions = [r for r in collector.records if r.getMessage() == "policy_decision"]
        return response, decisions

    try:
        # Positive: policy allows; whatever the (service-less) handler then does
        # is not a policy denial.
        ok, decisions = call("alice", "GET", "/v1/projects/proj-a/knowledge-config")
        assert decisions and decisions[-1].policy_allowed is True
        assert "error" not in (ok.json().get("detail") or {}) if isinstance(ok.json(), dict) else True
        ok2, decisions = call("alice", "POST", "/v1/search", json={"query": "x"})
        assert decisions[-1].policy_allowed is True
        # Cross tenant: bob has no grant on alice's project or pool; alice none on bob's.
        for user, path in (("bob", "/v1/projects/proj-a/knowledge-config"), ("alice", "/v1/projects/proj-b/knowledge-config")):
            denied, decisions = call(user, "GET", path)
            assert denied.status_code == 404
            assert denied.json()["detail"]["error"]["code"] == "not_found"
            assert decisions[-1].policy_allowed is False
        # Bob's own positive control, and the unbound/unknown user is denied.
        _, decisions = call("bob", "GET", "/v1/projects/proj-b/knowledge-config")
        assert decisions[-1].policy_allowed is True
        denied, _ = call("alice", "POST", "/v1/connectors", json={})
        assert denied.status_code == 404  # content grants never imply admin
        # Revoking alice's credential in the durable store takes effect at once.
        store.retire_credential("cred-alice")
        denied, decisions = call("alice", "GET", "/v1/projects/proj-a/knowledge-config")
        assert denied.status_code == 404 and decisions[-1].policy_allowed is False
        # An unreadable store fails closed with 503.
        store._rconn.close(); store._conn.close()
        down, _ = call("bob", "GET", "/v1/projects/proj-b/knowledge-config")
        assert down.status_code == 503
        assert down.json()["detail"]["error"]["code"] == "unavailable_dependency"
        assert client.get("/health/live").status_code == 200
    finally:
        policy_logger.removeHandler(collector)
        policy_enforcement.configure(None)


# ---- the SQLite authority database holds secrets: owner-only file modes


def _mode(path) -> int:
    import os
    import stat

    return stat.S_IMODE(os.stat(path).st_mode)


def test_a_new_authority_database_and_its_wal_files_are_owner_only(tmp_path):
    import os

    from authority_store_sql import SqliteAuthorityStore

    previous = os.umask(0o022)  # the usual umask would make a plain create world-readable
    try:
        path = tmp_path / "authority.db"
        store = SqliteAuthorityStore(str(path))
        try:
            for suffix in ("", "-wal", "-shm"):
                target = f"{path}{suffix}"
                assert os.path.exists(target), target
                assert _mode(target) == 0o600, (suffix, oct(_mode(target)))
        finally:
            store.close()
    finally:
        os.umask(previous)


def test_an_existing_looser_authority_database_is_tightened_on_open(tmp_path):
    import os

    from authority_store_sql import SqliteAuthorityStore

    path = tmp_path / "authority.db"
    SqliteAuthorityStore(str(path)).close()
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(f"{path}{suffix}"):
            os.chmod(f"{path}{suffix}", 0o644)
    store = SqliteAuthorityStore(str(path))
    try:
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(f"{path}{suffix}"):
                assert _mode(f"{path}{suffix}") == 0o600, suffix
    finally:
        store.close()


def test_an_owner_only_database_is_left_as_it_is(tmp_path):
    import os

    from authority_store_sql import SqliteAuthorityStore

    path = tmp_path / "authority.db"
    path.touch(mode=0o600)
    os.chmod(path, 0o400 | 0o200)
    SqliteAuthorityStore(str(path)).close()
    assert _mode(path) == 0o600


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
def test_a_link_at_the_database_or_a_sidecar_path_is_refused_and_its_target_untouched(tmp_path, suffix):
    import os

    from authority_store_sql import SqliteAuthorityStore

    target = tmp_path / "elsewhere"
    target.write_bytes(b"not the authority database")
    os.chmod(target, 0o644)
    path = tmp_path / "authority.db"
    if suffix:
        SqliteAuthorityStore(str(path)).close()  # a real database; only the sidecar is a link
        for name in ("-wal", "-shm"):
            if os.path.exists(f"{path}{name}"):
                os.remove(f"{path}{name}")
    os.symlink(target, f"{path}{suffix}")
    with pytest.raises(OSError):
        SqliteAuthorityStore(str(path))
    assert target.read_bytes() == b"not the authority database" and _mode(target) == 0o644


def test_a_fifo_at_the_database_path_is_refused_without_blocking(tmp_path):
    import os
    import threading

    from authority_store_sql import SqliteAuthorityStore

    path = tmp_path / "authority.db"
    os.mkfifo(path)
    outcome = {}

    def open_store():
        try:
            SqliteAuthorityStore(str(path))
        except Exception as error:  # noqa: BLE001
            outcome["error"] = error

    thread = threading.Thread(target=open_store, daemon=True)
    thread.start()
    thread.join(5)
    assert not thread.is_alive() and isinstance(outcome.get("error"), OSError), outcome
