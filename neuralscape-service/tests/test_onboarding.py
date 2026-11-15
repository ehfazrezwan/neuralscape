"""Membership service over the durable authority store (SQLite here, PostgreSQL in
the integration module): onboarding policy, invites, joining, removal."""

from __future__ import annotations

import threading

import pytest
from pydantic import ValidationError

import onboarding
from authority_store_sql import SqliteAuthorityStore, UnknownSubject
from contracts_policy import PolicyStatement
from onboarding import (
    HOSTED_V1,
    PRESERVE_V1,
    OnboardingPolicy,
    OnboardingService,
    default_statements,
    parse_invite,
)
from tests.test_authority_store_sql import Clock

T = "tenant-a"


def policy_json(**changes) -> dict:
    base = HOSTED_V1.model_dump(mode="python")
    base.update(changes)
    return base


def seed(backend, clock=None):
    store = backend.open(clock or Clock())
    store.add_subject("sub_admin", T, "human", "cred-admin")
    store.add_subject("sub_other", "tenant-b", "human", "cred-other")
    return store, OnboardingService(store, clock=store._clock, relay_enabled=lambda: False)


def actions(statements, handle_kind=None):
    return sorted({s.action for s in statements})


# ── the policy object (pure) ─────────────────────────────────────────────


def test_presets_are_valid_versioned_objects() -> None:
    for preset in onboarding.PRESETS.values():
        assert OnboardingPolicy.model_validate(preset.model_dump(mode="python")) == preset
        assert preset.version == "onboarding-v1"


@pytest.mark.parametrize(
    "changes",
    [
        {"tenant_actions": ("read", "manage_membership")},
        {"private_actions": ("read", "manage_credentials")},
        {"shared_actions": ("read", "publish_standard")},
        {"relay_actions": ("create", "manage_membership")},
        {"tenant_actions": ("read", "nonsense")},
        {"tenant_actions": ("read", "export")},          # tenant-wide export is always an explicit grant
        {"shared_actions": ("read", "update")},          # shared edits only through the row rules
        {"shared_actions": ("read", "delete")},
        {"private_actions": ("read", "read")},
        {"version": "onboarding-v2"},
        {"surprise": True},
    ],
)
def test_a_default_policy_can_never_grant_administrator_or_standards_authority(changes) -> None:
    with pytest.raises(ValidationError):
        OnboardingPolicy.model_validate(policy_json(**changes))


def test_default_statements_follow_the_policy_and_are_deterministic() -> None:
    hosted = default_statements(HOSTED_V1, "sub_n", T)
    again = default_statements(HOSTED_V1, "sub_n", T)
    assert [s.id for s in hosted] == [s.id for s in again]
    assert not {s.action for s in hosted} & {"manage_membership", "manage_credentials", "publish_standard", "share"}
    assert all(s.origin == "onboarding:hosted-v1" for s in hosted)
    assert all(s.condition is None for s in hosted)
    assert {s.subject_id for s in hosted} == {"sub_n"}
    preserved = default_statements(PRESERVE_V1, "sub_n", T)
    assert {"share", "create"} <= {s.action for s in preserved}
    conditioned = {(s.action, s.condition.kind) for s in preserved if s.condition}
    assert conditioned == {("update", "row_owner_is_principal"), ("delete", "row_owner_is_principal"),
                           ("update", "fields_subset_of")}
    assert len({s.id for s in hosted}) == len(hosted)


def test_relay_actions_are_granted_only_when_encrypted_sync_is_enabled() -> None:
    tenant_level = lambda s: s.resource.resolver == "ns:tenant"  # noqa: E731
    off = {s.action for s in default_statements(HOSTED_V1, "sub_n", T, relay_enabled=False) if tenant_level(s)}
    on = {s.action for s in default_statements(HOSTED_V1, "sub_n", T, relay_enabled=True) if tenant_level(s)}
    assert off == {"read"} and on == {"read", "create", "list", "update"}


def test_tenant_read_is_part_of_the_hosted_defaults_and_tenant_export_is_not() -> None:
    assert {"read"} == {s.action for s in default_statements(HOSTED_V1, "sub_n", T)
                        if s.resource.resolver == "ns:tenant"}
    assert {"read"} == {s.action for s in default_statements(PRESERVE_V1, "sub_n", T)
                        if s.resource.resolver == "ns:tenant"}


def test_invite_tokens_parse_strictly() -> None:
    assert parse_invite("nsi_inv_abc.secret") is not None
    for bad in ("", None, 5, "nsi_inv_abc", "nsi_abc.secret", "nsc_inv_abc.secret", "nsi_inv_abc.", "x" * 300):
        assert parse_invite(bad) is None


# ── store/service cases (shared with the PostgreSQL module) ──────────────


def case_a_member_joins_with_a_single_use_invite_and_receives_the_default_grants(backend):
    store, svc = seed(backend)
    issued = svc.invite("sub_admin")
    epoch_before = (store.current_policy(T) or type("P", (), {"policy_epoch": 0})()).policy_epoch
    joined = svc.join("newcomer", issued.token)
    assert joined.tenant_id == T and joined.policy_name == "hosted-v1" and joined.policy_epoch == epoch_before + 1
    assert svc.subject_for_user_id("newcomer") == joined.subject_id
    record = store.subject_record(joined.subject_id)
    assert (record.tenant_id, record.subject_kind, record.membership_version) == (T, "human", 1)
    mine = [s for s in store.current_policy(T).statements if s.subject_id == joined.subject_id]
    assert {s.id for s in mine} == {s.id for s in default_statements(HOSTED_V1, joined.subject_id, T)}
    with pytest.raises(ValueError):                                      # single use
        svc.join("someone-else", issued.token)
    assert svc.subject_for_user_id("someone-else") is None
    assert {m["subject_id"] for m in svc.list_members(T)} == {"sub_admin", joined.subject_id}


def case_concurrent_joins_with_one_invite_admit_exactly_one(backend):
    store, svc = seed(backend)
    issued = svc.invite("sub_admin")
    peers = [OnboardingService(backend.open(store._clock), clock=store._clock) for _ in range(6)]
    results: list[object] = []
    barrier = threading.Barrier(len(peers))

    def go(index, peer):
        barrier.wait()
        try:
            results.append(peer.join(f"user-{index}", issued.token))
        except Exception as error:  # noqa: BLE001
            results.append(error)

    threads = [threading.Thread(target=go, args=(i, p)) for i, p in enumerate(peers)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len([r for r in results if isinstance(r, onboarding.JoinResult)]) == 1
    assert len(svc.list_members(T)) == 2


def case_unusable_invites_are_refused_and_nothing_is_created(backend):
    clock = Clock()
    store, svc = seed(backend, clock)
    issued = svc.invite("sub_admin", ttl=60)
    invite_id, _, secret = issued.token[len(onboarding.INVITE_PREFIX):].partition(".")
    for bad in (f"nsi_{invite_id}.{secret}x", f"nsi_inv_unknown.{secret}", "garbage", ""):
        with pytest.raises(ValueError):
            svc.join("newcomer", bad)
    clock.now += 61
    with pytest.raises(ValueError):                                      # expired
        svc.join("newcomer", issued.token)
    fresh = svc.invite("sub_admin")
    assert store.revoke_invite(fresh.invite_id, tenant_id=T) is True
    with pytest.raises(ValueError):                                      # revoked
        svc.join("newcomer", fresh.token)
    assert store.revoke_invite(fresh.invite_id, tenant_id="tenant-b") is False
    assert svc.subject_for_user_id("newcomer") is None and len(svc.list_members(T)) == 1


def case_invites_are_created_only_by_members_with_bounded_lifetimes(backend):
    store, svc = seed(backend)
    with pytest.raises(UnknownSubject):
        svc.invite("sub_missing")
    for bad_ttl in (0, -5, onboarding.MAX_INVITE_TTL + 1, float("nan")):
        with pytest.raises(ValueError):
            svc.invite("sub_admin", ttl=bad_ttl)
    store.offboard_subject("sub_admin")
    with pytest.raises(UnknownSubject):
        svc.invite("sub_admin")
    with pytest.raises(ValueError):                                      # inviter must belong to the tenant
        store.create_invite(tenant_id=T, created_by="sub_other", invite_id="inv_x", secret_hash="h", expires_at=10**12)


def case_an_identity_that_is_already_bound_cannot_join_and_the_invite_survives(backend):
    store, svc = seed(backend)
    store.put_if_absent("local", "taken", "sub_other")                    # bound to another tenant's subject
    issued = svc.invite("sub_admin")
    with pytest.raises(ValueError):
        svc.join("taken", issued.token)
    assert store.invite_tenant(*parse_invite(issued.token)) == T            # the failed join consumed nothing
    assert len(svc.list_members(T)) == 1
    assert svc.join("fresh", issued.token).tenant_id == T


def case_removing_a_member_offboards_them_and_drops_their_grants(backend):
    store, svc = seed(backend)
    joined = svc.join("leaver", svc.invite("sub_admin").token)
    stayer = svc.join("stayer", svc.invite("sub_admin").token)
    account = store.create_service_account("sub_sa", T, joined.subject_id, "ci")
    before = store.current_policy(T).policy_epoch
    epoch = svc.remove_member(T, joined.subject_id, by_subject_id="sub_admin")
    assert epoch == before + 1
    after = store.current_policy(T)
    assert not [s for s in after.statements if s.subject_id == joined.subject_id]
    assert [s for s in after.statements if s.subject_id == stayer.subject_id]          # others keep theirs
    assert store.subject_record(joined.subject_id) is None
    assert store.service_account("sub_sa")["status"] == "disabled"                   # what they controlled goes too
    assert svc.subject_for_user_id("leaver") == joined.subject_id                      # the id is tombstoned, not reused
    assert {m["subject_id"] for m in svc.list_members(T)} == {"sub_admin", stayer.subject_id}
    for call in (lambda: svc.remove_member(T, joined.subject_id, by_subject_id="sub_admin"),   # again
                 lambda: svc.remove_member(T, "sub_other", by_subject_id="sub_admin"),         # another tenant
                 lambda: svc.remove_member(T, "sub_missing", by_subject_id="sub_admin")):
        with pytest.raises(UnknownSubject):
            call()
    with pytest.raises(ValueError):
        svc.remove_member(T, "sub_admin", by_subject_id="sub_admin")                  # not yourself
    again = svc.invite("sub_admin")
    with pytest.raises(ValueError):                                                    # a removed identity cannot silently rejoin
        svc.join("leaver", again.token)


def case_the_onboarding_policy_is_stored_per_tenant_and_defaults_to_hosted(backend):
    store, svc = seed(backend)
    assert svc.policy_for(T) == HOSTED_V1
    assert svc.set_policy(T, "preserve-v1") == PRESERVE_V1
    assert svc.policy_for(T) == PRESERVE_V1 and svc.policy_for("tenant-b") == HOSTED_V1
    custom = OnboardingPolicy.model_validate(policy_json(name="custom", tenant_actions=("read",)))
    svc.set_policy(T, custom)
    assert svc.policy_for(T) == custom
    for bad in ("nonsense", None):
        with pytest.raises((ValueError, ValidationError)):
            svc.set_policy(T, bad)  # type: ignore[arg-type]
    joined = svc.join("under-custom", svc.invite("sub_admin").token)
    mine = [s for s in store.current_policy(T).statements if s.subject_id == joined.subject_id]
    assert joined.policy_name == "custom" and {s.action for s in mine if s.resource.resolver == "ns:tenant"} == {"read"}


def case_grants_can_be_appended_atomically_and_idempotently(backend):
    store, svc = seed(backend)
    extra = default_statements(HOSTED_V1, "sub_admin", T)
    epoch = store.grant_statements(T, extra)
    assert store.grant_statements(T, extra) == epoch + 1                      # same ids: no duplicates, epoch still moves
    ids = [s.id for s in store.current_policy(T).statements]
    assert len(ids) == len(set(ids)) == len(extra)
    clash = PolicyStatement(id=extra[0].id, effect="deny", subject_id="sub_admin", credential_id=None,
                            action="read", resource=extra[0].resource)
    with pytest.raises(ValueError):
        store.grant_statements(T, [clash])


def _may(store, subject_id, action, handle) -> bool:
    import policy_enforcement
    from contracts_policy import PolicyEvaluationInput
    from contracts_policy_reference import evaluate_policy

    principal = policy_enforcement._build_principal(subject_id, store.subject_record(subject_id))
    decision = evaluate_policy(
        principal=principal,
        evaluation=PolicyEvaluationInput(schema_version="candidate-v1", action=action, resource=handle),
        policy=store.current_policy(T),
    )
    return decision.outcome == "allow"


def case_a_default_member_cannot_export_the_tenant_until_it_is_granted_explicitly(backend):
    from policy_enforcement import pool_handle, tenant_handle

    store, svc = seed(backend)
    one = svc.join("one", svc.invite("sub_admin").token)
    two = svc.join("two", svc.invite("sub_admin").token)
    for member in (one, two):
        assert _may(store, member.subject_id, "read", tenant_handle(T))                       # sign-in still works
        assert not _may(store, member.subject_id, "export", tenant_handle(T))               # the whole-tenant export does not
        assert _may(store, member.subject_id, "export", pool_handle(T, "private", member.subject_id))  # own data does
    svc.set_policy(T, "preserve-v1")
    three = svc.join("three", svc.invite("sub_admin").token)
    assert not _may(store, three.subject_id, "export", tenant_handle(T))
    epoch = svc.grant_tenant_export(T, one.subject_id)
    assert epoch == store.current_policy(T).policy_epoch
    assert _may(store, one.subject_id, "export", tenant_handle(T))                           # only the named member
    assert not _may(store, two.subject_id, "export", tenant_handle(T))
    assert not _may(store, three.subject_id, "export", tenant_handle(T))
    svc.grant_tenant_export(T, one.subject_id)                                               # idempotent
    assert len([s for s in store.current_policy(T).statements
                if s.action == "export" and s.resource.resolver == "ns:tenant"]) == 1
    for bad in ("sub_other", "sub_missing"):                                                 # another tenant, unknown
        with pytest.raises(UnknownSubject):
            svc.grant_tenant_export(T, bad)


def _make_admin(store, svc, user):
    import policy_enforcement

    joined = svc.join(user, svc.invite("sub_admin").token)
    store.grant_statements(T, [PolicyStatement(
        id=f"adm-{joined.subject_id}", effect="allow", subject_id=joined.subject_id, credential_id=None,
        action="manage_membership", resource=policy_enforcement._handle("membership", T, T))])
    return joined.subject_id


def _grant_admin(store):
    import policy_enforcement

    store.grant_statements(T, [PolicyStatement(
        id="adm-first", effect="allow", subject_id="sub_admin", credential_id=None,
        action="manage_membership", resource=policy_enforcement._handle("membership", T, T))])


def case_the_last_administrator_cannot_be_removed(backend):
    from onboarding import LastAdministrator

    store, svc = seed(backend)
    _grant_admin(store)
    member = svc.join("plain", svc.invite("sub_admin").token).subject_id
    second = _make_admin(store, svc, "second")
    # a non-administrator removing the only other administrator would leave none... after the
    # first removal the second is the only one left:
    svc.remove_member(T, "sub_admin", by_subject_id=second)
    with pytest.raises(LastAdministrator):
        svc.remove_member(T, second, by_subject_id=member)            # nobody administers afterwards
    assert store.subject_record(second) is not None                  # and nothing was removed
    assert any(s.action == "manage_membership" and s.subject_id == second
               for s in store.current_policy(T).statements)
    svc.remove_member(T, member, by_subject_id=second)               # removing a non-administrator is fine


def case_two_administrators_removing_each_other_at_once_leave_one(backend):
    from onboarding import LastAdministrator

    first_store, first_svc = seed(backend)
    _grant_admin(first_store)
    second = _make_admin(first_store, first_svc, "second")
    second_store = backend.open()
    second_svc = OnboardingService(second_store, clock=second_store._clock, relay_enabled=lambda: False)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def remove(svc, target, by):
        barrier.wait()
        try:
            svc.remove_member(T, target, by_subject_id=by)
            outcomes.append("removed")
        except LastAdministrator:
            outcomes.append("refused")
        except UnknownSubject:
            outcomes.append("gone")

    threads = [threading.Thread(target=remove, args=(first_svc, second, "sub_admin")),
               threading.Thread(target=remove, args=(second_svc, "sub_admin", second))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("removed") == 1, outcomes
    remaining = [s for s in first_store.current_policy(T).statements if s.action == "manage_membership"]
    assert len(remaining) == 1                                          # exactly one administrator is left
    survivor = remaining[0].subject_id
    assert first_store.subject_record(survivor) is not None


def case_outstanding_invites_are_capped_per_tenant(backend):
    store, svc = seed(backend)
    issued = [svc.invite("sub_admin") for _ in range(onboarding.MAX_OUTSTANDING_INVITES)]
    with pytest.raises(onboarding.TooManyInvites):
        svc.invite("sub_admin")
    svc.join("someone", issued[0].token)                                 # a used invite frees a slot
    assert svc.invite("sub_admin")
    with pytest.raises(onboarding.TooManyInvites):
        svc.invite("sub_admin")
    other = backend.open()
    other.add_subject("sub_b2", "tenant-c", "human", "cred-b2")
    assert OnboardingService(other, relay_enabled=lambda: False).invite("sub_b2")   # per tenant, not global


def case_removing_an_administrator_revokes_the_invites_they_issued(backend):
    store, svc = seed(backend)
    _grant_admin(store)
    second = _make_admin(store, svc, "second")
    pending = svc.invite(second)                                     # issued by the administrator to be removed
    kept = svc.invite("sub_admin")                                   # issued by the one who stays
    svc.remove_member(T, second, by_subject_id="sub_admin")
    with pytest.raises(ValueError):                                  # the removed administrator's invite is dead
        svc.join("mallory", pending.token)
    assert svc.subject_for_user_id("mallory") is None
    assert svc.join("newcomer", kept.token).subject_id               # other invites are untouched


class _LookAlike(str):
    """An identifier that claims to equal every other one."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = str.__hash__


def case_identifiers_that_compare_loosely_are_refused(backend):
    store, svc = seed(backend)
    _grant_admin(store)
    member = svc.join("plain", svc.invite("sub_admin").token).subject_id
    before = store.current_policy(T).statements
    with pytest.raises(ValueError):
        svc.remove_member(T, _LookAlike(member), by_subject_id="sub_admin")
    with pytest.raises(ValueError):
        svc.remove_member(T, member, by_subject_id=_LookAlike("sub_admin"))
    with pytest.raises(ValueError):                  # another tenant's member, through a loose tenant id
        svc.grant_tenant_export(_LookAlike(T), "sub_other")
    with pytest.raises(ValueError):
        svc.revoke_invite(_LookAlike("tenant-b"), _LookAlike("inv_x"))
    assert store.current_policy(T).statements == before
    assert store.current_policy("tenant-b") is None


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
    backend = SqliteBackend(str(tmp_path / "onboarding.db"))
    yield backend
    backend.close()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.__name__)
def test_sqlite_onboarding_conformance(case, sqlite_backend) -> None:
    case(sqlite_backend)


def test_the_migration_stores_the_preserving_policy_without_overwriting_a_choice(tmp_path) -> None:
    import policy_migration as pm

    store = SqliteAuthorityStore(str(tmp_path / "m.db"))
    try:
        inventory = pm.collect_inventory(pools=[pm.PoolObservation("alice", "private", None, None)])
        pm.apply_plan(store, inventory, T)
        assert OnboardingService(store).policy_for(T) == PRESERVE_V1
        OnboardingService(store).set_policy(T, "hosted-v1")                  # an operator's later choice
        pm.apply_plan(store, inventory, T)
        assert OnboardingService(store).policy_for(T) == HOSTED_V1
    finally:
        store.close()


def test_the_feature_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ONBOARDING_ENABLED", raising=False)
    assert onboarding.enabled() is False
    monkeypatch.setenv("ONBOARDING_ENABLED", "1")
    assert onboarding.enabled() is True


# ── a joined member's saves can be derived: the background stage needs its own grant ──


class _Sources:
    """The source row as the resolver sees it now: one record in one pool."""

    def __init__(self, namespace):
        from publication_fence import CurrentSource

        self._row = CurrentSource(1, namespace, namespace.pool_kind)

    def current(self, tenant_id, record_id):
        return self._row


def _admission(subject_id, namespace, epoch, version=1):
    from publication_fence import Admission

    return Admission(
        tenant_id=T, subject_id=subject_id, credential_id=f"token:{subject_id}", membership_version=version,
        policy_epoch=epoch, source_record_id="rec-1", source_revision=1, target_namespace=namespace,
    )


@pytest.mark.parametrize("preset", ["hosted-v1", "preserve-v1"])
def test_a_member_joined_with_the_default_grants_can_complete_a_save(tmp_path, preset) -> None:
    from identity_binding import NamespaceKey
    from publication_fence import PublicationFence

    store = SqliteAuthorityStore(str(tmp_path / f"{preset}.db"))
    store.add_subject("sub_admin", T, "human", "cred-admin")
    svc = OnboardingService(store, clock=store._clock, relay_enabled=lambda: False)
    svc.set_policy(T, preset)
    member = svc.join("joined", svc.invite("sub_admin").token)
    epoch = store.current_policy(T).policy_epoch
    own = NamespaceKey(T, "private", member.subject_id, None, None)
    decision = PublicationFence(store, _Sources(own)).evaluate(_admission(member.subject_id, own, epoch))
    assert decision.allowed, decision.reason                       # the stage that used to fail not_authorized
    # and nothing wider: the same member may not derive in another member's private pool or in a pool
    # the preset does not give them (the shared pool, under either preset: a separate decision)
    for other in (NamespaceKey(T, "private", "sub_admin", None, None), NamespaceKey(T, "shared", None, None, None),
                  NamespaceKey(T, "private", member.subject_id, "proj-1", None)):
        refused = PublicationFence(store, _Sources(other)).evaluate(_admission(member.subject_id, other, epoch))
        assert not refused.allowed, other
    store.close()


def test_the_default_background_grant_is_one_action_on_the_members_own_private_pool() -> None:
    from policy_enforcement import pool_handle

    for preset in (HOSTED_V1, PRESERVE_V1):
        derive = [s for s in default_statements(preset, "sub_n", T) if s.action.startswith("background_")]
        assert [(s.action, s.resource, s.condition) for s in derive] == [
            ("background_derive", pool_handle(T, "private", "sub_n"), None)
        ], preset.name


def test_the_background_actions_are_checked_only_by_background_operations_and_two_initiations() -> None:
    """The registry listing that bounds what the new default can reach.

    ``background_derive`` is checked by the worker's operations (queue functions and crons) and by
    exactly two user-facing initiations of a dreaming sweep.  The sweep is a system job (an omitted
    pool walks every pool), so both initiations check the action on the tenant, which no default
    grant names: a joined member, who holds it only on their own private pool, reaches neither.
    Any other operation appearing here is a widening and must be a reviewed change.
    """

    from policy_registry import REGISTRY

    background = {"background_derive", "background_replay"}
    carriers = {operation_id for operation_id, entry in REGISTRY.items() if entry.action in background}
    user_facing = sorted(i for i in carriers if not i.startswith(("arq_function:", "arq_cron:")))
    assert user_facing == [
        "mcp_tool:schedule_dream",
        "rest_extension:POST /v1/extensions/dreaming/run",
    ]
    # Elsewhere a user-facing operation names background work only as an ADDED check for a scoped
    # credential on its own target (durable work is fenced under it later): a narrowing, never a grant.
    for operation_id, entry in REGISTRY.items():
        if operation_id.startswith(("arq_function:", "arq_cron:")):
            continue
        extras = [(action, condition) for action, condition in entry.extra_actions if action in background]
        assert extras in ([], [("background_derive", "credential=scoped")]), operation_id
    # both initiations are tenant-level: a member's own-pool grant never reaches them
    assert all((REGISTRY[i].resolver_kind, REGISTRY[i].resolver_target) == ("tenant", "credential.tenant")
               for i in user_facing)


# ── a default member cannot start a dreaming sweep; an administrator granted it on the tenant can ──


_DREAM_OPERATIONS = ("mcp_tool:schedule_dream", "rest_extension:POST /v1/extensions/dreaming/run")


def _dream_world(tmp_path, preset):
    from policy_enforcement import PolicyEnforcer

    store = SqliteAuthorityStore(str(tmp_path / f"dream-{preset}.db"))
    svc = OnboardingService(store, clock=store._clock, relay_enabled=lambda: False)
    from identity_binding import bind_identity

    admin = bind_identity(store, "local", "admin").subject_id
    store.add_subject(admin, T, "human", "cred-admin")
    svc.set_policy(T, preset)
    dana = svc.join("dana", svc.invite(admin).token).subject_id
    svc.join("erin", svc.invite(admin).token)
    return store, PolicyEnforcer(store, store), admin, dana


def _decide(enforcer, operation_id, user_id, request):
    from policy_enforcement import RequestContext

    if operation_id.startswith("mcp_tool:"):
        context = RequestContext(user_id=user_id, arguments=request)
    else:
        context = RequestContext(user_id=user_id, body=request)
    return enforcer.decide(operation_id, context, mode="enforce", profile="legacy_compat").allowed


_DREAM_REQUESTS = (
    {},                                   # omitted pool: the worker sweeps every pool
    {"force": True},
    {"pool": "shared"},
    {"pool": "standard"},
    {"pool": "user--erin"},               # another member's private pool
    {"pool": "user--dana"},
    {"pool": "user--erin", "force": True},
    {"project_id": "p1"},
)


@pytest.mark.parametrize("preset", ["hosted-v1", "preserve-v1"])
@pytest.mark.parametrize("operation_id", _DREAM_OPERATIONS)
def test_a_default_member_cannot_start_any_dreaming_sweep(tmp_path, preset, operation_id) -> None:
    store, enforcer, _admin, _dana = _dream_world(tmp_path, preset)
    for request in _DREAM_REQUESTS:
        assert not _decide(enforcer, operation_id, "dana", request), (operation_id, request)
    store.close()


@pytest.mark.parametrize("operation_id", _DREAM_OPERATIONS)
def test_an_administrator_granted_tenant_derivation_can_start_a_sweep(tmp_path, operation_id) -> None:
    from policy_enforcement import privileged_handle, tenant_handle

    store, enforcer, admin, _dana = _dream_world(tmp_path, "hosted-v1")
    assert not _decide(enforcer, operation_id, "admin", {})  # nothing is implied for the administrator
    # An ordinary tenant-level grant is not enough: the operation is administrator-only (privileged handle).
    store.grant_statements(T, [PolicyStatement(
        id="dream-plain", effect="allow", subject_id=admin, credential_id=None,
        action="background_derive", resource=tenant_handle(T),
    )])
    assert not _decide(type(enforcer)(store, store), operation_id, "admin", {})
    store.grant_statements(T, [PolicyStatement(
        id="dream-admin", effect="allow", subject_id=admin, credential_id=None,
        action="background_derive", resource=privileged_handle("background_derive", tenant_handle(T)),
    )])
    enforcer = type(enforcer)(store, store)
    for request in ({}, {"pool": "shared"}, {"pool": "user--erin", "force": True}):
        assert _decide(enforcer, operation_id, "admin", request), request
    store.close()


def test_the_save_path_stays_allowed_for_a_default_member(tmp_path) -> None:
    from identity_binding import NamespaceKey
    from publication_fence import PublicationFence

    store, _enforcer, _admin, dana = _dream_world(tmp_path, "hosted-v1")
    own = NamespaceKey(T, "private", dana, None, None)
    epoch = store.current_policy(T).policy_epoch
    assert PublicationFence(store, _Sources(own)).evaluate(_admission(dana, own, epoch)).allowed
    store.close()


# ── tenant-level grants in the relay list are tenant-level grants ────────


def test_relay_actions_cannot_carry_tenant_wide_export() -> None:
    """``relay_actions`` are granted on the tenant handle, so ``export`` there is a whole-tenant export."""

    with pytest.raises(ValidationError, match="export"):
        OnboardingPolicy.model_validate(policy_json(relay_actions=("create", "export")))


def test_no_valid_policy_grants_tenant_export_even_with_the_relay_on() -> None:
    for preset in onboarding.PRESETS.values():
        statements = default_statements(preset, "sub_n", T, relay_enabled=True)
        assert not [s for s in statements if s.resource.kind == "tenant" and s.action == "export"]


@pytest.mark.parametrize("field", ["tenant_actions", "relay_actions"])
@pytest.mark.parametrize("action", ["restore", "background_derive", "background_replay", "delete", "share"])
def test_tenant_level_defaults_are_limited_to_an_explicit_allowlist(field, action) -> None:
    """Tenant-level ``restore`` replaces tenant data and the others act tenant-wide: none of them is
    ever a default.  Tenant-level defaults are an allowlist, so a new tenant-level action is refused
    until it is deliberately added."""

    base = ("read",) if field == "tenant_actions" else ("create", "list", "update")
    with pytest.raises(ValidationError, match="tenant-level"):
        OnboardingPolicy.model_validate(policy_json(**{field: (*base, action)}))


def test_the_presets_stay_inside_the_tenant_level_allowlist() -> None:
    for preset in onboarding.PRESETS.values():
        assert set(preset.tenant_actions) | set(preset.relay_actions) <= onboarding.TENANT_DEFAULT_ACTIONS
