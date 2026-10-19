"""Flagged action-policy enforcement: two tenants, human and service-account
principals, admin/content separation, legacy profile limits, fail-closed."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

import policy_enforcement
from authority_store import InMemoryAuthorityStore, SubjectRecord
from config import settings
from contracts_policy import (
    AccessPolicy,
    MembershipVersion,
    PolicyEvaluationInput,
    PolicyStatement,
    PrincipalContext,
)
from contracts_policy_reference import evaluate_policy
from identity_binding import InMemoryBindingStore, bind_identity
from policy_enforcement import PolicyEnforcer, RowInfo, enforce_request, pool_handle
from policy_registry import REGISTRY

CONTENT = ("read", "search", "list", "create", "update", "delete", "export")
SECRET_TEXT = "highly-sensitive-body-text"


class World:
    """Two tenants; humans alice (A) and bob (B), service account svc (A),
    a dictator-like standards publisher dave (A), and the legacy key principal."""

    def __init__(self) -> None:
        self.bindings = InMemoryBindingStore()
        self.authority = InMemoryAuthorityStore()
        self.ids: dict[str, str] = {}
        self._statements: dict[str, list[PolicyStatement]] = {"A": [], "B": []}
        self._members: dict[str, list[tuple[str, int]]] = {"A": [], "B": []}
        self._count = 0

    def member(self, key, tenant, kind="human", *, issuer="local", subject=None, credential="cred") -> str:
        subject_id = bind_identity(self.bindings, issuer, subject or key).subject_id
        self.ids[key] = subject_id
        self.authority.set_subject(
            subject_id, SubjectRecord(f"tenant-{tenant}", kind, f"{credential}-{key}", 1)
        )
        self._members[tenant].append((subject_id, 1))
        return subject_id

    def grant(self, key, tenant, actions, handle) -> None:
        for action in actions:
            self._count += 1
            self._statements[tenant].append(
                PolicyStatement(
                    id=f"s{self._count}",
                    effect="allow",
                    subject_id=self.ids[key],
                    credential_id=None,
                    action=action,
                    resource=handle,
                )
            )

    def publish(self) -> None:
        for tenant in ("A", "B"):
            self.authority.set_policy(
                AccessPolicy(
                    schema_version="candidate-v1",
                    tenant_id=f"tenant-{tenant}",
                    policy_epoch=4,
                    membership_versions=tuple(
                        MembershipVersion(subject_id=s, version=v)
                        for s, v in self._members[tenant]
                    ),
                    statements=tuple(self._statements[tenant]),
                )
            )


def pool(world: World, tenant: str, kind: str, owner: str | None = None, **kw):
    return pool_handle(f"tenant-{tenant}", kind, world.ids[owner] if owner else None, **kw)


@pytest.fixture
def world() -> World:
    w = World()
    for key, tenant, kind in (
        ("alice", "A", "human"),
        ("svc", "A", "service_account"),
        ("dave", "A", "human"),
        ("bob", "B", "human"),
    ):
        w.member(key, tenant, kind)
    w.member("legacy", "A", "service_account", subject="svc:legacy-shared-key")
    w.grant("alice", "A", CONTENT, pool(w, "A", "private", "alice"))
    w.grant("alice", "A", ("read", "search", "list", "create", "share"), pool(w, "A", "shared"))
    w.grant("alice", "A", ("count",), policy_enforcement._handle("tenant", "tenant-A", "tenant-A"))
    w.grant("svc", "A", ("read", "search", "list"), pool(w, "A", "shared"))
    w.grant("dave", "A", ("create", "publish_standard"), pool(w, "A", "standard"))
    w.grant("bob", "B", CONTENT, pool(w, "B", "private", "bob"))
    w.grant("bob", "B", ("count",), policy_enforcement._handle("tenant", "tenant-B", "tenant-B"))
    # The legacy key principal reproduces alice's private capability explicitly
    # and, deliberately, holds admin/standard grants that must NOT be usable.
    w.grant("legacy", "A", CONTENT, pool(w, "A", "private", "alice"))
    w.grant("legacy", "A", ("create", "publish_standard"), pool(w, "A", "standard"))
    w.grant("legacy", "A", ("manage_credentials",), policy_enforcement._handle("connector", "new", "tenant-A"))
    w.publish()
    return w


ROWS = {"memory-a": RowInfo("private", None, None, None, "private")}


def make_client(world: World, row_lookup=None) -> TestClient:
    app = FastAPI(dependencies=[Depends(enforce_request)])

    @app.middleware("http")
    async def identify(request: Request, call_next):
        request.state.user_id = request.headers.get("x-test-user")
        return await call_next(request)

    # Routes use the real registry paths/methods; handlers only echo.
    @app.post("/v1/search")
    async def search(payload: dict | None = None):
        return {"ok": "search"}

    @app.get("/v1/memories")
    async def list_memories():
        return {"ok": "list"}

    @app.get("/memories")
    async def legacy_list():
        return {"ok": "legacy-list"}

    @app.get("/v1/metrics")
    async def metrics():
        return {"ok": "count"}

    @app.post("/v1/memories/raw")
    async def raw(payload: dict | None = None):
        return {"ok": "raw"}

    @app.post("/v1/connectors")
    async def connector(payload: dict | None = None):
        return {"ok": "connector"}

    @app.get("/v1/memories/{memory_id}")
    async def get_memory(memory_id: str):
        return {"ok": "get"}

    @app.get("/v1/categories")
    async def categories():
        return {"ok": "categories"}

    @app.get("/not/in/registry")
    async def unregistered():
        return {"ok": "never"}

    policy_enforcement.configure(
        PolicyEnforcer(world.authority, world.bindings, row_lookup=row_lookup)
    )
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
    monkeypatch.setattr(settings, "ns_auth_profile", "legacy_compat")
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", "secret")
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    yield
    policy_enforcement.configure(None)


def as_user(client, user, method, path, **kw):
    headers = {"x-test-user": user} if user else {}
    return client.request(method, path, headers=headers, **kw)


# ── healthy positive controls ────────────────────────────────────────────


def test_human_allowed_on_own_private_pool_for_each_action_class(world) -> None:
    client = make_client(world)
    assert as_user(client, "alice", "POST", "/v1/search", json={"query": "x"}).status_code == 200
    assert as_user(client, "alice", "GET", "/v1/memories").status_code == 200
    assert as_user(client, "alice", "GET", "/memories?user_id=alice").status_code == 200
    assert as_user(client, "alice", "GET", "/v1/metrics").status_code == 200  # count
    assert as_user(client, "alice", "POST", "/v1/memories/raw", json={"visibility": "private"}).status_code == 200
    assert as_user(client, "bob", "GET", "/v1/memories").status_code == 200


def test_service_account_gets_only_its_own_grants(world) -> None:
    client = make_client(world)
    shared = {"visibility": "shared"}
    assert as_user(client, "svc", "POST", "/v1/search", json=shared).status_code == 200
    # Own private pool has no grant; shared pool allows read but not create.
    assert as_user(client, "svc", "GET", "/v1/memories").status_code == 404
    assert as_user(client, "svc", "POST", "/v1/memories/raw", json=shared).status_code == 404
    assert as_user(client, "svc", "GET", "/v1/metrics").status_code == 404


# ── tenant and principal boundaries ──────────────────────────────────────


def test_cross_tenant_read_list_and_count_are_denied(world) -> None:
    rows = {"memory-a": RowInfo("private", world.ids["alice"], None, None, "private")}
    client = make_client(world, row_lookup=lambda kind, ident: rows.get(ident))
    # Positive controls for the owner.
    assert as_user(client, "alice", "GET", "/v1/memories/memory-a").status_code == 200
    assert as_user(client, "alice", "GET", "/memories?user_id=alice").status_code == 200
    assert as_user(client, "alice", "GET", "/v1/metrics").status_code == 200
    # The other tenant's principal is denied each of read, list and count.
    assert as_user(client, "bob", "GET", "/v1/memories/memory-a").status_code == 404
    assert as_user(client, "bob", "GET", "/memories?user_id=alice").status_code == 404
    world.authority.set_policy(  # bob loses even the count grant he had
        AccessPolicy(
            schema_version="candidate-v1",
            tenant_id="tenant-B",
            policy_epoch=5,
            membership_versions=(MembershipVersion(subject_id=world.ids["bob"], version=1),),
            statements=(),
        )
    )
    assert as_user(client, "bob", "GET", "/v1/metrics").status_code == 404


def test_unknown_user_and_non_member_are_denied(world) -> None:
    client = make_client(world)
    assert as_user(client, "mallory", "GET", "/v1/memories").status_code == 404
    world.authority.remove_subject(world.ids["alice"])
    assert as_user(client, "alice", "GET", "/v1/memories").status_code == 404


def test_stale_membership_version_is_denied(world) -> None:
    client = make_client(world)
    world.authority.set_subject(
        world.ids["alice"],
        SubjectRecord("tenant-A", "human", "cred-alice", 2),  # policy still says 1
    )
    assert as_user(client, "alice", "GET", "/v1/memories").status_code == 404


def test_row_dependent_operations_deny_without_a_row_lookup(world) -> None:
    client = make_client(world)  # no lookup: the target is unresolved
    assert as_user(client, "alice", "GET", "/v1/memories/memory-a").status_code == 404


# ── admin / content / standard separation ────────────────────────────────


def test_content_grants_never_imply_admin_authority(world) -> None:
    client = make_client(world)
    for user in ("alice", "dave", "svc"):
        assert as_user(client, user, "POST", "/v1/connectors", json={}).status_code == 404
    principal = PrincipalContext(
        schema_version="candidate-v1",
        tenant_id="tenant-A",
        subject_id=world.ids["alice"],
        subject_kind="human",
        credential_id="cred-alice",
        membership_version=1,
        device_id=None,
        delegation=None,
    )
    owned = pool(world, "A", "private", "alice")
    policy = world.authority.current_policy("tenant-A")
    for action in ("manage_membership", "manage_credentials", "publish_standard"):
        decision = evaluate_policy(
            principal=principal,
            evaluation=PolicyEvaluationInput(
                schema_version="candidate-v1", action=action, resource=owned
            ),
            policy=policy,
        )
        assert decision.outcome == "deny", action


def test_standard_and_share_require_their_own_destination_grants(world) -> None:
    client = make_client(world)
    standard = {"visibility": "standard"}
    shared = {"visibility": "shared"}
    # alice may create in shared and holds share there; not standard.
    assert as_user(client, "alice", "POST", "/v1/memories/raw", json=shared).status_code == 200
    assert as_user(client, "alice", "POST", "/v1/memories/raw", json=standard).status_code == 404
    # dave may publish standards but has no grant on a shared pool.
    assert as_user(client, "dave", "POST", "/v1/memories/raw", json=standard).status_code == 200
    assert as_user(client, "dave", "POST", "/v1/memories/raw", json=shared).status_code == 404
    # An unspecified visibility may default to shared, so it needs share too.
    assert as_user(client, "alice", "POST", "/v1/memories/raw", json={}).status_code == 200
    world.grant("svc", "A", ("create",), pool(world, "A", "private", "svc"))
    world.publish()
    assert as_user(client, "svc", "POST", "/v1/memories/raw", json={}).status_code == 404


# ── legacy compatibility profile ─────────────────────────────────────────


def test_legacy_key_principal_cannot_perform_admin_credential_or_standard(
    world, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(settings, "neuralscape_api_key", "shared-key")
    client = make_client(world)
    # Ordinary capability reproduced by explicit grants; the asserted subject is audited only.
    with caplog.at_level(logging.INFO, logger="neuralscape.policy"):
        ok = as_user(client, None, "GET", "/memories?user_id=alice")
    assert ok.status_code == 200
    record = next(r for r in caplog.records if r.getMessage() == "policy_decision")
    assert record.policy_subject == world.ids["legacy"]
    assert record.policy_asserted_subject is None  # query assertions are not body assertions
    # Even though explicit grants exist, these three are never available.
    assert as_user(client, None, "POST", "/v1/connectors", json={"user_id": "alice"}).status_code == 404
    assert (
        as_user(client, None, "POST", "/v1/memories/raw", json={"visibility": "standard", "user_id": "alice"}).status_code
        == 404
    )
    # Ordinary body assertions are recorded with provenance, never bound.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="neuralscape.policy"):
        assert as_user(client, None, "POST", "/v1/search", json={"user_id": "alice"}).status_code == 200
    record = next(r for r in caplog.records if r.getMessage() == "policy_decision")
    assert record.policy_asserted_subject == "alice"
    assert record.policy_assertion_provenance == "legacy_assertion"
    assert record.policy_subject == world.ids["legacy"]


def test_enforced_profile_refuses_shared_key_and_auth_off(world, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ns_auth_profile", "enforced")
    monkeypatch.setattr(settings, "neuralscape_api_key", "shared-key")
    client = make_client(world)
    assert as_user(client, None, "GET", "/v1/memories").status_code == 404
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", "")
    assert as_user(client, None, "GET", "/v1/memories").status_code == 404
    assert as_user(client, "alice", "GET", "/v1/memories").status_code == 200


def test_auth_off_is_a_local_principal_only_in_legacy_compat(world, monkeypatch) -> None:
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", "")
    monkeypatch.setattr(settings, "default_user_id", "local-owner")
    world.member("local", "A", "human", subject="auth-disabled:local-owner")
    world.grant("local", "A", ("list",), pool(world, "A", "private", "local"))
    world.publish()
    client = make_client(world)
    assert as_user(client, None, "GET", "/v1/memories").status_code == 200


# ── fail closed, observe, off ────────────────────────────────────────────


def test_authority_outage_fails_closed_without_widening(world) -> None:
    client = make_client(world)
    assert as_user(client, "alice", "GET", "/v1/memories").status_code == 200
    world.authority.set_available(False)
    down = as_user(client, "alice", "GET", "/v1/memories")
    assert down.status_code == 503
    assert down.json()["detail"]["error"]["code"] == "unavailable_dependency"
    # Exempt routes are untouched; nothing falls back to a cached allow.
    assert as_user(client, "alice", "GET", "/v1/categories").status_code == 200
    world.authority.set_available(True)
    assert as_user(client, "alice", "GET", "/v1/memories").status_code == 200


def test_unconfigured_authority_denies_and_exempt_routes_pass(world) -> None:
    client = make_client(world)
    policy_enforcement.configure(None)
    assert as_user(client, "alice", "GET", "/v1/memories").status_code == 503
    assert as_user(client, "alice", "GET", "/v1/categories").status_code == 200


def test_unregistered_api_route_fails_closed(world) -> None:
    client = make_client(world)
    assert as_user(client, "alice", "GET", "/not/in/registry").status_code == 404


def test_denials_are_existence_safe(world) -> None:
    rows = {"present": RowInfo("private", world.ids["alice"], None, None, "private")}
    client = make_client(world, row_lookup=lambda kind, ident: rows.get(ident))
    forbidden = as_user(client, "bob", "GET", "/v1/memories/present")
    missing = as_user(client, "bob", "GET", "/v1/memories/absent")
    assert forbidden.status_code == missing.status_code == 404
    assert forbidden.json() == missing.json()
    assert forbidden.json()["detail"]["error"]["code"] == "not_found"


def test_observe_mode_logs_without_content_and_never_blocks(world, monkeypatch, caplog) -> None:
    monkeypatch.setattr(settings, "ns_policy_mode", "observe")
    client = make_client(world)
    with caplog.at_level(logging.INFO, logger="neuralscape.policy"):
        response = as_user(
            client, "svc", "POST", "/v1/memories/raw",
            json={"visibility": "shared", "text": SECRET_TEXT},
        )
    assert response.status_code == 200  # would be denied in enforce mode
    records = [r for r in caplog.records if r.getMessage() == "policy_decision"]
    assert len(records) == 1 and records[0].policy_allowed is False
    assert records[0].policy_mode == "observe"
    dumped = json.dumps({k: str(v) for r in caplog.records for k, v in vars(r).items()})
    assert SECRET_TEXT not in dumped


def test_observe_mode_never_raises_even_when_authority_is_down(world, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ns_policy_mode", "observe")
    client = make_client(world)
    world.authority.set_available(False)
    assert as_user(client, "alice", "GET", "/v1/memories").status_code == 200


def test_off_mode_touches_nothing(world, monkeypatch, caplog) -> None:
    monkeypatch.setattr(settings, "ns_policy_mode", "off")

    class Exploding(PolicyEnforcer):
        def decide(self, *args, **kwargs):
            raise AssertionError("enforcer consulted in off mode")

    client = make_client(world)
    policy_enforcement.configure(Exploding(world.authority, world.bindings))
    with caplog.at_level(logging.DEBUG, logger="neuralscape.policy"):
        assert as_user(client, "mallory", "GET", "/v1/memories").status_code == 200
        assert as_user(client, None, "GET", "/not/in/registry").status_code == 200
    assert not [r for r in caplog.records if r.name == "neuralscape.policy"]


def test_registered_exempt_operations_are_never_evaluated() -> None:
    assert REGISTRY["rest:GET /v1/categories"].status == "exempt"
    error = policy_enforcement.evaluate_request(
        "rest:GET /v1/categories", policy_enforcement.RequestContext(user_id=None)
    )
    assert error is None


def _scoped_context(scope):
    from policy_enforcement import RequestContext

    return RequestContext(user_id="u", subject_id="s", credential_id="c", credential_scope=scope)


@pytest.mark.parametrize("condition, scope, ledger, required", [
    ("credential=scoped", ("create",), True, True),
    ("credential=scoped", None, True, False),          # a sign-in credential
    ("credential=scoped", ("create",), False, False),  # no ledger, no fence
    ("credential=scope", ("create",), True, True),     # an unknown value fails closed
    ("credential=", None, False, True),
    ("credentials=scoped", None, False, True),          # an unknown condition fails closed
    ("nonsense", None, False, True),
])
def test_the_credential_condition_and_unknown_conditions_fail_closed(monkeypatch, condition, scope, ledger, required) -> None:
    from config import settings
    from policy_enforcement import PolicyEnforcer
    from policy_registry import _e

    monkeypatch.setattr(settings, "ledger_enabled", ledger)
    entry = _e("create", "mapped", "pool", "body.visibility", extra_actions=(("background_derive", condition),))
    actions = PolicyEnforcer(None, None)._required_actions(entry, _scoped_context(scope), None)  # type: ignore[arg-type]
    assert ("background_derive" in actions) is required
