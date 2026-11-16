"""End to end in enforce mode on the real app: scoped credentials, refresh
replay, revocation, offboarding and service accounts, with the existing token
format untouched."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import credentials
import main
import policy_enforcement
from authority_store_sql import SqliteAuthorityStore
from config import settings
from contracts_policy import PolicyStatement
from credentials import CredentialService
from identity_binding import bind_identity
from policy_enforcement import PolicyEnforcer, pool_handle
from tokens import issue_user_token

T = "tenant-a"
SECRET = "e2e-credentials-secret"


def stmt(counter, subject, action, handle):
    return PolicyStatement(id=f"s{next(counter)}", effect="allow", subject_id=subject,
                           credential_id=None, action=action, resource=handle)


class World:
    def __init__(self, monkeypatch, tmp_path) -> None:
        import itertools

        self.counter = itertools.count(1)
        self.store = SqliteAuthorityStore(str(tmp_path / "e2e.db"))
        self.service = CredentialService(self.store, refresh_grace=0)
        self.subjects = {}
        for user in ("alice", "bob"):
            sid = bind_identity(self.store, "local", user).subject_id
            self.store.add_subject(sid, T, "human", f"cred-{user}")
            self.subjects[user] = sid
        self.extra: list[PolicyStatement] = []
        self.publish()
        monkeypatch.setattr(settings, "credentials_enabled", True)
        monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
        monkeypatch.setattr(settings, "ns_auth_profile", "legacy_compat")
        monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
        monkeypatch.setattr(settings, "neuralscape_api_key", "")
        credentials.configure(self.service)
        policy_enforcement.configure(PolicyEnforcer(self.store, self.store))
        svc = MagicMock()
        svc.search.return_value = []
        svc.store_raw.return_value = []
        monkeypatch.setattr(main, "_service", svc)
        tasks = MagicMock()
        tasks.enqueue_raw = AsyncMock(return_value="task-1")
        monkeypatch.setattr(main, "_task_manager", tasks)
        self.svc, self.tasks = svc, tasks
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.hmac = {u: issue_user_token(u, SECRET, 3600) for u in self.subjects}

    def publish(self) -> None:
        a, b = self.subjects["alice"], self.subjects["bob"]
        own = pool_handle(T, "private", a)
        statements = [
            *[stmt(self.counter, a, act, own) for act in ("read", "search", "list", "create")],
            stmt(self.counter, a, "manage_credentials", policy_enforcement._handle("credential", a, T)),
            stmt(self.counter, a, "manage_membership", policy_enforcement._handle("service_account", T, T)),
            stmt(self.counter, b, "read", pool_handle(T, "private", b)),
            *self.extra,
        ]
        self.store.publish_policy(T, statements)

    def grant_for_account(self, account: str) -> None:
        a = self.subjects["alice"]
        self.extra = [
            stmt(self.counter, account, "list", pool_handle(T, "private", account)),
            stmt(self.counter, a, "manage_credentials", policy_enforcement._handle("service_account", account, T)),
            stmt(self.counter, a, "manage_membership", policy_enforcement._handle("service_account", account, T)),
        ]
        self.publish()

    def call(self, method, path, bearer=None, **kw):
        headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
        return self.client.request(method, path, headers=headers, **kw)

    def close(self) -> None:
        credentials.configure(None)
        policy_enforcement.configure(None)
        self.store.close()


@pytest.fixture
def w(monkeypatch, tmp_path):
    world = World(monkeypatch, tmp_path)
    yield world
    world.close()


def issue(w, scope, bearer=None, **extra):
    return w.call("POST", "/v1/credentials", bearer or w.hmac["alice"], json={"scope": scope, **extra})


def test_existing_tokens_still_authenticate_and_issue_a_scoped_credential(w) -> None:
    assert w.call("POST", "/v1/search", w.hmac["alice"], json={"query": "q"}).status_code == 200
    response = issue(w, ["search", "read"])
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "Bearer" and body["scope"] == ["read", "search"]
    assert body["access_token"].startswith("nsc_") and body["refresh_token"].startswith("nsr_")


def test_a_scope_beyond_the_subjects_grants_is_refused(w) -> None:
    assert issue(w, ["delete"]).status_code == 400
    assert issue(w, ["search", "export"]).status_code == 400
    assert issue(w, []).status_code == 422
    assert issue(w, ["search"], access_ttl_seconds=10**9).status_code == 404  # ttl out of range


def test_a_scoped_credential_can_only_do_what_its_scope_allows(w) -> None:
    token = issue(w, ["search"]).json()["access_token"]
    assert w.call("POST", "/v1/search", token, json={"query": "q"}).status_code == 200
    # alice may create and list, but this credential's scope does not include them
    for method, path, kw in (
        ("POST", "/v1/memories/raw", {"json": {"content": "x", "category": "preference", "visibility": "private"}}),
        ("GET", "/v1/memories", {}),
    ):
        denied = w.call(method, path, token, **kw)
        assert denied.status_code == 404 and denied.json()["detail"]["error"]["code"] == "not_found"
    assert w.tasks.enqueue_raw.await_count == 0
    # the same subject's ordinary token is not narrowed
    assert w.call("GET", "/v1/memories", w.hmac["alice"]).status_code == 200


def test_a_scoped_credential_cannot_mint_a_broader_one(w) -> None:
    token = issue(w, ["search", "manage_credentials"]).json()["access_token"]
    assert issue(w, ["search"], bearer=token).status_code == 200
    assert issue(w, ["search", "read"], bearer=token).status_code == 400   # read is outside its scope


def test_refresh_rotates_and_replay_revokes_the_family(w) -> None:
    first = issue(w, ["search"]).json()
    second = w.call("POST", "/v1/credentials/refresh", json={"refresh_token": first["refresh_token"]})
    assert second.status_code == 200
    second = second.json()
    assert w.call("POST", "/v1/search", first["access_token"], json={"query": "q"}).status_code == 401
    assert w.call("POST", "/v1/search", second["access_token"], json={"query": "q"}).status_code == 200
    # replay of the consumed token: indistinguishable 401, and the successor dies
    replay = w.call("POST", "/v1/credentials/refresh", json={"refresh_token": first["refresh_token"]})
    assert replay.status_code == 401
    unknown = w.call("POST", "/v1/credentials/refresh", json={"refresh_token": "nsr_cred_x.y"})
    assert unknown.status_code == 401 and unknown.json() == replay.json()
    assert w.call("POST", "/v1/search", second["access_token"], json={"query": "q"}).status_code == 401
    assert w.call("POST", "/v1/credentials/refresh", json={"refresh_token": second["refresh_token"]}).status_code == 401


def test_revocation_takes_effect_immediately_and_is_owner_only(w) -> None:
    issued = issue(w, ["search"]).json()
    other = w.call("POST", "/v1/credentials/revoke", w.hmac["bob"], json={"credential_id": issued["credential_id"]})
    missing = w.call("POST", "/v1/credentials/revoke", w.hmac["bob"], json={"credential_id": "cred_none"})
    assert other.status_code == missing.status_code
    # bob has no manage_credentials grant, so both are policy denials with one shape
    assert other.json() == missing.json()
    assert w.call("POST", "/v1/search", issued["access_token"], json={"query": "q"}).status_code == 200
    revoked = w.call("POST", "/v1/credentials/revoke", w.hmac["alice"], json={"credential_id": issued["credential_id"]})
    assert revoked.status_code == 200 and revoked.json() == {"revoked": True}
    assert w.call("POST", "/v1/search", issued["access_token"], json={"query": "q"}).status_code == 401


def test_offboarding_the_subject_stops_every_credential(w) -> None:
    issued = issue(w, ["search"]).json()
    w.store.offboard_subject(w.subjects["alice"])
    assert w.call("POST", "/v1/search", issued["access_token"], json={"query": "q"}).status_code == 401
    assert w.call("POST", "/v1/credentials/refresh", json={"refresh_token": issued["refresh_token"]}).status_code == 401


def test_service_account_lifecycle_with_rotation_and_disable(w) -> None:
    created = w.call("POST", "/v1/service-accounts", w.hmac["alice"], json={"label": "ci"})
    assert created.status_code == 200
    account = created.json()["subject_id"]
    assert created.json()["controller_subject_id"] == w.subjects["alice"]
    w.grant_for_account(account)
    # bob (not the controller, no grant) cannot issue for it
    assert w.call("POST", f"/v1/service-accounts/{account}/credentials", w.hmac["bob"], json={"scope": ["list"]}).status_code == 404
    first = w.call("POST", f"/v1/service-accounts/{account}/credentials", w.hmac["alice"], json={"scope": ["list"]})
    assert first.status_code == 200
    first = first.json()
    assert w.call("GET", "/v1/memories", first["access_token"]).status_code == 200      # least authority: its own grants
    assert w.call("POST", "/v1/search", first["access_token"], json={"query": "q"}).status_code == 404  # no search grant
    second = w.call("POST", f"/v1/service-accounts/{account}/credentials", w.hmac["alice"], json={"scope": ["list"]})
    assert second.status_code == 200                                                      # two active: no downtime
    second = second.json()
    assert w.call("GET", "/v1/memories", first["access_token"]).status_code == 200
    assert w.call("GET", "/v1/memories", second["access_token"]).status_code == 200
    third = w.call("POST", f"/v1/service-accounts/{account}/credentials", w.hmac["alice"], json={"scope": ["list"]})
    assert third.status_code == 409
    w.call("POST", "/v1/credentials/revoke", w.hmac["alice"], json={"credential_id": first["credential_id"]})
    assert w.call("GET", "/v1/memories", first["access_token"]).status_code == 401
    assert w.call("GET", "/v1/memories", second["access_token"]).status_code == 200
    # the service account cannot administer anything, including itself
    assert w.call("POST", "/v1/service-accounts", second["access_token"], json={"label": "nested"}).status_code == 404
    assert issue(w, ["list"], bearer=second["access_token"]).status_code == 404
    disabled = w.call("POST", f"/v1/service-accounts/{account}/disable", w.hmac["alice"])
    assert disabled.status_code == 200
    assert w.call("GET", "/v1/memories", second["access_token"]).status_code == 401
    assert w.call("POST", f"/v1/service-accounts/{account}/credentials", w.hmac["alice"], json={"scope": ["list"]}).status_code == 404


def test_a_service_account_cannot_be_created_by_a_non_human(w) -> None:
    account = w.call("POST", "/v1/service-accounts", w.hmac["alice"], json={"label": "ci"}).json()["subject_id"]
    w.grant_for_account(account)
    token = w.call("POST", f"/v1/service-accounts/{account}/credentials", w.hmac["alice"], json={"scope": ["list"]}).json()["access_token"]
    # even with an explicit grant the controller must be a human
    w.extra.append(stmt(w.counter, account, "manage_membership", policy_enforcement._handle("service_account", T, T)))
    w.publish()
    assert w.call("POST", "/v1/service-accounts", token, json={"label": "nested"}).status_code in {404}


def test_the_feature_is_off_by_default_and_off_without_enforcement(w, monkeypatch) -> None:
    token = issue(w, ["search"]).json()["access_token"]
    monkeypatch.setattr(settings, "ns_policy_mode", "observe")
    assert w.call("POST", "/v1/search", token, json={"query": "q"}).status_code == 401   # scope would be unenforced
    assert w.call("POST", "/v1/search", w.hmac["alice"], json={"query": "q"}).status_code == 200
    monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
    monkeypatch.setattr(settings, "credentials_enabled", False)
    assert w.call("POST", "/v1/credentials", w.hmac["alice"], json={"scope": ["search"]}).status_code == 404
    assert w.call("POST", "/v1/search", token, json={"query": "q"}).status_code == 401
    assert w.call("POST", "/v1/search", w.hmac["alice"], json={"query": "q"}).status_code == 200


def test_an_unreadable_store_fails_closed_for_scoped_tokens(w) -> None:
    token = issue(w, ["search"]).json()["access_token"]
    w.store._rconn.close(); w.store._conn.close()
    down = w.call("POST", "/v1/search", token, json={"query": "q"})
    assert down.status_code == 503
    assert w.call("GET", "/health/live").status_code == 200


def test_a_scoped_credential_cannot_mint_a_longer_lived_one_or_a_refresh_token(w) -> None:
    parent = issue(w, ["search", "read", "manage_credentials"], access_ttl_seconds=60, with_refresh=False).json()
    token = parent["access_token"]
    # a 60-second token must not be able to ask for a 90-day refresh token
    refused = issue(w, ["search"], bearer=token, with_refresh=True, refresh_ttl_seconds=7776000)
    assert refused.status_code == 400
    child = issue(w, ["search"], bearer=token, access_ttl_seconds=900)
    assert child.status_code == 200
    child = child.json()
    assert child["refresh_token"] is None                       # no refresh unless explicitly permitted
    assert child["access_expires_at"] <= parent["access_expires_at"]
    # revoking the minting credential revokes what it minted
    assert w.call("POST", "/v1/search", child["access_token"], json={"query": "q"}).status_code == 200
    revoked = w.call("POST", "/v1/credentials/revoke", w.hmac["alice"], json={"credential_id": parent["credential_id"]})
    assert revoked.json() == {"revoked": True}
    assert w.call("POST", "/v1/search", child["access_token"], json={"query": "q"}).status_code == 401


def test_an_ordinary_token_still_gets_a_refresh_token_by_default(w) -> None:
    assert issue(w, ["search"]).json()["refresh_token"].startswith("nsr_")


def test_a_duplicate_refresh_within_the_grace_is_idempotent_over_http(w) -> None:
    w.service._refresh_grace = 30.0
    first = issue(w, ["search"]).json()
    one = w.call("POST", "/v1/credentials/refresh", json={"refresh_token": first["refresh_token"]})
    two = w.call("POST", "/v1/credentials/refresh", json={"refresh_token": first["refresh_token"]})
    assert one.status_code == two.status_code == 200
    assert one.json() == two.json()
    assert w.call("POST", "/v1/search", one.json()["access_token"], json={"query": "q"}).status_code == 200


def test_offboarding_a_controller_stops_their_service_accounts_over_http(w) -> None:
    account = w.call("POST", "/v1/service-accounts", w.hmac["alice"], json={"label": "ci"}).json()["subject_id"]
    w.grant_for_account(account)
    token = w.call("POST", f"/v1/service-accounts/{account}/credentials", w.hmac["alice"],
                   json={"scope": ["list"]}).json()["access_token"]
    assert w.call("GET", "/v1/memories", token).status_code == 200
    w.store.offboard_subject(w.subjects["alice"])
    assert w.call("GET", "/v1/memories", token).status_code == 401


# --- durable work: a scoped credential needs background_derive at the request, as the fence demands later ---


def _raw(w, token, visibility="private"):
    return w.call("POST", "/v1/memories/raw", token,
                  json={"content": "x", "category": "preference", "visibility": visibility})


def _grant_derivation(w, *pools):
    a = w.subjects["alice"]
    w.extra = [stmt(w.counter, a, action, handle) for handle in pools for action in ("create", "background_derive")]
    w.publish()


def test_a_scoped_credential_without_background_derive_is_refused_at_the_request(w, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ledger_enabled", True)
    token = issue(w, ["create"]).json()["access_token"]
    denied = _raw(w, token)
    assert denied.status_code == 404 and denied.json()["detail"]["error"]["code"] == "not_found"
    assert w.tasks.enqueue_raw.await_count == 0  # nothing accepted: no intent to be fenced later


def test_a_scoped_credential_with_background_derive_is_admitted(w, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ledger_enabled", True)
    _grant_derivation(w, pool_handle(T, "private", w.subjects["alice"]))
    token = issue(w, ["create", "background_derive"]).json()["access_token"]
    assert _raw(w, token).status_code == 202
    assert w.tasks.enqueue_raw.await_count == 1


def test_background_derive_is_evaluated_on_the_target_pool_at_the_request(w, monkeypatch) -> None:
    # alice may create in the shared pool, but derive only in her own: a scoped shared write is refused now,
    # not accepted and then fenced
    monkeypatch.setattr(settings, "ledger_enabled", True)
    a = w.subjects["alice"]
    _grant_derivation(w, pool_handle(T, "private", a))
    w.extra.append(stmt(w.counter, a, "create", pool_handle(T, "shared", None)))
    w.extra.append(stmt(w.counter, a, "share", pool_handle(T, "shared", None)))
    w.publish()
    token = issue(w, ["create", "background_derive", "share"]).json()["access_token"]
    assert _raw(w, token, visibility="shared").status_code == 404
    assert w.tasks.enqueue_raw.await_count == 0


def test_sign_in_credentials_and_ledger_off_deployments_are_unaffected(w, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ledger_enabled", True)
    assert _raw(w, w.hmac["alice"]).status_code == 202  # alice holds no background_derive grant
    monkeypatch.setattr(settings, "ledger_enabled", False)
    token = issue(w, ["create"]).json()["access_token"]
    assert _raw(w, token).status_code == 202  # no ledger, no fence: create alone is enough
    assert w.tasks.enqueue_raw.await_count == 2
