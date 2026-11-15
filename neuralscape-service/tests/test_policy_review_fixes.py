"""Review findings on slice 02: the shared key under policy, the administrator operations of a single-user
install, scoped-credential gating, owner-only database files and refresh lifetimes.  App-level, SQLite."""

from __future__ import annotations

import os
import stat
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import credentials
import main
import policy_enforcement
import policy_migration as pm
from authority_store_sql import SqliteAuthorityStore
from config import settings
from contracts_policy import PolicyStatement
from credentials import CredentialService
from policy_enforcement import PolicyEnforcer, _handle
from tokens import issue_user_token

T = "tenant-a"
SECRET = "review-fixes-secret"
KEY = "the-shared-key"
SOLO = "solo-admin"


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "default_user_id", SOLO)
    monkeypatch.setattr(settings, "dictator_user_ids", "")
    store = SqliteAuthorityStore(str(tmp_path / "review.db"))
    pools = [pm.PoolObservation(u, "private", None, None) for u in (SOLO, "bob")]
    report = pm.apply_plan(
        store, pm.collect_inventory(aliases=({}, {}), pools=pools, extra_user_ids=(SOLO, "bob")), T,
        include_legacy_key=True,
    )
    solo = report.subject_ids[SOLO]
    # the bound shared-key identity may hold credential management; the key itself must still not
    store.grant_statements(T, [PolicyStatement(
        id="solo-manage", effect="allow", subject_id=solo, credential_id=None,
        action="manage_credentials", resource=_handle("credential", solo, T))])
    monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
    monkeypatch.setattr(settings, "ns_auth_profile", "legacy_compat")
    monkeypatch.setattr(settings, "credentials_enabled", True)
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", KEY)
    monkeypatch.setattr(settings, "neuralscape_api_key_user_id", SOLO)
    monkeypatch.setattr(settings, "neuralscape_legacy_key_trust_body_user_id", False)
    monkeypatch.setattr(settings, "default_user_id", SOLO)
    monkeypatch.setattr(settings, "dictator_user_ids", "")
    credentials.configure(CredentialService(store, refresh_grace=0))
    policy_enforcement.configure(PolicyEnforcer(store, store))
    service = MagicMock()
    service.list_memories.return_value = []
    monkeypatch.setattr(main, "_service", service)
    monkeypatch.setattr(main, "_task_manager", MagicMock(enqueue_raw=AsyncMock(return_value="t")))
    client = TestClient(main.app, raise_server_exceptions=False)

    class W:
        pass

    w = W()
    w.client, w.store, w.service = client, store, service
    w.tokens = {u: issue_user_token(u, SECRET, 3600) for u in (SOLO, "bob")}
    w.call = lambda bearer, method, path, **kw: client.request(  # noqa: E731
        method, path, headers={"Authorization": f"Bearer {bearer}"}, **kw
    )
    yield w
    credentials.configure(None)
    policy_enforcement.configure(None)
    store.close()


def test_the_shared_key_is_refused_under_the_enforced_profile(world, monkeypatch) -> None:
    w = world
    assert w.call(KEY, "GET", "/v1/memories").status_code == 200        # legacy_compat: as before
    monkeypatch.setattr(settings, "ns_auth_profile", "enforced")
    refused = w.call(KEY, "GET", "/v1/memories")
    assert refused.status_code == 404 and refused.json()["detail"]["error"]["code"] == "not_found"
    assert w.call(w.tokens[SOLO], "GET", "/v1/memories").status_code == 200  # a token user is not affected


def test_the_shared_key_cannot_mint_or_manage_credentials(world) -> None:
    w = world
    for method, path, body in (
        ("POST", "/v1/credentials", {"scope": ["list"]}),
        ("POST", "/v1/service-accounts", {"label": "x"}),
    ):
        response = w.call(KEY, method, path, json=body)
        assert response.status_code in (403, 404), (path, response.text)
    # the same bound user, as themself with a token, may
    assert w.call(w.tokens[SOLO], "POST", "/v1/credentials", json={"scope": ["list"]}).status_code == 200


def test_the_shared_key_is_still_the_asserted_user_not_an_administrator(world, monkeypatch) -> None:
    w = world
    monkeypatch.setattr("knowledge.router.set_project_config", MagicMock())
    response = w.call(KEY, "PUT", "/v1/projects/p1/knowledge-config", json={"project_id": "p1"})
    assert response.status_code in (403, 404)                           # never the administrator's grant


def test_a_single_user_install_keeps_its_administrator_under_enforce(world, monkeypatch) -> None:
    """Owner decision: with DICTATOR_USER_IDS empty the default user is the administrator."""

    w = world
    saved = MagicMock()
    monkeypatch.setattr("knowledge.router.set_project_config", saved)
    ok = w.call(w.tokens[SOLO], "PUT", "/v1/projects/p1/knowledge-config", json={"project_id": "p1"})
    assert ok.status_code == 200, ok.text
    assert saved.call_count == 1
    no = w.call(w.tokens["bob"], "PUT", "/v1/projects/p1/knowledge-config", json={"project_id": "p1"})
    assert no.status_code == 404 and no.json()["detail"]["error"]["code"] == "not_found"
    assert saved.call_count == 1                                         # policy refused before the handler
    # the MCP counterpart is judged the same way
    from auth import current_user_id

    for user, allowed in ((SOLO, True), ("bob", False)):
        token = current_user_id.set(user)
        try:
            denial = policy_enforcement.enforce_mcp_call("set_project_knowledge_config", {"project_id": "p1"})
        finally:
            current_user_id.reset(token)
        assert (denial is None) is allowed, user


def test_a_listed_administrator_replaces_the_default_user(world, monkeypatch) -> None:
    assert pm.admin_user_ids(["carol"], default_user="solo") == frozenset({"carol"})
    assert pm.admin_user_ids([], default_user="solo") == frozenset({"solo"})
    assert pm.admin_user_ids([], default_user="") == frozenset()


def test_credential_routes_need_both_flags(world, monkeypatch) -> None:
    w = world
    monkeypatch.setattr(settings, "ns_policy_mode", "off")
    assert w.call(w.tokens[SOLO], "POST", "/v1/credentials", json={"scope": ["list"]}).status_code == 404
    monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
    monkeypatch.setattr(settings, "credentials_enabled", False)
    assert w.call(w.tokens[SOLO], "POST", "/v1/credentials", json={"scope": ["list"]}).status_code == 404
    # unauthenticated refresh is a 404 (not a 401) while the feature is off
    assert w.client.post("/v1/credentials/refresh", json={"refresh_token": "nsr_x.y"}).status_code == 404


def test_a_refreshed_credential_keeps_the_lifetime_it_was_issued_with(world) -> None:
    w = world
    issued = w.call(w.tokens[SOLO], "POST", "/v1/credentials",
                    json={"scope": ["list"], "access_ttl_seconds": 60, "refresh_ttl_seconds": 600}).json()
    again = w.client.post("/v1/credentials/refresh", json={"refresh_token": issued["refresh_token"]}).json()
    assert again["access_expires_at"] - issued["access_expires_at"] < 5          # still ~60 s, not 15 min
    assert again["refresh_expires_at"] - issued["refresh_expires_at"] < 5        # still ~10 min, not 30 days


def test_a_database_that_cannot_be_made_owner_only_is_refused(tmp_path, monkeypatch) -> None:
    path = tmp_path / "loose.db"
    path.write_bytes(b"")
    os.chmod(path, 0o644)

    def refuse(*args, **kwargs):
        raise PermissionError("not the owner")

    monkeypatch.setattr(os, "fchmod", refuse)
    with pytest.raises(OSError, match="owner-only"):
        SqliteAuthorityStore(str(path))
    monkeypatch.undo()
    SqliteAuthorityStore(str(path)).close()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
