"""Artifact reads and downloads require ``read`` on the owning pool."""

from __future__ import annotations

import itertools

import pytest
from fastapi.testclient import TestClient

import main
import policy_enforcement
from authority_store_sql import SqliteAuthorityStore
from config import settings
from contracts_policy import PolicyStatement
from identity_binding import bind_identity
from policy_enforcement import PolicyEnforcer, pool_handle
from policy_migration import artifact_row_lookup
from tokens import issue_user_token

SECRET = "artifact-secret"
GLOBAL_ID, PROJECT_ID, BOB_ID = "a" * 16, "b" * 16, "c" * 16


def stmt(counter, subject, action, handle):
    return PolicyStatement(id=f"s{next(counter)}", effect="allow", subject_id=subject,
                           credential_id=None, action=action, resource=handle)


@pytest.fixture
def world(tmp_path, monkeypatch):
    counter = itertools.count(1)
    store = SqliteAuthorityStore(str(tmp_path / "a.db"))
    ids = {}
    for user, tenant in (("alice", "tenant-a"), ("bob", "tenant-a"), ("carol", "tenant-b")):
        ids[user] = bind_identity(store, "local", user).subject_id
        store.add_subject(ids[user], tenant, "human", f"c-{user}")
    store.publish_policy("tenant-a", [
        stmt(counter, ids["alice"], "read", pool_handle("tenant-a", "private", ids["alice"], None)),
        stmt(counter, ids["alice"], "read", pool_handle("tenant-a", "private", ids["alice"], "p1")),
        stmt(counter, ids["bob"], "read", pool_handle("tenant-a", "private", ids["bob"], None)),
    ])
    store.publish_policy("tenant-b", [
        stmt(counter, ids["carol"], "read", pool_handle("tenant-b", "private", ids["carol"], None)),
    ])
    root = tmp_path / "ingest"
    for user, project, file_id, text in (("alice", "_global", GLOBAL_ID, "alice-global"),
                                         ("alice", "p1", PROJECT_ID, "alice-project"),
                                         ("bob", "_global", BOB_ID, "bob-global")):
        directory = root / user / project / "notes"
        directory.mkdir(parents=True)
        (directory / f"{file_id}.txt").write_text(text)
    (root / "zed" / "_global" / "notes").mkdir(parents=True)          # an owner with no binding
    (root / "zed" / "_global" / "notes" / f"{'d' * 16}.txt").write_text("orphan")
    monkeypatch.setattr(settings, "ingest_storage_dir", str(root))
    monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
    monkeypatch.setattr(settings, "ns_auth_profile", "legacy_compat")
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    policy_enforcement.configure(PolicyEnforcer(store, store, row_lookup=artifact_row_lookup(settings, store)))
    tokens = {u: issue_user_token(u, SECRET, 3600) for u in ids}
    client = TestClient(main.app, raise_server_exceptions=False)
    yield client, tokens, ids
    policy_enforcement.configure(None)
    store.close()


def get(client, tokens, user, file_id):
    return client.get(f"/v1/ingest/artifacts/{file_id}", headers={"Authorization": f"Bearer {tokens[user]}"})


def test_the_owner_downloads_global_and_project_artifacts(world) -> None:
    client, tokens, _ = world
    first = get(client, tokens, "alice", GLOBAL_ID)
    assert (first.status_code, first.text) == (200, "alice-global")
    second = get(client, tokens, "alice", PROJECT_ID)
    assert (second.status_code, second.text) == (200, "alice-project")
    assert get(client, tokens, "bob", BOB_ID).text == "bob-global"


def test_a_same_tenant_stranger_and_another_tenant_are_denied(world) -> None:
    client, tokens, _ = world
    for user in ("bob", "carol"):
        for file_id in (GLOBAL_ID, PROJECT_ID):
            denied = get(client, tokens, user, file_id)
            assert denied.status_code == 404 and "alice-" not in denied.text
            assert denied.json()["detail"]["error"]["code"] == "not_found"


def test_unknown_malformed_and_orphaned_ids_look_exactly_like_a_forbidden_one(world) -> None:
    client, tokens, _ = world
    forbidden = get(client, tokens, "bob", GLOBAL_ID).json()
    for file_id in ("e" * 16, "short", "..__..__etc__passwd", "d" * 16, "A" * 16):
        response = get(client, tokens, "alice", file_id)
        assert response.status_code == 404 and response.json() == forbidden, file_id


def test_a_project_artifact_needs_a_grant_on_that_project_pool(world) -> None:
    client, tokens, ids = world
    # bob's own pool grant does not extend to a project pool he does not hold
    assert get(client, tokens, "bob", PROJECT_ID).status_code == 404


def test_revoking_the_pool_grant_stops_downloads_at_once(world, tmp_path) -> None:
    client, tokens, ids = world
    assert get(client, tokens, "alice", GLOBAL_ID).status_code == 200
    store = policy_enforcement._enforcer._authority
    store.publish_policy("tenant-a", [])
    assert get(client, tokens, "alice", GLOBAL_ID).status_code == 404


def test_exemplar_downloads_fail_closed_until_a_memory_backed_lookup_exists(world, monkeypatch) -> None:
    client, tokens, _ = world
    monkeypatch.setattr(settings, "exemplar_store_enabled", True)
    response = client.get(f"/v1/ingest/exemplars/{GLOBAL_ID}", headers={"Authorization": f"Bearer {tokens['bob']}"})
    assert response.status_code == 404 and response.json()["detail"]["error"]["code"] == "not_found"


def test_off_mode_leaves_artifact_downloads_to_the_existing_owner_scoping(world, monkeypatch) -> None:
    client, tokens, _ = world
    monkeypatch.setattr(settings, "ns_policy_mode", "off")
    assert get(client, tokens, "alice", GLOBAL_ID).status_code == 200
    other = get(client, tokens, "bob", GLOBAL_ID)
    assert other.status_code == 404 and "detail" in other.json() and "error" not in other.json()["detail"]
