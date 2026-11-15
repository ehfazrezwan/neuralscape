"""Legacy shared API key acts as ONE identity (owner decision, option A).

Before: a caller presenting ``NEURALSCAPE_API_KEY`` had no ``request.state.user_id``,
so every route trusted the body/query ``user_id`` and MCP trusted
``arguments["user_id"]``: any key holder could act as any user.

Now: the key is bound to ``NEURALSCAPE_API_KEY_USER_ID`` (else ``DEFAULT_USER_ID``)
exactly like a per-user token; a body user naming someone else is a 400, MCP
ignores ``arguments["user_id"]``. ``NEURALSCAPE_LEGACY_KEY_TRUST_BODY_USER_ID=true``
restores the old behaviour. Also covers token revocation
(``NEURALSCAPE_REVOKED_TOKENS``) and the ``issue_user_token.py --no-expiry`` rule.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import main
from auth import BearerAuthMiddleware
from config import Settings, settings
from main import app
from tokens import issue_user_token, token_fingerprint, verify_user_token

KEY = "legacy-shared-key-abc"
SECRET = "revocation-test-secret"
SERVICE_DIR = Path(__file__).resolve().parent.parent
AUTH = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture(autouse=True)
def mock_backends():
    """Same lazy-init stubs as tests/test_auth.py: no real mem0/Graphiti/Redis."""
    saved = (main._memory, main._graphiti, main._bridge, main._async_memory, main._task_manager)
    main._memory = MagicMock(name="Memory")
    main._graphiti = MagicMock(name="Graphiti")
    main._bridge = MagicMock(name="AsyncBridge")
    main._async_memory = None
    tm = MagicMock(name="TaskManager")
    tm.connect = AsyncMock()
    tm.close = AsyncMock()
    tm.pool = MagicMock()
    tm.pool.ping = AsyncMock()
    tm.enqueue_store = AsyncMock(return_value="task-1")
    main._task_manager = tm
    yield tm
    main._memory, main._graphiti, main._bridge, main._async_memory, main._task_manager = saved


@pytest.fixture
def service():
    svc = MagicMock(name="MemoryService")
    svc.list_memories.return_value = []
    saved = main._service
    main._service = svc
    yield svc
    main._service = saved


@pytest.fixture
def shared_key(monkeypatch):
    """Shared key configured, bound-identity defaults, no per-user secret."""
    monkeypatch.setattr(settings, "neuralscape_api_key", KEY)
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", "")
    monkeypatch.setattr(settings, "neuralscape_api_key_user_id", "")
    monkeypatch.setattr(settings, "neuralscape_legacy_key_trust_body_user_id", False)
    monkeypatch.setattr(settings, "default_user_id", "the-default-user")


def _list(client, user_id=None, headers=AUTH):
    params = {"user_id": user_id} if user_id else {}
    return client.get("/v1/memories", params=params, headers=headers)


# ──────────────────────────────────────────────
# (a) NEURALSCAPE_API_KEY_USER_ID=alice
# ──────────────────────────────────────────────


class TestBoundToConfiguredUser:
    @pytest.fixture(autouse=True)
    def alice(self, shared_key, monkeypatch):
        monkeypatch.setattr(settings, "neuralscape_api_key_user_id", "alice")

    def test_body_user_matching_bound_identity_ok(self, service):
        resp = _list(TestClient(app, raise_server_exceptions=False), "alice")
        assert resp.status_code == 200
        assert service.list_memories.call_args.kwargs["user_id"] == "alice"

    def test_body_user_naming_someone_else_is_400(self, service):
        resp = _list(TestClient(app, raise_server_exceptions=False), "bob")
        assert resp.status_code == 400
        assert "does not match" in resp.json()["detail"]
        service.list_memories.assert_not_called()

    def test_no_body_user_acts_as_bound_identity(self, service):
        resp = _list(TestClient(app, raise_server_exceptions=False))
        assert resp.status_code == 200
        assert service.list_memories.call_args.kwargs["user_id"] == "alice"

    def test_deprecation_header_kept(self, service):
        resp = _list(TestClient(app, raise_server_exceptions=False), "alice")
        assert "deprecated" in resp.headers["X-Neuralscape-Deprecation"]

    def test_async_write_for_other_user_is_400_and_not_enqueued(self, mock_backends):
        client = TestClient(app, raise_server_exceptions=False)
        body = {"messages": [{"role": "user", "content": "hi"}], "user_id": "bob"}
        resp = client.post("/memories/async", json=body, headers=AUTH)
        assert resp.status_code == 400
        mock_backends.enqueue_store.assert_not_called()

    def test_async_write_without_user_is_enqueued_as_bound_identity(self, mock_backends):
        client = TestClient(app, raise_server_exceptions=False)
        body = {"messages": [{"role": "user", "content": "hi"}]}
        resp = client.post("/memories/async", json=body, headers=AUTH)
        assert resp.status_code == 200
        assert mock_backends.enqueue_store.call_args.kwargs["user_id"] == "alice"

    def test_legacy_root_route_with_other_user_is_400(self, service):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/memories", params={"user_id": "bob"}, headers=AUTH)
        assert resp.status_code == 400

    def test_batch_item_naming_another_user_is_400(self, mock_backends):
        client = TestClient(app, raise_server_exceptions=False)
        body = {"memories": [
            {"content": "x", "user_id": "alice", "category": "preference"},
            {"content": "y", "user_id": "bob", "category": "preference"},
        ]}
        mock_backends.enqueue_raw_batch = AsyncMock(return_value="b1")
        resp = client.post("/v1/memories/raw/batch", json=body, headers=AUTH)
        assert resp.status_code == 400
        assert "Item 1" in resp.json()["detail"]
        mock_backends.enqueue_raw_batch.assert_not_called()

    def test_wrong_key_still_401(self):
        resp = _list(TestClient(app, raise_server_exceptions=False), "alice",
                     headers={"Authorization": "Bearer nope"})
        assert resp.status_code == 401


# ──────────────────────────────────────────────
# (b) no NEURALSCAPE_API_KEY_USER_ID: bound to DEFAULT_USER_ID
# ──────────────────────────────────────────────


class TestBoundToDefaultUser:
    def test_default_user_is_the_bound_identity(self, shared_key, service):
        client = TestClient(app, raise_server_exceptions=False)
        assert _list(client).status_code == 200
        assert service.list_memories.call_args.kwargs["user_id"] == "the-default-user"
        assert _list(client, "the-default-user").status_code == 200

    def test_other_body_user_is_400(self, shared_key, service):
        resp = _list(TestClient(app, raise_server_exceptions=False), "mallory")
        assert resp.status_code == 400
        service.list_memories.assert_not_called()

    def test_blank_configured_identity_falls_back_to_default(self, shared_key, monkeypatch):
        monkeypatch.setattr(settings, "neuralscape_api_key_user_id", "   ")
        assert settings.legacy_key_identity() == "the-default-user"


# ──────────────────────────────────────────────
# (c) opt-out: NEURALSCAPE_LEGACY_KEY_TRUST_BODY_USER_ID=true
# ──────────────────────────────────────────────


class TestOptOutTrustsBody:
    @pytest.fixture(autouse=True)
    def opt_out(self, shared_key, monkeypatch):
        monkeypatch.setattr(settings, "neuralscape_legacy_key_trust_body_user_id", True)
        # A configured binding must be ignored under the opt-out.
        monkeypatch.setattr(settings, "neuralscape_api_key_user_id", "alice")

    def test_any_body_user_trusted(self, service):
        client = TestClient(app, raise_server_exceptions=False)
        assert _list(client, "bob").status_code == 200
        assert service.list_memories.call_args.kwargs["user_id"] == "bob"

    def test_no_body_user_falls_back_to_default_user(self, service):
        assert _list(TestClient(app, raise_server_exceptions=False)).status_code == 200
        assert service.list_memories.call_args.kwargs["user_id"] == "the-default-user"

    def test_async_write_for_any_user_enqueued(self, mock_backends):
        client = TestClient(app, raise_server_exceptions=False)
        body = {"messages": [{"role": "user", "content": "hi"}], "user_id": "bob"}
        assert client.post("/memories/async", json=body, headers=AUTH).status_code == 200
        assert mock_backends.enqueue_store.call_args.kwargs["user_id"] == "bob"

    def test_startup_logs_warning_when_on(self, caplog, monkeypatch):
        s = Settings(_env_file=None, google_api_key="g", neo4j_password="p",
                     neuralscape_api_key=KEY, neuralscape_legacy_key_trust_body_user_id=True)
        with caplog.at_level(logging.WARNING):
            s.validate_required()
        assert any("LEGACY_KEY_TRUST_BODY_USER_ID" in r.message for r in caplog.records)

    def test_no_startup_warning_when_off(self, caplog):
        s = Settings(_env_file=None, google_api_key="g", neo4j_password="p", neuralscape_api_key=KEY)
        with caplog.at_level(logging.WARNING):
            s.validate_required()
        assert not any("LEGACY_KEY_TRUST_BODY_USER_ID" in r.message for r in caplog.records)


class TestSettingsValidation:
    def test_defaults_are_bound_not_trusting(self):
        s = Settings(_env_file=None)
        assert s.neuralscape_legacy_key_trust_body_user_id is False
        assert s.neuralscape_api_key_user_id == ""

    def test_env_names_map_to_settings(self, monkeypatch):
        monkeypatch.setenv("NEURALSCAPE_API_KEY_USER_ID", "carol")
        monkeypatch.setenv("NEURALSCAPE_LEGACY_KEY_TRUST_BODY_USER_ID", "true")
        monkeypatch.setenv("NEURALSCAPE_REVOKED_TOKENS", "AB, cd")
        s = Settings(_env_file=None)
        assert s.neuralscape_api_key_user_id == "carol"
        assert s.neuralscape_legacy_key_trust_body_user_id is True
        assert s.revoked_token_digests() == {"ab", "cd"}

    def test_malformed_bound_user_refuses_startup(self):
        s = Settings(_env_file=None, google_api_key="g", neo4j_password="p",
                     neuralscape_api_key=KEY, neuralscape_api_key_user_id="not valid!")
        with pytest.raises(ValueError, match="NEURALSCAPE_API_KEY_USER_ID"):
            s.validate_required()


# ──────────────────────────────────────────────
# Dots in the legacy key, with the per-user secret also configured
# ──────────────────────────────────────────────


class TestDottedLegacyKey:
    def test_dotted_key_authenticates_and_is_bound(self, shared_key, service, monkeypatch):
        monkeypatch.setattr(settings, "neuralscape_api_key", "legacy.key.with.dots")
        monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
        client = TestClient(app, raise_server_exceptions=False)
        headers = {"Authorization": "Bearer legacy.key.with.dots"}
        assert _list(client, None, headers).status_code == 200
        assert service.list_memories.call_args.kwargs["user_id"] == "the-default-user"
        assert _list(client, "bob", headers).status_code == 400

    def test_non_ascii_bearer_is_401_not_500(self, shared_key):
        client = TestClient(app, raise_server_exceptions=False)
        # Raw header bytes: a non-ASCII key made hmac.compare_digest(str, str) raise.
        resp = client.get("/v1/memories", headers={"Authorization": "Bearer caf\u00e9-key".encode("utf-8")})
        assert resp.status_code == 401


# ──────────────────────────────────────────────
# Places that branch on "is there a token identity?"
# ──────────────────────────────────────────────


class TestBoundIdentityInOwnerChecks:
    def test_task_status_is_owner_checked_for_bound_identity(self, shared_key, mock_backends):
        """/v1/memories/status passes enforce_owner=True with the bound user."""
        mock_backends.get_status = AsyncMock(return_value={"task_id": "t", "status": "queued"})
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/memories/status/t1", headers=AUTH)
        assert resp.status_code == 200
        kwargs = mock_backends.get_status.call_args.kwargs
        assert kwargs["user_id"] == "the-default-user"
        assert kwargs["enforce_owner"] is True

    def test_task_status_unchecked_under_opt_out(self, shared_key, mock_backends, monkeypatch):
        monkeypatch.setattr(settings, "neuralscape_legacy_key_trust_body_user_id", True)
        mock_backends.get_status = AsyncMock(return_value={"task_id": "t", "status": "queued"})
        TestClient(app, raise_server_exceptions=False).get("/v1/memories/status/t1", headers=AUTH)
        assert mock_backends.get_status.call_args.kwargs["enforce_owner"] is False


# ──────────────────────────────────────────────
# MCP: arguments["user_id"] is ignored in favour of the bound identity
# ──────────────────────────────────────────────


def _mcp_probe_app():
    """A tiny app behind the real middleware whose route runs a real MCP tool call."""
    import mcp_server

    probe = FastAPI()
    probe.add_middleware(BearerAuthMiddleware)

    @probe.post("/probe")
    async def run(request: Request):
        args = await request.json()
        await mcp_server.call_tool("list_memories", args)
        return {"ok": True}

    return probe


class TestMcpBoundIdentity:
    @pytest.fixture
    def mcp_service(self, monkeypatch):
        import mcp_server

        svc = MagicMock(name="MemoryService")
        svc.list_memories.return_value = []
        monkeypatch.setattr(mcp_server, "_service", svc)
        return svc

    def test_argument_user_ignored_for_bound_key(self, shared_key, mcp_service, monkeypatch):
        monkeypatch.setattr(settings, "neuralscape_api_key_user_id", "alice")
        client = TestClient(_mcp_probe_app())
        assert client.post("/probe", json={"user_id": "bob"}, headers=AUTH).status_code == 200
        assert mcp_service.list_memories.call_args.kwargs["user_id"] == "alice"

    def test_default_user_when_no_setting(self, shared_key, mcp_service):
        client = TestClient(_mcp_probe_app())
        client.post("/probe", json={"user_id": "bob"}, headers=AUTH)
        assert mcp_service.list_memories.call_args.kwargs["user_id"] == "the-default-user"

    def test_argument_user_trusted_under_opt_out(self, shared_key, mcp_service, monkeypatch):
        monkeypatch.setattr(settings, "neuralscape_legacy_key_trust_body_user_id", True)
        client = TestClient(_mcp_probe_app())
        client.post("/probe", json={"user_id": "bob"}, headers=AUTH)
        assert mcp_service.list_memories.call_args.kwargs["user_id"] == "bob"

    def test_context_var_reset_after_request(self, shared_key, mcp_service):
        from auth import current_user_id

        TestClient(_mcp_probe_app()).post("/probe", json={}, headers=AUTH)
        assert current_user_id.get() is None


# ──────────────────────────────────────────────
# Token revocation (NEURALSCAPE_REVOKED_TOKENS)
# ──────────────────────────────────────────────


class TestTokenRevocation:
    @pytest.fixture(autouse=True)
    def per_user_secret(self, monkeypatch):
        monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
        monkeypatch.setattr(settings, "neuralscape_api_key", "")
        monkeypatch.setattr(settings, "neuralscape_revoked_tokens", "")

    @staticmethod
    def _headers(token):
        return {"Authorization": f"Bearer {token}"}

    def test_unrevoked_token_works(self, service):
        token = issue_user_token("alice", SECRET, 3600)
        assert _list(TestClient(app, raise_server_exceptions=False), None,
                     self._headers(token)).status_code == 200

    def test_revoked_token_is_401_and_others_still_work(self, service, monkeypatch):
        leaked = issue_user_token("alice", SECRET, 3600)
        fine = issue_user_token("alice", SECRET, 7200)
        monkeypatch.setattr(settings, "neuralscape_revoked_tokens",
                            f"deadbeef,{token_fingerprint(leaked)}")
        client = TestClient(app, raise_server_exceptions=False)
        assert _list(client, None, self._headers(leaked)).status_code == 401
        assert _list(client, None, self._headers(fine)).status_code == 200

    def test_revoked_non_expiring_token_is_401(self, service, monkeypatch):
        leaked = issue_user_token("ci", SECRET, None)
        monkeypatch.setattr(settings, "neuralscape_revoked_tokens", token_fingerprint(leaked).upper())
        assert _list(TestClient(app, raise_server_exceptions=False), None,
                     self._headers(leaked)).status_code == 401

    def test_verify_user_token_rejects_revoked(self, monkeypatch):
        tok = issue_user_token("alice", SECRET, 3600)
        assert verify_user_token(tok, SECRET) is not None
        monkeypatch.setattr(settings, "neuralscape_revoked_tokens", token_fingerprint(tok))
        assert verify_user_token(tok, SECRET) is None

    def test_revoked_token_cannot_log_in_via_oauth_paste(self, monkeypatch):
        """oauth.py verifies pasted tokens with verify_user_token: revocation applies."""
        import oauth

        tok = issue_user_token("alice", SECRET, 3600)
        monkeypatch.setattr(settings, "neuralscape_revoked_tokens", token_fingerprint(tok))
        assert oauth.verify_user_token(tok, SECRET) is None

    def test_unreadable_revocation_list_fails_closed(self, monkeypatch):
        tok = issue_user_token("alice", SECRET, 3600)

        # A value the CSV parser cannot split: parsing raises.
        monkeypatch.setattr(settings, "neuralscape_revoked_tokens", None)
        assert verify_user_token(tok, SECRET) is None

    def test_unreadable_revocation_list_is_401_at_the_middleware(self, service, monkeypatch):
        tok = issue_user_token("alice", SECRET, 3600)
        monkeypatch.setattr(settings, "neuralscape_revoked_tokens", None)
        resp = _list(TestClient(app, raise_server_exceptions=False), None, self._headers(tok))
        assert resp.status_code == 401

    def test_empty_list_changes_nothing(self):
        assert settings.revoked_token_digests() == set()

    def test_fingerprint_is_sha256_hex(self):
        import hashlib

        assert token_fingerprint("abc") == hashlib.sha256(b"abc").hexdigest()


# ──────────────────────────────────────────────
# Scripts
# ──────────────────────────────────────────────


def _run(script, *args, stdin=None, env_extra=None):
    import os

    env = {**os.environ, "NEURALSCAPE_USER_TOKEN_SECRET": SECRET, **(env_extra or {})}
    return subprocess.run(
        [sys.executable, str(SERVICE_DIR / "scripts" / script), *args],
        input=stdin, capture_output=True, text=True, env=env, timeout=60,
    )


class TestIssueUserTokenScript:
    def test_days_zero_without_no_expiry_fails(self):
        r = _run("issue_user_token.py", "--user", "alice", "--days", "0")
        assert r.returncode != 0
        assert "--no-expiry" in r.stderr
        assert r.stdout.strip() == ""

    def test_negative_days_fails(self):
        assert _run("issue_user_token.py", "--user", "alice", "--days", "-3").returncode != 0

    def test_no_expiry_flag_issues_token_without_exp(self):
        r = _run("issue_user_token.py", "--user", "alice", "--no-expiry")
        assert r.returncode == 0
        payload = verify_user_token(r.stdout.strip(), SECRET)
        assert payload == {"user_id": "alice"}

    def test_no_expiry_with_days_is_rejected(self):
        r = _run("issue_user_token.py", "--user", "alice", "--days", "5", "--no-expiry")
        assert r.returncode != 0

    def test_default_still_expires(self):
        r = _run("issue_user_token.py", "--user", "alice")
        assert r.returncode == 0
        assert "exp" in verify_user_token(r.stdout.strip(), SECRET)


class TestRevokeTokenScript:
    def test_prints_digest_from_stdin_and_never_the_token(self):
        tok = issue_user_token("alice", SECRET, 3600)
        r = _run("revoke_token.py", stdin=tok + "\n")
        assert r.returncode == 0
        assert r.stdout.strip() == token_fingerprint(tok)
        assert tok not in r.stdout + r.stderr

    def test_reads_token_from_env_var(self):
        r = _run("revoke_token.py", "--token-env", "LEAKED", env_extra={"LEAKED": "abc"})
        assert r.stdout.strip() == token_fingerprint("abc")

    def test_empty_input_fails(self):
        assert _run("revoke_token.py", stdin="").returncode == 2
