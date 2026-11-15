"""Revoking a credential ends the OAuth sessions minted from it (sec-13, item 3).

Codes, access tokens and refresh tokens carry ``src``: the digest of the pasted
credential (or the federated identity) the session descends from. It survives
refresh rotation, and revoking it (env list or live Redis set) kills the
sessions. Revoking an access/refresh token directly works too; a revoked
refresh token also revokes its family.
"""

import asyncio
import base64
import hashlib
import secrets
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main
import oauth
import tokens
from config import settings
from login_providers import AuthorizeContext, LoginResult
from main import app
from tokens import (
    REVOKED_SET_KEY,
    identity_fingerprint,
    issue_user_token,
    token_fingerprint,
    verify_payload,
    verify_user_token,
)

SECRET = secrets.token_hex(16)
USER = "alice"
REDIRECT = "https://claude.ai/api/mcp/auth_callback"


def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


VERIFIER = _b64(secrets.token_bytes(40))
CHALLENGE = _b64(hashlib.sha256(VERIFIER.encode()).digest())


@pytest.fixture(autouse=True)
def stack(monkeypatch):
    main._async_memory = None
    tm = MagicMock()
    tm.connect = AsyncMock()
    tm.close = AsyncMock()
    monkeypatch.setattr(main, "_task_manager", tm)
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_public_url", "https://neuralscape.example.com")
    monkeypatch.setattr(settings, "neuralscape_revoked_tokens", "")
    monkeypatch.setattr(settings, "neuralscape_auth_disabled", False)
    monkeypatch.setattr(settings, "revocation_refresh_interval_seconds", 0.0)

    seen, seen_refresh, fams = set(), set(), {}

    async def consume(jti):
        first = jti not in seen
        seen.add(jti)
        return first

    async def consume_refresh(jti, ttl):
        first = jti not in seen_refresh
        seen_refresh.add(jti)
        return first

    async def family_revoked(fam):
        return fam in fams

    async def revoke_family(fam, ttl):
        fams[fam] = ttl

    monkeypatch.setattr(oauth, "_consume_code_jti", consume)
    monkeypatch.setattr(oauth, "_consume_refresh_jti", consume_refresh)
    monkeypatch.setattr(oauth, "_family_revoked", family_revoked)
    monkeypatch.setattr(oauth, "_revoke_family", revoke_family)

    class Redis:
        members = set()

        async def smembers(self, key):
            return {m.encode() for m in Redis.members} if key == REVOKED_SET_KEY else set()

        async def aclose(self):
            return None

    Redis.members = set()
    monkeypatch.setattr(tokens, "_new_redis_client", Redis)
    tokens.reset_revocation_snapshot()
    stack_obj = type("S", (), {"families": fams, "redis": Redis})
    return stack_obj


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def _register(client):
    r = client.post("/oauth/register", json={"redirect_uris": [REDIRECT]})
    assert r.status_code == 201, r.text
    return r.json()["client_id"]


def _code_from(resp):
    assert resp.status_code == 303, resp.text
    return resp.headers["location"].split("code=")[1].split("&")[0]


def _paste(client, client_id, pasted):
    return client.post(
        "/oauth/authorize",
        data={
            "client_id": client_id,
            "redirect_uri": REDIRECT,
            "state": "s",
            "code_challenge": CHALLENGE,
            "code_challenge_method": "S256",
            "token": pasted,
        },
        follow_redirects=False,
    )


def _exchange(client, client_id, code):
    return client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": VERIFIER,
            "redirect_uri": REDIRECT,
            "client_id": client_id,
        },
    )


def _refresh(client, client_id, refresh):
    return client.post(
        "/oauth/token",
        data={"grant_type": "refresh_token", "refresh_token": refresh, "client_id": client_id},
    )


def _session(client, ttl=3600):
    """Paste a fresh admin token, finish the flow. Returns (pasted, client_id, tokens)."""
    pasted = issue_user_token(USER, SECRET, ttl)
    client_id = _register(client)
    code = _code_from(_paste(client, client_id, pasted))
    resp = _exchange(client, client_id, code)
    assert resp.status_code == 200, resp.text
    return pasted, client_id, resp.json()


def _revoke_env(monkeypatch, *digests):
    monkeypatch.setattr(settings, "neuralscape_revoked_tokens", ",".join(digests))


class TestSrcClaim:
    def test_pasted_credential_is_stamped_into_code_access_and_refresh(self, client):
        pasted = issue_user_token(USER, SECRET, 3600)
        client_id = _register(client)
        code = _code_from(_paste(client, client_id, pasted))
        src = token_fingerprint(pasted)
        assert verify_payload(code, SECRET)["src"] == src
        tok = _exchange(client, client_id, code).json()
        assert verify_payload(tok["access_token"], SECRET)["src"] == src
        assert verify_payload(tok["refresh_token"], SECRET)["src"] == src

    def test_src_survives_refresh_rotation(self, client):
        pasted, client_id, tok = _session(client)
        new = _refresh(client, client_id, tok["refresh_token"]).json()
        again = _refresh(client, client_id, new["refresh_token"]).json()
        for t in (new, again):
            assert verify_payload(t["access_token"], SECRET)["src"] == token_fingerprint(pasted)
            assert verify_payload(t["refresh_token"], SECRET)["src"] == token_fingerprint(pasted)

    def test_pasting_an_oauth_access_token_keeps_the_original_src(self, client):
        pasted, client_id, tok = _session(client)
        code = _code_from(_paste(client, client_id, tok["access_token"]))
        assert verify_payload(code, SECRET)["src"] == token_fingerprint(pasted)

    def test_federated_login_stamps_an_identity_marker(self, client, monkeypatch):
        monkeypatch.setattr(settings, "auth_provider", "google")
        client_id = _register(client)
        ctx = AuthorizeContext(client_id, REDIRECT, "s", CHALLENGE, "S256", "")
        resp = asyncio.run(oauth._finish_login(LoginResult(ctx=ctx, user_id=USER)))
        code = _code_from(resp)
        assert verify_payload(code, SECRET)["src"] == identity_fingerprint("google", USER)


class TestRevokingTheSourceEndsTheSession:
    def test_env_list_ends_access_and_refresh(self, client, monkeypatch):
        pasted, client_id, tok = _session(client)
        other_pasted, other_cid, other = _session(client, ttl=7200)
        assert verify_user_token(tok["access_token"], SECRET) is not None
        _revoke_env(monkeypatch, token_fingerprint(pasted))
        assert verify_user_token(tok["access_token"], SECRET) is None
        r = _refresh(client, client_id, tok["refresh_token"])
        assert r.status_code == 400 and r.json()["error"] == "invalid_grant"
        # another session (different pasted token) is untouched
        assert verify_user_token(other["access_token"], SECRET) is not None
        assert _refresh(client, other_cid, other["refresh_token"]).status_code == 200

    def test_live_redis_denylist_ends_the_session_without_restart(self, client, stack):
        pasted, client_id, tok = _session(client)
        stack.redis.members.add(token_fingerprint(pasted))
        r = _refresh(client, client_id, tok["refresh_token"])
        assert r.status_code == 400 and r.json()["error"] == "invalid_grant"
        asyncio.run(tokens.refresh_revocations(force=True))
        assert verify_user_token(tok["access_token"], SECRET) is None

    def test_rotated_descendants_die_too(self, client, monkeypatch):
        pasted, client_id, tok = _session(client)
        new = _refresh(client, client_id, tok["refresh_token"]).json()
        _revoke_env(monkeypatch, token_fingerprint(pasted))
        assert verify_user_token(new["access_token"], SECRET) is None
        assert _refresh(client, client_id, new["refresh_token"]).status_code == 400

    def test_unrevoke_restores_the_session(self, client, stack):
        pasted, client_id, tok = _session(client)
        stack.redis.members.add(token_fingerprint(pasted))
        assert _refresh(client, client_id, tok["refresh_token"]).status_code == 400
        stack.redis.members.clear()
        # the refused refresh did not consume the token: it works again
        assert _refresh(client, client_id, tok["refresh_token"]).status_code == 200

    def test_revoked_source_cannot_redeem_an_issued_code(self, client, monkeypatch):
        pasted = issue_user_token(USER, SECRET, 3600)
        client_id = _register(client)
        code = _code_from(_paste(client, client_id, pasted))
        _revoke_env(monkeypatch, token_fingerprint(pasted))
        r = _exchange(client, client_id, code)
        assert r.status_code == 400 and r.json()["error"] == "invalid_grant"

    def test_revoked_identity_ends_a_federated_session(self, client, monkeypatch):
        monkeypatch.setattr(settings, "auth_provider", "supabase")
        client_id = _register(client)
        ctx = AuthorizeContext(client_id, REDIRECT, "s", CHALLENGE, "S256", "")
        code = _code_from(asyncio.run(oauth._finish_login(LoginResult(ctx=ctx, user_id=USER))))
        tok = _exchange(client, client_id, code).json()
        _revoke_env(monkeypatch, identity_fingerprint("supabase", USER))
        assert verify_user_token(tok["access_token"], SECRET) is None
        assert _refresh(client, client_id, tok["refresh_token"]).status_code == 400


class TestRevokingASessionTokenDirectly:
    def test_revoked_access_token_is_refused_but_the_refresh_still_works(self, client, monkeypatch):
        _, client_id, tok = _session(client)
        _revoke_env(monkeypatch, token_fingerprint(tok["access_token"]))
        assert verify_user_token(tok["access_token"], SECRET) is None
        assert _refresh(client, client_id, tok["refresh_token"]).status_code == 200

    def test_revoked_refresh_token_is_refused_and_revokes_its_family(self, client, stack, monkeypatch):
        _, client_id, tok = _session(client)
        fam = verify_payload(tok["refresh_token"], SECRET)["fam"]
        _revoke_env(monkeypatch, token_fingerprint(tok["refresh_token"]))
        r = _refresh(client, client_id, tok["refresh_token"])
        assert r.status_code == 400 and r.json()["error"] == "invalid_grant"
        assert fam in stack.families
