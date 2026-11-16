"""OAUTH_REDIRECT_HOST_ALLOWLIST: optional host restriction on OAuth redirect URIs.

Empty (the default) keeps the existing rule. When set, a redirect host must be
listed at registration and again at authorize (GET and POST), so a client
registered before the list was set is refused. Loopback hosts stay allowed.
"""

import base64
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main
import oauth
from config import settings
from main import app
from tokens import issue_user_token, sign_payload

SECRET = "oauth-allowlist-test-secret"
PUBLIC_URL = "https://neuralscape.example.com"
LISTED = "https://claude.ai/api/mcp/auth_callback"
UNLISTED = "https://phish.example.net/cb"


def _challenge() -> str:
    verifier = "verifier-bytes-must-be-43-128-chars-long-xxxxxxxx"
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    main._async_memory = None
    tm = MagicMock()
    tm.connect = AsyncMock()
    tm.close = AsyncMock()
    monkeypatch.setattr(main, "_task_manager", tm)
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_public_url", PUBLIC_URL)
    monkeypatch.setattr(settings, "auth_provider", "token")
    monkeypatch.setattr(settings, "oauth_redirect_host_allowlist", "")
    seen: set = set()

    async def _consume(jti):
        if jti in seen:
            return False
        seen.add(jti)
        return True

    with patch.object(oauth, "_consume_code_jti", side_effect=_consume, create=True):
        yield


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def _allow(monkeypatch, value: str) -> None:
    monkeypatch.setattr(settings, "oauth_redirect_host_allowlist", value)


def _register(client, uri):
    return client.post("/oauth/register", json={"redirect_uris": [uri]})


def _authorize_get(client, client_id, uri):
    return client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": uri,
            "code_challenge": _challenge(),
            "code_challenge_method": "S256",
        },
    )


def _authorize_post(client, client_id, uri):
    return client.post(
        "/oauth/authorize",
        data={
            "client_id": client_id,
            "redirect_uri": uri,
            "code_challenge": _challenge(),
            "code_challenge_method": "S256",
            "token": issue_user_token("alice", SECRET, 600),
        },
        follow_redirects=False,
    )


def test_empty_list_keeps_current_behaviour(client):
    assert _register(client, LISTED).status_code == 201
    assert _register(client, UNLISTED).status_code == 201
    assert _register(client, "http://evil.example.com/cb").status_code == 400  # unchanged rule


def test_listed_host_registers_and_unlisted_is_refused(client, monkeypatch):
    _allow(monkeypatch, "claude.ai")
    assert _register(client, LISTED).status_code == 201
    refused = _register(client, UNLISTED)
    assert refused.status_code == 400
    body = refused.json()
    assert body["error"] == "invalid_redirect_uri"
    assert "allowlist" in body["error_description"]


def test_one_unlisted_uri_among_listed_ones_refuses_the_registration(client, monkeypatch):
    _allow(monkeypatch, "claude.ai")
    resp = client.post("/oauth/register", json={"redirect_uris": [LISTED, UNLISTED]})
    assert resp.status_code == 400 and resp.json()["error"] == "invalid_redirect_uri"


def test_matching_is_case_insensitive_on_entry_and_uri(client, monkeypatch):
    _allow(monkeypatch, " Claude.AI , other.example.org ")
    assert _register(client, "https://CLAUDE.ai/cb").status_code == 201
    assert _register(client, "https://other.example.org/cb").status_code == 201


@pytest.mark.parametrize(
    "entry,uri,ok",
    [
        ("claude.ai", "https://claude.ai/cb", True),
        ("claude.ai", "https://claude.ai:443/cb", True),  # explicit default port == bare host
        ("claude.ai", "https://claude.ai:8443/cb", False),  # bare host = default port only
        ("claude.ai:8443", "https://claude.ai:8443/cb", True),
        ("claude.ai:8443", "https://claude.ai/cb", False),  # host:port does not open the default port
        ("claude.ai:8443", "https://claude.ai:9000/cb", False),
        ("claude.ai", "https://sub.claude.ai/cb", False),  # exact hostnames, no suffix match
        ("claude.ai", "https://evilclaude.ai/cb", False),
        ("claude.ai", "https://claude.ai.evil.example/cb", False),
    ],
)
def test_bare_host_vs_host_port(client, monkeypatch, entry, uri, ok):
    _allow(monkeypatch, entry)
    assert _register(client, uri).status_code == (201 if ok else 400)


@pytest.mark.parametrize(
    "uri",
    ["http://127.0.0.1:53682/callback", "http://localhost:7777/cb", "http://[::1]:7777/cb"],
)
def test_loopback_native_clients_stay_allowed_when_list_is_set(client, monkeypatch, uri):
    _allow(monkeypatch, "claude.ai")
    assert _register(client, uri).status_code == 201


def test_list_does_not_widen_the_base_validator(client, monkeypatch):
    _allow(monkeypatch, "evil.example.com")
    assert _register(client, "http://evil.example.com/cb").status_code == 400  # http still refused
    assert _register(client, "https://evil.example.com@claude.ai/cb").status_code == 400  # userinfo


def test_get_authorize_refuses_a_client_registered_before_the_list_was_set(client, monkeypatch):
    client_id = _register(client, UNLISTED).json()["client_id"]
    assert _authorize_get(client, client_id, UNLISTED).status_code == 200  # list empty
    _allow(monkeypatch, "claude.ai")
    resp = _authorize_get(client, client_id, UNLISTED)
    assert resp.status_code == 400 and resp.json()["error"] == "invalid_request"


def test_post_authorize_refuses_a_client_registered_before_the_list_was_set(client, monkeypatch):
    client_id = _register(client, UNLISTED).json()["client_id"]
    _allow(monkeypatch, "claude.ai")
    resp = _authorize_post(client, client_id, UNLISTED)
    assert resp.status_code == 400 and resp.json()["error"] == "invalid_request"
    assert "location" not in resp.headers  # no code, no redirect


def test_listed_client_still_authorizes_with_the_list_set(client, monkeypatch):
    _allow(monkeypatch, "claude.ai")
    client_id = _register(client, LISTED).json()["client_id"]
    assert _authorize_get(client, client_id, LISTED).status_code == 200
    resp = _authorize_post(client, client_id, LISTED)
    assert resp.status_code == 303 and resp.headers["location"].startswith(LISTED)


def test_loopback_client_authorizes_with_the_list_set(client, monkeypatch):
    uri = "http://127.0.0.1:53682/callback"
    _allow(monkeypatch, "claude.ai")
    client_id = _register(client, uri).json()["client_id"]
    assert _authorize_get(client, client_id, uri).status_code == 200
    assert _authorize_post(client, client_id, uri).status_code == 303


def test_hand_signed_client_with_unlisted_host_is_refused_at_authorize(client, monkeypatch):
    """A client_id is a signed payload; the list is enforced at authorize regardless of how it was issued."""
    _allow(monkeypatch, "claude.ai")
    forged = sign_payload({"typ": "client", "redirect_uris": [UNLISTED], "iat": 1, "jti": "x"}, SECRET)
    assert _authorize_get(client, forged, UNLISTED).status_code == 400
    assert _authorize_post(client, forged, UNLISTED).status_code == 400


def test_config_parse_helper(monkeypatch):
    _allow(monkeypatch, " A.example , b.example:8443,, ")
    assert settings.oauth_redirect_host_allowlist_set() == {"a.example", "b.example:8443"}
    _allow(monkeypatch, "")
    assert settings.oauth_redirect_host_allowlist_set() == set()
