"""Live token revocation on a real Redis (sec-13).

A revoked per-user token must stop working with NO restart: the operator runs
``scripts/revoke_token.py --redis`` (a real subprocess, as in production), and
the bearer middleware (the real ``BearerAuthMiddleware``) refuses the token
within the snapshot interval. ``--unrevoke`` restores it, and an unreachable
Redis keeps enforcing the env list and the last-known snapshot.

Needs only Redis (the ``stores`` fixture probes it with the others); every
digest written here is removed again on cleanup.
"""

import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import pytest
import redis
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import tokens
from auth import BearerAuthMiddleware
from config import settings
from tokens import REVOKED_SET_KEY, issue_user_token, token_fingerprint

pytestmark = pytest.mark.integration

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "revoke_token.py"
UNREACHABLE_REDIS = "redis://127.0.0.1:1"


@pytest.fixture
def secret():
    return secrets.token_hex(24)


@pytest.fixture
def api(monkeypatch, stores, secret, flags):
    """The real middleware on a real Redis, 0 s snapshot interval (every request re-reads)."""
    flags(
        neuralscape_user_token_secret=secret,
        neuralscape_api_key="",
        neuralscape_auth_disabled=False,
        neuralscape_revoked_tokens="",
        redis_url=stores.redis_url,
        revocation_refresh_interval_seconds=0.0,
    )
    tokens.reset_revocation_snapshot()
    app = FastAPI()
    app.add_middleware(BearerAuthMiddleware)

    @app.get("/v1/whoami")
    async def whoami(request: Request):
        return {"user_id": request.state.user_id}

    yield TestClient(app, raise_server_exceptions=False)
    tokens.reset_revocation_snapshot()


def _get(api, token):
    return api.get("/v1/whoami", headers={"Authorization": f"Bearer {token}"})


def _script(stores, *args, token=None):
    """Run the real CLI; the token goes through the child's environment, never argv."""
    env = {**os.environ, "REDIS_URL": stores.redis_url}
    if token is not None:
        env["NS_TEST_TOKEN"] = token
        args = ("--token-env", "NS_TEST_TOKEN", *args)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60
    )


@pytest.fixture
def digests(stores, unique_ns):
    """Remove every digest a test revoked, whatever happens."""
    added = []
    client = redis.Redis.from_url(stores.redis_url)
    unique_ns.on_cleanup(lambda: client.srem(REVOKED_SET_KEY, *added) if added else None)
    unique_ns.on_cleanup(client.close)
    return added


def test_revoke_via_script_is_401_with_no_restart_and_unrevoke_restores(
    api, stores, secret, digests, unique_ns
):
    token = issue_user_token(unique_ns.user_id, secret, 3600)
    other = issue_user_token(unique_ns.user_id, secret, 7200)
    digests.append(token_fingerprint(token))
    assert _get(api, token).status_code == 200

    done = _script(stores, "--redis", token=token)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == token_fingerprint(token)
    assert token not in done.stdout + done.stderr

    assert _get(api, token).status_code == 401  # same process, no restart
    assert _get(api, other).status_code == 200  # an unrevoked token is unaffected

    listed = _script(stores, "--list")
    assert token_fingerprint(token) in listed.stdout.split()

    assert _script(stores, "--unrevoke", token=token).returncode == 0
    assert _get(api, token).status_code == 200


def test_revocation_lands_within_the_snapshot_interval(api, stores, secret, digests, unique_ns, flags):
    flags(revocation_refresh_interval_seconds=1.0)
    token = issue_user_token(unique_ns.user_id, secret, 3600)
    digests.append(token_fingerprint(token))
    assert _get(api, token).status_code == 200
    assert _script(stores, "--redis", token=token).returncode == 0
    deadline = time.monotonic() + 1.0 + 3.0  # interval + slack
    while _get(api, token).status_code != 401:
        assert time.monotonic() < deadline, "revocation not enforced within the interval"
        time.sleep(0.1)


def test_cold_start_without_redis_env_list_holds_and_other_tokens_get_503(api, secret, flags, unique_ns):
    leaked = issue_user_token(unique_ns.user_id, secret, 3600)
    fine = issue_user_token(unique_ns.user_id, secret, 7200)
    flags(redis_url=UNREACHABLE_REDIS, neuralscape_revoked_tokens=token_fingerprint(leaked))
    tokens.reset_revocation_snapshot()  # cold start, Redis down
    assert _get(api, leaked).status_code == 401  # env list: the hard guarantee
    refused = _get(api, fine)
    assert refused.status_code == 503 and refused.headers["retry-after"]  # retry, do not drop credentials


def test_flag_off_cold_start_without_redis_fails_open_on_the_redis_list(api, secret, flags, unique_ns):
    leaked = issue_user_token(unique_ns.user_id, secret, 3600)
    fine = issue_user_token(unique_ns.user_id, secret, 7200)
    flags(
        redis_url=UNREACHABLE_REDIS,
        neuralscape_revoked_tokens=token_fingerprint(leaked),
        revocation_fail_closed_on_cold_start=False,
    )
    tokens.reset_revocation_snapshot()
    assert _get(api, leaked).status_code == 401
    assert _get(api, fine).status_code == 200


def test_api_started_without_redis_recovers_when_redis_appears(
    api, stores, secret, digests, unique_ns, flags
):
    """Redis cannot be stopped on the gate, so "down at start" is modelled by starting the
    app against an unreachable URL and then pointing it at the real Redis (what a Redis that
    comes up later looks like to the process: the same URL starts answering)."""
    revoked = issue_user_token(unique_ns.user_id, secret, 3600)
    fine = issue_user_token(unique_ns.user_id, secret, 7200)
    digests.append(token_fingerprint(revoked))
    assert _script(stores, "--redis", token=revoked).returncode == 0

    flags(redis_url=UNREACHABLE_REDIS, revocation_refresh_interval_seconds=1.0,
          revocation_cold_retry_seconds=0.2)
    tokens.reset_revocation_snapshot()
    assert _get(api, fine).status_code == 503  # started without Redis

    flags(redis_url=stores.redis_url)  # Redis is reachable now
    deadline = time.monotonic() + 0.2 + 3.0  # cold retry interval + slack
    while _get(api, fine).status_code != 200:
        assert time.monotonic() < deadline, "did not recover within the cold-start retry interval"
        time.sleep(0.05)
    assert _get(api, revoked).status_code == 401  # the digest already in Redis is enforced


def test_redis_going_away_keeps_the_last_snapshot_then_catches_up(
    api, stores, secret, digests, unique_ns, flags
):
    first = issue_user_token(unique_ns.user_id, secret, 3600)
    second = issue_user_token(unique_ns.user_id, secret, 7200)
    digests.extend([token_fingerprint(first), token_fingerprint(second)])
    assert _script(stores, "--redis", token=first).returncode == 0
    assert _get(api, first).status_code == 401  # snapshot now holds `first`

    flags(redis_url=UNREACHABLE_REDIS)  # outage
    assert _get(api, first).status_code == 401  # still enforced from the snapshot
    assert _get(api, second).status_code == 200

    flags(redis_url=stores.redis_url)  # Redis back
    assert _script(stores, "--redis", token=second).returncode == 0
    assert _get(api, second).status_code == 401
    assert _get(api, first).status_code == 401


def test_oauth_session_minted_from_a_pasted_token_dies_when_it_is_revoked(
    api, stores, secret, digests, unique_ns, flags
):
    """Paste a token on the real consent flow (real Redis replay guard), then revoke it live."""
    import base64
    import hashlib

    import main

    flags(neuralscape_public_url="https://neuralscape.example.com")
    client = TestClient(main.app, raise_server_exceptions=False)
    pasted = issue_user_token(unique_ns.user_id, secret, 3600)
    digests.append(token_fingerprint(pasted))
    redirect = "https://claude.ai/api/mcp/auth_callback"
    client_id = client.post("/oauth/register", json={"redirect_uris": [redirect]}).json()["client_id"]
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(40)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()

    def grant(**data):
        return client.post("/oauth/token", data={"client_id": client_id, **data})

    resp = client.post(
        "/oauth/authorize",
        data={"client_id": client_id, "redirect_uri": redirect, "state": "s", "code_challenge": challenge,
              "code_challenge_method": "S256", "token": pasted},
        follow_redirects=False,
    )
    code = resp.headers["location"].split("code=")[1].split("&")[0]
    tok = grant(grant_type="authorization_code", code=code, code_verifier=verifier, redirect_uri=redirect).json()
    refresh = tok["refresh_token"]
    assert verify_user_token_ok(tok["access_token"], secret)
    rotated = grant(grant_type="refresh_token", refresh_token=refresh).json()  # works while unrevoked
    assert "refresh_token" in rotated

    assert _script(stores, "--redis", token=pasted).returncode == 0

    refused = grant(grant_type="refresh_token", refresh_token=rotated["refresh_token"])
    assert refused.status_code == 400 and refused.json()["error"] == "invalid_grant"
    assert not verify_user_token_ok(rotated["access_token"], secret)
    assert _get(api, rotated["access_token"]).status_code == 401  # the middleware agrees

    assert _script(stores, "--unrevoke", token=pasted).returncode == 0
    assert grant(grant_type="refresh_token", refresh_token=rotated["refresh_token"]).status_code == 200


def verify_user_token_ok(token, secret):
    import asyncio

    asyncio.run(tokens.refresh_revocations(force=True))
    return tokens.verify_user_token(token, secret) is not None
