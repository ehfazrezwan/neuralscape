import base64
import hashlib
import hmac
import json
import subprocess

import pytest

from nsaccept import tokens

# Computed by hand with hashlib/hmac/base64 directly (and cross-checked once
# against the service's own signer): claims {"exp":2000000000,"user_id":"acc-test-a"},
# secret "test-secret".
VECTOR_PAYLOAD_JSON = '{"exp":2000000000,"user_id":"acc-test-a"}'
VECTOR_PAYLOAD_B64 = "eyJleHAiOjIwMDAwMDAwMDAsInVzZXJfaWQiOiJhY2MtdGVzdC1hIn0"
VECTOR_TOKEN = VECTOR_PAYLOAD_B64 + ".-WuXstFERenAf1QEvTCwJ5dzqmx5FTmzMwfqnMUGEfU"


def test_sign_matches_hard_coded_vector():
    got = tokens.sign_claims({"user_id": "acc-test-a", "exp": 2000000000}, "test-secret")
    assert got == VECTOR_TOKEN


def test_mint_local_produces_verifiable_claims():
    tok = tokens.mint_local("acc-x-a", "s3cret", ttl_seconds=100, now=1000)
    payload_b64, sig_b64 = tok.split(".")
    assert "=" not in tok
    expected = hmac.new(b"s3cret", payload_b64.encode(), hashlib.sha256).digest()
    assert base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4)) == expected
    claims = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
    assert claims == {"user_id": "acc-x-a", "exp": 1100}


def test_wrong_secret_token_differs_each_call_and_is_well_formed():
    t1, t2 = tokens.wrong_secret_token("u"), tokens.wrong_secret_token("u")
    assert t1 != t2 and t1.count(".") == 1


def test_empty_secret_rejected():
    with pytest.raises(tokens.TokenError):
        tokens.sign_claims({"user_id": "u"}, "")


def _fake_run(stdout="", returncode=0, stderr=""):
    def run(cmd, **kw):
        run.cmd = cmd
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
    return run


def test_mint_via_compose_command_and_parsing():
    run = _fake_run(stdout="warning: something\n" + VECTOR_TOKEN + "\n")
    got = tokens.mint_via_compose("proj", "acc-test-a", "api", run=run)
    assert got == VECTOR_TOKEN
    assert run.cmd[:6] == ["docker", "compose", "-p", "proj", "exec", "-T"]
    assert "api" in run.cmd and "scripts/issue_user_token.py" in run.cmd


def test_mint_via_compose_failure_is_scrubbed():
    run = _fake_run(returncode=1, stderr=f"boom {VECTOR_TOKEN}")
    with pytest.raises(tokens.TokenError) as ei:
        tokens.mint_via_compose("proj", "u", run=run)
    assert VECTOR_TOKEN not in str(ei.value)


def test_scrub_removes_token_shaped_text():
    assert VECTOR_TOKEN not in tokens.scrub(f"Bearer {VECTOR_TOKEN} end")
