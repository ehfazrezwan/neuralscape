"""Tests for HMAC user-token primitives in tokens.py.

These are pure-Python — no FastAPI / Qdrant / Neo4j needed — so the test
exercises the full sign/verify cycle, malformed inputs, expiry, and
tamper-resistance without spinning up the service.
"""
import time

import pytest

from tokens import issue_user_token, verify_user_token


class TestIssueUserToken:
    def test_emits_two_segment_format(self):
        token = issue_user_token("alice", "topsecret", 3600)
        assert "." in token
        parts = token.split(".")
        assert len(parts) == 2
        assert parts[0]  # payload segment non-empty
        assert parts[1]  # signature segment non-empty

    def test_two_different_users_produce_different_tokens(self):
        a = issue_user_token("alice", "topsecret", 3600)
        b = issue_user_token("bob", "topsecret", 3600)
        assert a != b

    def test_empty_user_id_rejected(self):
        with pytest.raises(ValueError, match="user_id is required"):
            issue_user_token("", "topsecret", 3600)

    def test_empty_secret_rejected(self):
        with pytest.raises(ValueError, match="secret is required"):
            issue_user_token("alice", "", 3600)


class TestVerifyUserToken:
    def test_valid_token_returns_payload_with_user_id(self):
        token = issue_user_token("alice", "topsecret", 3600)
        payload = verify_user_token(token, "topsecret")
        assert payload is not None
        assert payload["user_id"] == "alice"
        assert "exp" in payload

    def test_wrong_secret_returns_none(self):
        token = issue_user_token("alice", "topsecret", 3600)
        assert verify_user_token(token, "different-secret") is None

    def test_tampered_signature_returns_none(self):
        token = issue_user_token("alice", "topsecret", 3600)
        # Flip the last character of the signature segment
        flipped = token[:-1] + ("a" if token[-1] != "a" else "b")
        assert verify_user_token(flipped, "topsecret") is None

    def test_tampered_payload_returns_none(self):
        """Modifying the payload without re-signing must fail verification."""
        token = issue_user_token("alice", "topsecret", 3600)
        payload_seg, sig_seg = token.split(".", 1)
        # Add a stray byte to the payload segment — won't decode cleanly, or
        # if it does, the signature won't match.
        broken = f"{payload_seg}X.{sig_seg}"
        assert verify_user_token(broken, "topsecret") is None

    def test_expired_token_returns_none(self):
        """A token with exp in the past must verify as None."""
        token = issue_user_token("alice", "topsecret", -1)
        time.sleep(0.05)
        assert verify_user_token(token, "topsecret") is None

    def test_no_dot_returns_none(self):
        """An opaque legacy API key shouldn't accidentally validate as a token."""
        assert verify_user_token("just-an-opaque-shared-api-key", "topsecret") is None

    def test_three_segments_returns_none(self):
        assert verify_user_token("a.b.c", "topsecret") is None

    def test_empty_segments_return_none(self):
        assert verify_user_token(".", "topsecret") is None
        assert verify_user_token("a.", "topsecret") is None
        assert verify_user_token(".b", "topsecret") is None

    def test_empty_token_returns_none(self):
        assert verify_user_token("", "topsecret") is None

    def test_empty_secret_returns_none(self):
        token = issue_user_token("alice", "topsecret", 3600)
        assert verify_user_token(token, "") is None

    def test_garbage_b64_returns_none(self):
        # Two segments but neither is valid base64
        assert verify_user_token("!@#$.%^&*", "topsecret") is None

    def test_payload_must_be_a_dict(self):
        """A token whose payload base64-decodes to a JSON list, not dict."""
        # Hand-craft: payload = JSON list, signed correctly.
        import base64, hmac, hashlib, json
        bad_payload = json.dumps(["alice", 9999999999]).encode()
        b64 = base64.urlsafe_b64encode(bad_payload).rstrip(b"=").decode()
        sig = hmac.new(b"topsecret", b64.encode(), hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
        token = f"{b64}.{sig_b64}"
        assert verify_user_token(token, "topsecret") is None

    def test_payload_must_have_string_user_id(self):
        """Numeric user_id is rejected even if the signature is valid."""
        import base64, hmac, hashlib, json
        bad_payload = json.dumps({"user_id": 42, "exp": 9999999999}).encode()
        b64 = base64.urlsafe_b64encode(bad_payload).rstrip(b"=").decode()
        sig = hmac.new(b"topsecret", b64.encode(), hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
        token = f"{b64}.{sig_b64}"
        assert verify_user_token(token, "topsecret") is None

    def test_payload_with_no_exp_is_accepted_forever(self):
        """Missing exp claim means non-expiring (caller's choice)."""
        import base64, hmac, hashlib, json
        payload = json.dumps({"user_id": "alice"}).encode()
        b64 = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        sig = hmac.new(b"topsecret", b64.encode(), hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
        token = f"{b64}.{sig_b64}"
        p = verify_user_token(token, "topsecret")
        assert p is not None and p["user_id"] == "alice"

    def test_long_user_id_round_trips(self):
        long_id = "user_" + "x" * 90
        token = issue_user_token(long_id, "topsecret", 3600)
        p = verify_user_token(token, "topsecret")
        assert p is not None and p["user_id"] == long_id
