"""Self-tests for the `legacy-key` profile machinery (no stack): the `harness-env` header, the generated key
reaching the harness and never the results, and the fixtures failing (not skipping) without it."""

import json
import os
import re
from pathlib import Path

import pytest

from nsaccept import fixtures, profile, runctx, tokens

ACCEPTANCE = Path(__file__).resolve().parents[1]
KEY = "ab" * 32  # a stand-in generated key (hex32): never a real secret


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ("NS_TOKEN_SECRET", "NS_TOKEN_A", "NS_TOKEN_B", "NS_USER_A", "NS_USER_B", "NS_COMPOSE_PROJECT", "NS_RUN_ID"):
        monkeypatch.delenv(k, raising=False)
    tokens._SECRETS.clear()
    yield
    tokens._SECRETS.clear()


def test_the_shipped_profile_binds_a_generated_key_to_a_fixed_identity_and_hands_it_over():
    p = profile.load(ACCEPTANCE / "profiles" / "legacy-key.env")
    assert p.placeholders == {"NEURALSCAPE_API_KEY": "hex32"}
    assert p.flags["NEURALSCAPE_API_KEY_USER_ID"] == "acc-legacy"  # an identifier, not redacted
    assert "NEURALSCAPE_LEGACY_KEY_TRUST_BODY_USER_ID" not in p.flags and "NEURALSCAPE_USER_TOKEN_SECRET" not in p.flags
    assert p.harness_env() == {"NS_LEGACY_KEY": "NEURALSCAPE_API_KEY"} and p.harness_env_errors() == []


@pytest.mark.parametrize("path", sorted((ACCEPTANCE / "profiles").glob("*.env")), ids=lambda p: p.stem)
def test_every_shipped_profiles_harness_env_headers_are_valid(path):
    assert profile.load(path).harness_env_errors() == []


def test_harness_env_rejects_a_non_ns_variable_a_literal_and_a_missing_placeholder():
    text = "# harness-env: PATH=X_KEY, NS_A=PLAIN, NS_B=MISSING, noequals\nX_KEY=<generate:hex32>\nPLAIN=literal\n"
    p = profile.parse_text(text)
    errors = p.harness_env_errors()
    assert len(errors) == 4 and any("PATH" in e for e in errors) and any("PLAIN" in e for e in errors) and any("MISSING" in e for e in errors)
    ok = profile.parse_text("# harness-env: NS_A=X_KEY\nX_KEY=<generate:hex32>\n")
    assert ok.harness_env() == {"NS_A": "X_KEY"} and ok.harness_env_errors() == []


def test_a_literal_key_is_still_redacted_but_an_identifier_is_not():
    p = profile.parse_text("NEURALSCAPE_API_KEY=hunter2hunter2\nNEURALSCAPE_API_KEY_USER_ID=acc-legacy\nSOME_SECRET_ID=abc\n")
    assert p.flags["NEURALSCAPE_API_KEY"] == profile.REDACTED
    assert p.flags["NEURALSCAPE_API_KEY_USER_ID"] == "acc-legacy" and p.flags["SOME_SECRET_ID"] == "abc"


def test_the_handed_over_key_reaches_the_ctx_but_not_its_repr_or_a_scrubbed_message(monkeypatch):
    monkeypatch.setenv("NS_TOKEN_SECRET", "s")
    c = runctx.build("http://x", "legacy-key", profile_flags={"NEURALSCAPE_API_KEY_USER_ID": "acc-legacy"}, legacy_key=KEY)
    assert c.legacy_key == KEY and c.legacy_user == "acc-legacy"
    assert KEY not in repr(c)
    assert KEY not in tokens.scrub(f"GET failed with Authorization: Bearer {KEY}") and "<redacted-secret>" in tokens.scrub(f"key={KEY}")


def test_no_key_means_no_legacy_identity(monkeypatch):
    monkeypatch.setenv("NS_TOKEN_SECRET", "s")
    c = runctx.build("http://x", "default")
    assert c.legacy_key is None and c.legacy_user is None


def test_short_or_empty_secrets_are_not_registered():
    tokens.register_secret("")
    tokens.register_secret(None)
    tokens.register_secret("abc")
    assert tokens._SECRETS == set()
    assert tokens.scrub("plain text abc") == "plain text abc"


def test_the_legacy_fixtures_fail_loudly_without_the_key_and_never_skip(monkeypatch):
    monkeypatch.setenv("NS_TOKEN_SECRET", "s")
    c = runctx.build("http://x", "legacy-key", profile_flags={"NEURALSCAPE_API_KEY_USER_ID": "acc-legacy"})
    with pytest.raises(pytest.fail.Exception, match="NS_LEGACY_KEY is missing"):
        fixtures._legacy_ready(c)
    c2 = runctx.build("http://x", "legacy-key", legacy_key=KEY)
    with pytest.raises(pytest.fail.Exception, match="missing from the profile flags"):
        fixtures._legacy_ready(c2)
    c3 = runctx.build("http://x", "legacy-key", profile_flags={"NEURALSCAPE_API_KEY_USER_ID": "acc-legacy"}, legacy_key=KEY)
    fixtures._legacy_ready(c3)  # complete: no failure


def test_the_scenarios_never_interpolate_the_key_into_a_message():
    for line in (ACCEPTANCE / "scenarios" / "test_legacy_key.py").read_text().splitlines():
        assert not ("legacy_key" in line and re.search(r"\bf[\"']", line)), f"an f-string mentions the legacy key: {line.strip()}"
