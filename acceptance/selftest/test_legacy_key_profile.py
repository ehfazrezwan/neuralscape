"""Self-tests for the `legacy-key` profile machinery (no stack): the `harness-env` header, the generated key
reaching the harness and never the results, and the fixtures failing (not skipping) without it."""

import json
import os
import re
from pathlib import Path

import pytest

import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nsaccept import fixtures, profile, pytest_plugin, runctx, tokens

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


# ── isolation from the union profiles ────────────────────────────


def test_the_legacy_key_profile_is_exclusive_and_no_other_profile_touches_the_auth_surface():
    profiles_dir = ACCEPTANCE / "profiles"
    mine = profile.load(profiles_dir / "legacy-key.env")
    assert mine.exclusive(), "legacy-key.env must declare `# exclusive: own-restart`"
    for path in profiles_dir.glob("*.env"):
        if path.stem == "legacy-key":
            continue
        other = profile.load(path)
        assert not any(k.startswith("NEURALSCAPE_API_KEY") or k.startswith("NEURALSCAPE_LEGACY_KEY") for k in other.flags), (
            f"{path.name} sets the legacy shared key: it would change the auth surface of its whole run"
        )
        assert not other.exclusive(), f"{path.name} is exclusive: only legacy-key is"
    assert "own-restart" in (ACCEPTANCE / "README.md").read_text(), "README Profiles must document the exclusive profile"


def test_only_the_legacy_key_scenarios_use_the_legacy_fixtures_or_profile():
    for path in (ACCEPTANCE / "scenarios").glob("test_*.py"):
        text = path.read_text()
        uses = re.search(r"\bmcp_legacy\b|profiles\([^)]*legacy-key", text)
        assert bool(uses) == (path.name == "test_legacy_key.py"), f"{path.name}: legacy-key fixtures/profile outside test_legacy_key.py"
    assert 'pytestmark = profiles("legacy-key")' in (ACCEPTANCE / "scenarios" / "test_legacy_key.py").read_text()


# ── the env -> ctx handoff and the scrubber, end to end ──────────


def test_handoff_returns_and_registers_the_exported_value_only_for_declared_variables():
    p = profile.load(ACCEPTANCE / "profiles" / "legacy-key.env")
    assert pytest_plugin.handoff(p, {"NS_LEGACY_KEY": KEY, "NS_OTHER": "zz" * 8}) == {"NS_LEGACY_KEY": KEY}
    assert KEY not in tokens.scrub(f"x {KEY} y")  # registered
    assert pytest_plugin.handoff(p, {}) == {}


def test_every_encoded_form_of_a_registered_secret_is_redacted():
    import base64, urllib.parse

    secret = "Ab/+=zq9" + "x" * 24
    tokens.register_secret(secret)
    for form in (secret, urllib.parse.quote(secret, safe=""), urllib.parse.quote_plus(secret),
                 base64.b64encode(secret.encode()).decode(), base64.urlsafe_b64encode(secret.encode()).decode().rstrip("=")):
        assert form not in tokens.scrub(f"prefix {form} suffix"), form


class _Stack(BaseHTTPRequestHandler):
    """Answers only what the REST identity scenario asks; reports the identity named in ``who``."""

    who = "acc-legacy"

    def log_message(self, *a):
        pass

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"user_id": type(self).who})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._send({"results": []})


def _run_scenario(tmp_path, who):
    handler = type("H", (_Stack,), {"who": who})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = tmp_path / "res.json"
    env = {**os.environ, "PYTHONPATH": str(ACCEPTANCE), "PYTHONDONTWRITEBYTECODE": "1", "NS_TOKEN_SECRET": "selftest-secret", "NS_LEGACY_KEY": KEY}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-c", str(ACCEPTANCE / "pytest.ini"), "--rootdir", str(ACCEPTANCE),
             str(ACCEPTANCE / "scenarios" / "test_legacy_key.py"), "-p", "nsaccept.pytest_plugin", "-p", "no:cacheprovider", "-q",
             "-k", "exactly_the_configured_identity_on_rest", "--ns-base-url", f"http://127.0.0.1:{srv.server_address[1]}",
             "--ns-profile", "legacy-key", "--ns-out", str(out)],
            env=env, capture_output=True, text=True,
        )
    finally:
        srv.shutdown()
        srv.server_close()
    return proc, json.loads(out.read_text())


def test_the_exported_key_reaches_the_legacy_fixture_end_to_end(tmp_path):
    """Fails if the plugin stops handing NS_LEGACY_KEY to the ctx: the fixture would error instead of the scenario passing."""
    proc, res = _run_scenario(tmp_path, "acc-legacy")
    assert [s["result"] for s in res["scenarios"]] == ["pass"], (proc.stdout[-600:], res["scenarios"])


def test_the_exported_key_never_reaches_the_results_even_when_a_message_would_carry_it(tmp_path):
    """The stub answers with the key as the identity, so the assertion message names it: only the registered scrubber keeps it out."""
    proc, res = _run_scenario(tmp_path, KEY)
    (row,) = res["scenarios"]
    assert row["result"] == "fail" and "<redacted-secret>" in row["detail"], row
    assert KEY not in json.dumps(res)
