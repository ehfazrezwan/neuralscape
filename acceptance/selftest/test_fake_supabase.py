"""Self-tests for the fake Supabase auth service (no stack, no docker, no network beyond loopback).

Covers: the stdlib RS256 signer (checked against its own public verifier and, when PyJWT is importable,
against PyJWT), the JWKS the sidecar would serve, the Supabase-shaped claims, the sidecar's safety (the
``nsc-acc-supabase`` name goes through the same rules as the sink), the profile, and the scenario's browser
flow run against a small stub of the stack's OAuth endpoints (cookie jar, link page, PKCE exchange, and the
login-binding refusal), so the harness plumbing is exercised without a stack.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import threading
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import pytest

from nsaccept import fakeidp, guard, profile, sidecar
from nsaccept.client import NSClient

ACCEPTANCE = Path(__file__).resolve().parents[1]


def _load_scenario():
    spec = importlib.util.spec_from_file_location("oauth_supabase_login_scenario", ACCEPTANCE / "scenarios" / "test_oauth_supabase_login.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


scn = _load_scenario()


# ── the signer ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def key():
    return fakeidp.RsaKey.generate(kid="k-test")


def _parts(token: str):
    head, payload, sig = token.split(".")
    return f"{head}.{payload}".encode(), fakeidp.b64url_decode(sig)


def test_a_generated_key_is_2048_bits_and_signs_verifiable_rs256_tokens(key):
    assert key.n.bit_length() == 2048 and key.e == 65537 and key.size == 256
    token = fakeidp.sign_jwt(key, fakeidp.supabase_claims("http://x:8080", "sub-1", "a@b.example"))
    signing_input, sig = _parts(token)
    assert fakeidp.verify_rs256(signing_input, sig, key.n)
    assert not fakeidp.verify_rs256(signing_input + b"x", sig, key.n)
    assert not fakeidp.verify_rs256(signing_input, sig[:-1] + bytes([sig[-1] ^ 1]), key.n)
    other = fakeidp.RsaKey.generate(kid="k-other")
    assert not fakeidp.verify_rs256(signing_input, sig, other.n)


def test_the_jwk_publishes_only_public_numbers_and_the_kid(key):
    jwks = key.jwks()
    (jwk,) = jwks["keys"]
    assert jwk["kid"] == "k-test" and jwk["kty"] == "RSA" and jwk["use"] == "sig" and jwk["alg"] == "RS256"
    assert set(jwk) == {"kty", "use", "alg", "kid", "n", "e"}, "the JWKS must carry no private member (d, p, q, ...)"
    assert int.from_bytes(fakeidp.b64url_decode(jwk["n"]), "big") == key.n
    assert int.from_bytes(fakeidp.b64url_decode(jwk["e"]), "big") == 65537
    assert str(key.d) not in json.dumps(jwks) and "d" not in jwk


def test_the_header_names_rs256_and_the_kid_and_a_forged_token_can_claim_another_kid(key):
    head = json.loads(fakeidp.b64url_decode(fakeidp.sign_jwt(key, {"a": 1}).split(".")[0]))
    assert head == {"alg": "RS256", "typ": "JWT", "kid": "k-test"}
    forged = fakeidp.sign_jwt(key, {"a": 1}, kid="published")
    assert json.loads(fakeidp.b64url_decode(forged.split(".")[0]))["kid"] == "published"
    hs = fakeidp.sign_hs256(b"s" * 32, {"a": 1}, kid="published")
    assert json.loads(fakeidp.b64url_decode(hs.split(".")[0])) == {"alg": "HS256", "typ": "JWT", "kid": "published"}


def test_the_signature_agrees_with_pyjwt_when_it_is_installed(key):
    jwt = pytest.importorskip("jwt")
    token = fakeidp.sign_jwt(key, fakeidp.supabase_claims("http://x:8080", "s", "a@b.example"))
    jwk = jwt.PyJWK.from_dict(key.jwk())
    claims = jwt.decode(token, jwk.key, algorithms=["RS256"], audience="authenticated", options={"require": ["exp"]})
    assert claims["email"] == "a@b.example"
    with pytest.raises(jwt.InvalidAudienceError):
        jwt.decode(fakeidp.sign_jwt(key, fakeidp.supabase_claims("http://x:8080", "s", "a@b.example", aud="anon")), jwk.key,
                   algorithms=["RS256"], audience="authenticated")


# ── the claims ───────────────────────────────────────────────────


def test_claims_are_supabase_shaped_and_each_knob_changes_one_thing():
    c = fakeidp.supabase_claims("http://nsc-acc-supabase:8080/", "sub-9", "p@q.example", now=1000)
    assert c["iss"] == "http://nsc-acc-supabase:8080/auth/v1" and c["aud"] == "authenticated" and c["role"] == "authenticated"
    assert c["sub"] == "sub-9" and c["email"] == "p@q.example" and c["exp"] == 4600 and c["email_verified"] is True
    assert "email_verified" not in c["user_metadata"]
    assert "email_verified" not in fakeidp.supabase_claims("http://x", "s", "e@f.g", verified=False)
    assert fakeidp.supabase_claims("http://x", "s", "e@f.g", verified=False, user_metadata_verified=True)["user_metadata"]["email_verified"] is True
    assert "exp" not in fakeidp.supabase_claims("http://x", "s", "e@f.g", with_exp=False)
    assert fakeidp.supabase_claims("http://x", "s", "e@f.g", now=100, ttl=-5)["exp"] == 95
    assert fakeidp.supabase_claims("http://x", "s", "e@f.g", aud="anon")["aud"] == "anon"


def test_the_expected_slug_follows_the_documented_rule():
    assert fakeidp.expected_slug("alice.smith@example.com") == "alice.smith-example.com"
    assert fakeidp.expected_slug("  Harbor+Crew@Acceptance-Placeholder.example ") == "harbor-crew-acceptance-placeholder.example"
    assert fakeidp.expected_slug("@@x@@") == "x"
    assert len(fakeidp.expected_slug("a" * 300 + "@b.example")) == 100


def test_page_readers_extract_the_state_and_the_link_ticket_and_refuse_other_pages():
    page = '<script>redirectTo: \'http://127.0.0.1:1/oauth/supabase/callback?state=abc.DEF-9_=\'</script>'
    assert fakeidp.consent_login_state(page.replace("'", "&#x27;"), "http://127.0.0.1:1/") == "abc.DEF-9_="
    with pytest.raises(AssertionError):
        fakeidp.consent_login_state("<p>no button</p>", "http://127.0.0.1:1")
    link = '<input type="hidden" name="ticket" value="t.k&amp;n"><code>x</code>Continue as a new account (<code>who-x.example</code>)'
    assert fakeidp.link_page_fields(link) == {"ticket": "t.k&n", "suggested": "who-x.example"}
    with pytest.raises(AssertionError):
        fakeidp.link_page_fields("<p>Sign-in failed</p>")


# ── the sidecar and the profile ──────────────────────────────────


class Recorder:
    def __init__(self, logs=""):
        self.calls, self.logs = [], logs

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        out = "nstest-x_default\n" if cmd[1:3] == ["network", "ls"] else (self.logs if cmd[1] == "logs" else "")
        return subprocess.CompletedProcess(cmd, 0, out, "")


@pytest.fixture(autouse=True)
def no_live_sidecars():
    sidecar._LIVE.clear()
    yield
    sidecar._LIVE.clear()


def _fake(rec, key, project="nstest-x"):
    return sidecar.Sink(project, {fakeidp.JWKS_PATH: key.jwks()}, name=fakeidp.FAKE_SUPABASE_NAME, run=rec, clock=time.monotonic, sleep=lambda s: None)


def test_the_fake_supabase_is_a_capped_nsc_acc_sidecar_on_the_stack_network_that_is_always_removed(key):
    rec = Recorder(logs=json.dumps({"ready": True, "port": 8080}) + "\n")
    s = _fake(rec, key).start()
    run = next(c for c in rec.calls if c[1] == "run")
    for flag, value in (("--name", "nsc-acc-supabase"), ("--network", "nstest-x_default"), ("--memory", "64m"), ("--cpus", "0.25"), ("--read-only", None)):
        assert flag in run and (value is None or run[run.index(flag) + 1] == value), flag
    assert not any(a in run for a in ("-p", "--publish", "-v", "--volume", "--privileged"))
    env = next(a for a in run if a.startswith("NSC_FIXTURES="))
    assert json.loads(env.split("=", 1)[1]) == {fakeidp.JWKS_PATH: key.jwks()}
    assert str(key.d) not in " ".join(run), "the private exponent reached the docker command line"
    s.stop()
    assert rec.calls[-1] == ["docker", "rm", "-f", "nsc-acc-supabase"] and sidecar._LIVE == set()


def test_the_fake_supabase_refuses_production_and_foreign_names_and_a_third_sidecar(key):
    for project in ("neuralscape", "NeuralScape", "something-else"):
        with pytest.raises(guard.GuardError):
            _fake(Recorder(), key, project=project)
    with pytest.raises(sidecar.SidecarError):
        sidecar.Sink("nstest-x", {}, name="supabase")
    sidecar._LIVE.update({"nsc-acc-sink", "nsc-acc-other"})
    with pytest.raises(sidecar.SidecarError):
        _fake(Recorder(logs=json.dumps({"ready": True}) + "\n"), key).start()


def test_the_sidecar_serves_the_jwks_for_real_and_logs_who_asked(key):
    import os
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "NSC_PORT": str(port), "NSC_FIXTURES": json.dumps({fakeidp.JWKS_PATH: key.jwks()}), "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.Popen([sys.executable, "-c", sidecar.SERVER_SCRIPT], env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert json.loads(proc.stdout.readline())["ready"] is True
        got = httpx.get(f"http://127.0.0.1:{port}{fakeidp.JWKS_PATH}")
        assert got.status_code == 200 and got.json() == key.jwks()
        assert json.loads(proc.stdout.readline())["path"] == fakeidp.JWKS_PATH
    finally:
        proc.kill()
        proc.wait()


def test_the_profile_points_the_stack_at_the_sidecar_and_declares_what_the_gate_needs():
    p = profile.load(ACCEPTANCE / "profiles" / "oauth-supabase.env")
    assert p.flags["SUPABASE_URL"] == f"http://{fakeidp.FAKE_SUPABASE_NAME}:{sidecar.PORT}"
    assert p.headers["requires"] == ["compose"] and p.headers["sidecars"] == ["nsc-acc-supabase"]
    assert p.flags["AUTH_PROVIDER"] == "supabase" and p.placeholders["SUPABASE_ANON_KEY"] == "hex32"
    assert p.flags["AUTH_ALLOWED_DOMAINS"] == scn.DOMAIN and p.flags["AUTH_EMAIL_ALLOWLIST"] == scn.GUEST_ADDRESS
    assert p.flags["AUTH_IDENTITY_MAP"] == f"{scn.MAPPED_EMAIL}:{scn.MAPPED_USER}"
    assert "SUPABASE_JWT_SECRET" not in p.flags, "the profile must use the JWKS path, with no shared secret"
    assert all(v in ("", "<redacted>") or not re.search(r"[0-9a-f]{32,}", v) for v in p.flags.values()), "a secret-looking literal in the profile"


# ── the scenario's browser flow, against a stub of the stack's OAuth endpoints ──


class StackStub:
    """Just enough of /oauth/* to drive ``login()``: DCR, consent page, callback (+ optional login binding), link page, code, token."""

    def __init__(self, bind: bool):
        self.bind, self.known, self.codes, self.states, self.nonces_used = bind, set(), {}, {}, set()
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, body="", headers=()):
                raw = body.encode() if isinstance(body, str) else body
                self.send_response(status)
                for k, v in headers:
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _form(self):
                n = int(self.headers.get("Content-Length") or 0)
                return {k: v[0] for k, v in parse_qs(self.rfile.read(n).decode(), keep_blank_values=True).items()}

            def do_POST(self):  # noqa: N802
                path = self.path.split("?")[0]
                if path == "/oauth/register":
                    return self._send(201, json.dumps({"client_id": "cid"}), [("Content-Type", "application/json")])
                form = self._form() if path != "/oauth/register" else {}
                if path == "/oauth/supabase/callback":
                    return stub.callback(self, form)
                if path == "/oauth/link":
                    return self._send(*stub.redirect(form["ticket"].split("|")[0], form["ticket"].split("|")[1]))
                if path == "/oauth/token":
                    ctx = stub.codes.pop(form["code"], None)
                    if ctx is None or hashlib.sha256(form["code_verifier"].encode()).digest() != ctx["challenge"]:
                        return self._send(400, json.dumps({"error": "invalid_grant"}), [("Content-Type", "application/json")])
                    payload = base64.urlsafe_b64encode(json.dumps({"user_id": ctx["user"]}).encode()).rstrip(b"=").decode()
                    return self._send(200, json.dumps({"token_type": "Bearer", "access_token": payload + ".sig", "refresh_token": "r"}),
                                      [("Content-Type", "application/json")])
                self._send(404)

            def do_GET(self):  # noqa: N802
                u = urlparse(self.path)
                if u.path == "/oauth/authorize":
                    q = {k: v[0] for k, v in parse_qs(u.query).items()}
                    nonce = f"n{len(stub.states) + 1}"
                    state = f"{nonce}.sig"
                    stub.states[state] = {"app_state": q["state"], "challenge": fakeidp.b64url_decode(q["code_challenge"]), "nonce": nonce}
                    cb = f"http://127.0.0.1:{stub.port}/oauth/supabase/callback?state={state}"
                    cookie = [("Set-Cookie", f"ns_login_bind={nonce}; Path=/oauth; HttpOnly; SameSite=Lax; Max-Age=600")] if stub.bind else []
                    return self._send(200, f"<script>redirectTo: '{escape(cb)}'</script>", [("Content-Type", "text/html"), *cookie])
                self._send(404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def redirect(self, state: str, user: str):
        st = self.states[state]
        code = f"c{len(self.codes) + 1}"
        self.codes[code] = {"challenge": st["challenge"], "user": user}
        self.known.add(user)
        loc = f"{scn.REDIRECT}?{urlencode({'code': code, 'state': st['app_state']})}"
        return 303, "", [("Location", loc)]

    def callback(self, h, form):
        st = self.states.get(form.get("state", ""))
        if st is None:
            return h._send(400, "<p>Login session expired or invalid.</p>")
        if self.bind:
            cookies = dict(c.strip().split("=", 1) for c in (h.headers.get("Cookie") or "").split(";") if "=" in c)
            if cookies.get("ns_login_bind") != st["nonce"] or st["nonce"] in self.nonces_used:
                return h._send(400, "<p>This sign-in was not started in this browser, or it has expired.</p>")
            self.nonces_used.add(st["nonce"])
        email = json.loads(fakeidp.b64url_decode(form["access_token"].split(".")[1]))["email"]
        user = scn.MAPPED_USER if email == scn.MAPPED_EMAIL else fakeidp.expected_slug(email)
        if user in self.known:
            return h._send(*self.redirect(form["state"], user))
        if user == scn.MAPPED_USER:
            return h._send(*self.redirect(form["state"], user))
        page = (f'<form action="/oauth/link"><input type="hidden" name="ticket" value="{escape(form["state"])}|{escape(user)}">'
                f"Continue as a new account (<code>{escape(user)}</code>)</form>")
        return h._send(200, page, [("Content-Type", "text/html")])


class FakeIdp:
    def __init__(self, key):
        self.key, self.url = key, "http://nsc-acc-supabase:8080"

    def token(self, sub, email, **kw):
        return fakeidp.sign_jwt(self.key, fakeidp.supabase_claims(self.url, sub, email, **kw))


@pytest.fixture
def stack(request):
    s = StackStub(bind=getattr(request, "param", False))
    yield s
    s.stop()


def _ctx(stack):
    return SimpleNamespace(base_url=stack.url, profile_flags={"NEURALSCAPE_PUBLIC_URL": stack.url})


@pytest.fixture
def anon_client(stack):
    c = NSClient(stack.url)
    yield c
    c.close()


def test_login_walks_consent_callback_link_page_and_pkce_exchange_for_a_new_person(stack, anon_client, key):
    d = scn.login(_ctx(stack), anon_client, FakeIdp(key), "harbor.r1@acceptance-placeholder.example", "sb-1")
    assert d.saw_link_page and d.suggested == "harbor.r1-acceptance-placeholder.example" and d.user_id == d.suggested
    again = scn.login(_ctx(stack), anon_client, FakeIdp(key), "harbor.r1@acceptance-placeholder.example", "sb-1")
    assert not again.saw_link_page and again.user_id == d.user_id


def test_login_for_a_mapped_email_goes_straight_to_the_code(stack, anon_client, key):
    d = scn.login(_ctx(stack), anon_client, FakeIdp(key), scn.MAPPED_EMAIL, "sb-2")
    assert not d.saw_link_page and d.user_id == scn.MAPPED_USER


@pytest.mark.parametrize("stack", [True], indirect=True)
def test_the_browser_that_started_the_login_completes_it_under_binding_and_another_browser_is_refused(stack, anon_client, key):
    ctx, idp = _ctx(stack), FakeIdp(key)
    assert scn.login(ctx, anon_client, idp, scn.MAPPED_EMAIL, "sb-3").user_id == scn.MAPPED_USER  # the cookie jar carries the binding
    attacker, victim = scn.Browser(stack.url), scn.Browser(stack.url)
    try:
        st = scn.start_login(ctx, anon_client, attacker)
        assert scn._binding_cookies(attacker) and not scn._binding_cookies(victim)
        scn.check_refused(victim.callback(st.login_state, idp.token("sb-4", scn.MAPPED_EMAIL)), status=(400, 403))
        done = attacker.callback(st.login_state, idp.token("sb-4", scn.MAPPED_EMAIL))
        scn.code_from(done, st)
        scn.check_refused(attacker.callback(st.login_state, idp.token("sb-4", scn.MAPPED_EMAIL)), status=(400, 403))  # replay
    finally:
        attacker.close()
        victim.close()


def test_without_binding_the_stub_accepts_a_foreign_browser_which_the_expected_red_scenario_would_flag(stack, anon_client, key):
    ctx, idp = _ctx(stack), FakeIdp(key)
    attacker, victim = scn.Browser(stack.url), scn.Browser(stack.url)
    try:
        st = scn.start_login(ctx, anon_client, attacker)
        resp = victim.callback(st.login_state, idp.token("sb-5", scn.MAPPED_EMAIL))
        assert resp.status_code == 303  # exactly what test_a_callback_from_a_browser_that_did_not_start_the_login_is_refused trips on
        with pytest.raises(AssertionError):
            scn.check_refused(resp, status=(400, 403))
    finally:
        attacker.close()
        victim.close()


def test_pkce_binds_the_code_to_the_verifier(stack, anon_client, key):
    ctx, idp = _ctx(stack), FakeIdp(key)
    b = scn.Browser(stack.url)
    try:
        st = scn.start_login(ctx, anon_client, b)
        code = scn.code_from(b.callback(st.login_state, idp.token("sb-6", scn.MAPPED_EMAIL)), st)
    finally:
        b.close()
    assert scn.exchange(anon_client, st, code, verifier="x" * 43).status_code == 400
    assert scn.exchange(anon_client, st, code).status_code == 400, "a code whose first redemption was refused for a wrong verifier is gone (the stub consumes it)"


# ── the scenario's own checks can fail ────────────────────────────


def _resp(status, text="", headers=None):
    return httpx.Response(status, text=text, headers=headers or {})


def test_check_refused_accepts_a_refusal_and_rejects_redirects_codes_and_wrong_pages():
    scn.check_refused(_resp(403, "<p>This account is not authorized.</p>"), status=403, needle="not authorized")
    scn.check_refused(_resp(400, "<p>This sign-in was not started in this browser</p>"), status=400, needle=scn.STATE_REFUSED)
    for bad in (
        _resp(303, "", {"location": "http://c/cb?code=1"}),
        _resp(200, "<p>fine</p>"),
        _resp(403, "<p>other</p>"),
        _resp(403, "<p>This account is not authorized. http://c/cb?code=abc</p>"),
        _resp(401, "<p>Could not verify</p>"),
    ):
        with pytest.raises(AssertionError):
            scn.check_refused(bad, status=403, needle="not authorized")
    with pytest.raises(AssertionError):
        scn.check_refused(_resp(403, "<p>x</p>", {"location": "http://c"}), status=403)


def test_check_fetched_fails_when_the_stack_never_asked(key):
    class Sink:
        def __init__(self, paths):
            self.paths = paths

        def requests(self, method=None, path=None):
            return [{"method": "GET", "path": p} for p in self.paths if path in (None, p)]

    scn.check_fetched(Sink([fakeidp.JWKS_PATH]), fakeidp.JWKS_PATH)
    with pytest.raises(AssertionError, match="control precondition"):
        scn.precondition(scn.check_fetched, Sink(["/other"]), fakeidp.JWKS_PATH)
    scn.precondition(scn.check_fetched, Sink([fakeidp.JWKS_PATH]), fakeidp.JWKS_PATH)
    with pytest.raises(AssertionError, match="login binding not deployed"):
        scn.require_binding_deployed(_resp(200, "x"))
    scn.require_binding_deployed(_resp(200, "x", {"set-cookie": "ns_login_bind=a; HttpOnly"}))
    with pytest.raises(AssertionError):
        scn.check_fetched(Sink(["/other"]), fakeidp.JWKS_PATH)


def test_token_user_reads_the_user_from_an_access_token():
    payload = base64.urlsafe_b64encode(json.dumps({"user_id": "u-1"}).encode()).rstrip(b"=").decode()
    assert scn.token_user(payload + ".sig") == "u-1"
