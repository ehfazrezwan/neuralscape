#!/usr/bin/env python3
"""Manual end-to-end OAuth login test for Neuralscape (any AUTH_PROVIDER).

Drives the full MCP-style OAuth flow against a *running* service exactly the way
Claude Cowork / Claude Code would:

  Dynamic Client Registration → GET /oauth/authorize → (you sign in in the
  browser) → the service redirects the auth code to a local loopback URL this
  script is listening on → POST /oauth/token → an authenticated /v1 call.

It prints the resolved ``user_id`` (decoded from the access token payload — no
server secret needed) so you can confirm the email→user_id mapping.

Usage (service must be running):
    uv run python scripts/oauth_login_test.py --base http://localhost:8199

Stdlib only. The browser step is yours; everything else is automated.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.server
import json
import secrets
import sys
import time
import urllib.parse
import urllib.request
import webbrowser


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# A browser-like UA: a tunnel fronted by Cloudflare 403s the default
# "Python-urllib/x" agent as a bot, which would break every call below.
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 neuralscape-oauth-test"

# Bypass any http(s)_proxy/ALL_PROXY env vars: a corporate proxy (e.g. Umbrella)
# would otherwise route even 127.0.0.1 through itself and stall. We hit a known
# host directly.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _open(req):
    return _OPENER.open(req, timeout=_HTTP_TIMEOUT)


_HTTP_TIMEOUT = 20  # fail fast instead of hanging on a tunnel stall


def _post_json(url: str, obj: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json", "User-Agent": _UA}, method="POST",
    )
    with _open(req) as r:
        return r.status, json.loads(r.read())


def _post_form(url: str, data: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=urllib.parse.urlencode(data).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": _UA},
        method="POST",
    )
    with _open(req) as r:
        return r.status, json.loads(r.read())


def _get_auth(url: str, token: str) -> tuple[int, object]:
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "User-Agent": _UA})
    with _open(req) as r:
        return r.status, json.loads(r.read())


_captured: dict[str, str] = {}


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return
        _captured.update(dict(urllib.parse.parse_qsl(parsed.query)))
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h2>Login captured. Close this tab and return to the terminal.</h2>")

    def log_message(self, *a):  # silence
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8199", help="Neuralscape base URL")
    ap.add_argument("--port", type=int, default=8765, help="local loopback callback port")
    ap.add_argument("--timeout", type=int, default=300, help="seconds to wait for sign-in")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    # Use 127.0.0.1 (not "localhost") everywhere: "localhost" resolves to ::1
    # first, but Docker publishes the port on IPv4 only — an IPv6 attempt hangs.
    redirect = f"http://127.0.0.1:{args.port}/callback"

    # Ensure prints appear immediately (stdout can be block-buffered under some
    # runners, which would hide the authorize URL while we wait for the browser).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # 1) Dynamic Client Registration (what the MCP client does on first connect)
    print(f"→ registering OAuth client at {base} …", flush=True)
    _, reg = _post_json(f"{base}/oauth/register", {"redirect_uris": [redirect]})
    client_id = reg["client_id"]
    print("✓ registered client", flush=True)

    # 2) PKCE
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    state = secrets.token_urlsafe(8)

    # 3) authorize URL
    authz = f"{base}/oauth/authorize?" + urllib.parse.urlencode({
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "resource": f"{base}/mcp",
    })

    # 4) capture server + open browser
    srv = http.server.HTTPServer(("127.0.0.1", args.port), _Handler)
    srv.timeout = 2
    print("\nOpen this URL in your browser and sign in:\n\n  " + authz + "\n", flush=True)
    print("Waiting for sign-in (Ctrl-C to cancel)…", flush=True)
    try:
        webbrowser.open(authz)
    except Exception:
        pass

    deadline = time.time() + args.timeout
    while "code" not in _captured and "error" not in _captured and time.time() < deadline:
        srv.handle_request()

    if "error" in _captured:
        print(f"✗ authorization error: {_captured.get('error')} — {_captured}")
        return 1
    if "code" not in _captured:
        print("✗ timed out waiting for sign-in")
        return 1
    if _captured.get("state") != state:
        print("✗ STATE MISMATCH — possible CSRF; aborting")
        return 1

    # 5) exchange the code for tokens (PKCE)
    _, tok = _post_form(f"{base}/oauth/token", {
        "grant_type": "authorization_code",
        "code": _captured["code"],
        "code_verifier": verifier,
        "redirect_uri": redirect,
        "client_id": client_id,
    })
    access = tok["access_token"]
    # NB: Neuralscape access tokens are a 2-segment HMAC token "payload.sig"
    # (NOT a 3-segment JWT), so the payload is segment [0], not [1].
    payload = json.loads(_b64url_decode(access.split(".")[0]))
    print("\n✅ access token obtained")
    print(f"   resolved user_id : {payload.get('user_id')}")
    print(f"   expires_in       : {tok.get('expires_in')}s")
    print(f"   refresh_token    : {'present' if tok.get('refresh_token') else 'missing'}")

    # 6) prove the token authenticates a real request
    status, projects = _get_auth(f"{base}/v1/projects", access)
    print(f"\n✓ GET /v1/projects [{status}] → {projects}")
    print("\n🎉 end-to-end OAuth login works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
