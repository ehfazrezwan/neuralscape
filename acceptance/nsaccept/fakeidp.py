"""A fake Supabase auth service, harness side: a per-run RSA key, its JWKS, and signed session tokens.

The stack verifies a Supabase session JWT against ``{SUPABASE_URL}/auth/v1/.well-known/jwks.json``.
For the acceptance run that URL points at the ``nsc-acc-supabase`` sidecar (see ``sidecar.Sink``), which
serves a JWKS holding only the PUBLIC half of a key this module generates in memory for the run. The
private half never leaves the harness process and is never written anywhere: the harness signs the
session tokens itself, exactly the part a real Supabase project would do after a browser sign-in.

Pure standard library (no JWT or crypto dependency): textbook RSA with PKCS#1 v1.5 / SHA-256, which is
what RS256 is. This is test tooling for a throwaway stack, not a crypto library.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from html import unescape

#: The sidecar's name and where it serves the keys (the stack's own path for a Supabase project).
FAKE_SUPABASE_NAME = "nsc-acc-supabase"
JWKS_PATH = "/auth/v1/.well-known/jwks.json"

#: ASN.1 DigestInfo prefix for SHA-256 (RFC 8017 sec 9.2 note 1).
_SHA256_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")
_E = 65537
_SMALL_PRIMES = [p for p in range(3, 2000, 2) if all(p % q for q in range(3, int(p**0.5) + 1, 2))]


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _int_b64(n: int) -> str:
    return b64url(n.to_bytes((n.bit_length() + 7) // 8, "big"))


# ── RSA, just enough for RS256 ───────────────────────────────────


def _probable_prime(n: int, rounds: int = 24) -> bool:
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _prime(bits: int) -> int:
    while True:
        # top two bits set so p*q has exactly 2*bits bits; low bit set so it is odd
        cand = secrets.randbits(bits) | (1 << (bits - 1)) | (1 << (bits - 2)) | 1
        if _probable_prime(cand):
            return cand


def _emsa_pkcs1_v15(message: bytes, k: int) -> bytes:
    digest_info = _SHA256_PREFIX + hashlib.sha256(message).digest()
    return b"\x00\x01" + b"\xff" * (k - len(digest_info) - 3) + b"\x00" + digest_info


@dataclass
class RsaKey:
    """One RSA key pair with a ``kid``. ``d`` is the private exponent; keep the object in memory only."""

    n: int
    d: int = field(repr=False)
    kid: str
    e: int = _E

    @classmethod
    def generate(cls, kid: str, bits: int = 2048) -> "RsaKey":
        while True:
            p, q = _prime(bits // 2), _prime(bits // 2)
            if p == q:
                continue
            phi = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)
            if math.gcd(_E, phi) != 1:
                continue
            return cls(n=p * q, d=pow(_E, -1, phi), kid=kid)

    @property
    def size(self) -> int:
        return (self.n.bit_length() + 7) // 8

    def jwk(self) -> dict:
        """The PUBLIC key as a JWK (what a Supabase project publishes)."""
        return {"kty": "RSA", "use": "sig", "alg": "RS256", "kid": self.kid, "n": _int_b64(self.n), "e": _int_b64(self.e)}

    def jwks(self) -> dict:
        return {"keys": [self.jwk()]}

    def sign_rs256(self, signing_input: bytes) -> bytes:
        em = int.from_bytes(_emsa_pkcs1_v15(signing_input, self.size), "big")
        return pow(em, self.d, self.n).to_bytes(self.size, "big")


def verify_rs256(signing_input: bytes, signature: bytes, n: int, e: int = _E) -> bool:
    """Public-key check of an RS256 signature (used by the self-tests; the stack uses PyJWT)."""
    k = (n.bit_length() + 7) // 8
    if len(signature) != k:
        return False
    em = pow(int.from_bytes(signature, "big"), e, n).to_bytes(k, "big")
    return hmac.compare_digest(em, _emsa_pkcs1_v15(signing_input, k))


# ── JWTs ─────────────────────────────────────────────────────────


def _json_b64(obj: dict) -> str:
    return b64url(json.dumps(obj, separators=(",", ":")).encode())


def sign_jwt(key: RsaKey, claims: dict, *, kid: str | None = None) -> str:
    header = {"alg": "RS256", "typ": "JWT", "kid": kid or key.kid}
    signing_input = f"{_json_b64(header)}.{_json_b64(claims)}"
    return f"{signing_input}.{b64url(key.sign_rs256(signing_input.encode('ascii')))}"


def sign_hs256(secret: bytes, claims: dict, *, kid: str | None = None) -> str:
    """A symmetric token (the algorithm-confusion probe): the verifier must not accept it."""
    header = {"alg": "HS256", "typ": "JWT", **({"kid": kid} if kid else {})}
    signing_input = f"{_json_b64(header)}.{_json_b64(claims)}"
    return f"{signing_input}.{b64url(hmac.new(secret, signing_input.encode('ascii'), hashlib.sha256).digest())}"


def decode_unverified(token: str) -> dict:
    """The payload of a JWT without checking anything (for the self-tests and assertions)."""
    return json.loads(b64url_decode(token.split(".")[1]))


def supabase_claims(
    supabase_url: str,
    sub: str,
    email: str,
    *,
    verified: bool = True,
    user_metadata_verified: bool | None = None,
    aud: str = "authenticated",
    iss: str | None = None,
    ttl: int = 3600,
    now: int | None = None,
    with_exp: bool = True,
) -> dict:
    """Claims shaped like a Supabase Auth (GoTrue) access token.

    ``verified`` sets the top-level ``email_verified`` claim, the one the stack's verifier trusts.
    ``user_metadata_verified`` sets the user-writable ``user_metadata.email_verified``, which the stack
    must NOT trust (a user can edit it).
    """
    now = int(time.time()) if now is None else now
    claims: dict = {
        "iss": iss if iss is not None else f"{supabase_url.rstrip('/')}/auth/v1",
        "aud": aud,
        "sub": sub,
        "iat": now,
        "role": "authenticated",
        "aal": "aal1",
        "session_id": str(uuid.uuid4()),
        "email": email,
        "phone": "",
        "app_metadata": {"provider": "google", "providers": ["google"]},
        "user_metadata": {"email": email, **({} if user_metadata_verified is None else {"email_verified": user_metadata_verified})},
        "is_anonymous": False,
    }
    if verified:
        claims["email_verified"] = True
    if with_exp:
        claims["exp"] = now + ttl
    return claims


# ── slugs and pages (what a browser would read) ──────────────────

_DISALLOWED = re.compile(r"[^a-zA-Z0-9_.\-]+")
_EDGE = re.compile(r"^[._\-]+|[._\-]+$")


def expected_slug(email: str) -> str:
    """The user id the documented email slug rule gives (AUTH.md: ``alice.smith@example.com`` -> ``alice.smith-example.com``).

    Written out here on purpose rather than imported: the harness must not import the service.
    """
    slug = _EDGE.sub("", _DISALLOWED.sub("-", email.strip().lower()))
    return _EDGE.sub("", slug[:100]) if len(slug) > 100 else slug


def consent_login_state(page: str, public_url: str) -> str:
    """The signed ``state`` a Supabase consent page puts in its ``redirectTo`` (what the browser carries to the callback)."""
    m = re.search(re.escape(public_url.rstrip("/")) + r"/oauth/supabase/callback\?state=([A-Za-z0-9_.\-=%]+)", unescape(page))
    if not m:
        raise AssertionError("the consent page carries no /oauth/supabase/callback?state=... redirect")
    return m.group(1)


def link_page_fields(page: str) -> dict:
    """``ticket`` and the suggested id from the first-login link page."""
    t = re.search(r'name="ticket" value="([^"]+)"', page)
    s = re.search(r"Continue as a new account \(<code>([^<]+)</code>\)", page)
    if not t or not s:
        raise AssertionError("not the first-login link page (no ticket / suggested account)")
    return {"ticket": unescape(t.group(1)), "suggested": unescape(s.group(1))}
