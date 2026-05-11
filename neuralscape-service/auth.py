"""Bearer token authentication for Neuralscape API.

Supports two token formats simultaneously:

1. **Per-user HMAC tokens** (multi-user model): two segments
   `base64url({user_id, exp})` + `.` + `hmac_sha256(secret, payload)`.
   When `NEURALSCAPE_USER_TOKEN_SECRET` is set, tokens of this shape are
   accepted; the verified user_id is attached to ``request.state.user_id``
   so routes pull identity from the token, not the request body.

2. **Legacy shared API key** (single-user / pre-multi-user installs):
   one segment, opaque string. Validated against ``NEURALSCAPE_API_KEY``.
   When this path matches, ``request.state.user_id`` is **not** set
   (routes must fall back to body ``user_id``), and a deprecation
   warning header is added to the response.

When neither secret is configured, auth is disabled (local dev).

Public paths (`/health`) always bypass auth.
"""

import hmac
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings
from tokens import issue_user_token, verify_user_token  # noqa: F401 (re-export)

logger = logging.getLogger(__name__)

# Paths that never require authentication
PUBLIC_PATHS = {"/health", "/api/v1/health"}

# Header set on responses when a caller used a legacy shared API key, to
# nudge them toward per-user tokens. Non-fatal.
DEPRECATION_HEADER = "X-Neuralscape-Deprecation"
DEPRECATION_MSG = "shared-api-key auth is deprecated; issue per-user tokens via scripts/issue_user_token.py"


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Bearer token middleware supporting per-user HMAC + legacy shared keys."""

    async def dispatch(self, request: Request, call_next):
        # Auth disabled when neither secret is set (local dev convenience)
        token_secret = settings.neuralscape_user_token_secret
        legacy_key = settings.neuralscape_api_key
        if not token_secret and not legacy_key:
            return await call_next(request)

        # Health checks are always public
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
            )
        token = auth_header[7:]  # strip "Bearer "

        # Try the per-user HMAC token path first (preferred). The token
        # carries the user_id we attach to request.state for downstream
        # routes to use authoritatively.
        if token_secret and "." in token:
            payload = verify_user_token(token, token_secret)
            if payload is None:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or expired token"},
                )
            request.state.user_id = payload["user_id"]
            request.state.auth_mode = "user_token"
            return await call_next(request)

        # Fall back to the legacy shared API key. user_id is NOT set on
        # request.state — routes must read it from the request body
        # (trust-based, pre-multi-user behavior).
        if legacy_key and hmac.compare_digest(token, legacy_key):
            request.state.auth_mode = "legacy_shared"
            response = await call_next(request)
            response.headers[DEPRECATION_HEADER] = DEPRECATION_MSG
            return response

        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid API key"},
        )
