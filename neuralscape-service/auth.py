"""Bearer token authentication middleware for Neuralscape API.

When NEURALSCAPE_API_KEY is set, all endpoints except /health require
a valid Authorization: Bearer <token> header. When unset (local dev),
auth is disabled and all requests pass through.
"""

import hmac
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings

logger = logging.getLogger(__name__)

# Paths that never require authentication
PUBLIC_PATHS = {"/health", "/api/v1/health"}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Auth disabled when no API key is configured (local dev)
        if not settings.neuralscape_api_key:
            return await call_next(request)

        # Health checks are always public
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Extract and validate Bearer token
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
            )

        token = auth_header[7:]  # Strip "Bearer " prefix
        if not hmac.compare_digest(token, settings.neuralscape_api_key):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"},
            )

        return await call_next(request)
