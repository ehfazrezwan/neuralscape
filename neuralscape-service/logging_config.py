"""Structured logging configuration for neuralscape-service.

Configures structlog for JSON output in production with key event correlation.
Import and call configure_logging() once at startup (in main.py lifespan).
"""

import logging
import re
import sys
import uuid

import structlog

from config import settings

REQUEST_ID_HEADER = "X-Request-ID"
# An incoming id is echoed into logs and a response header, so accept only a
# short token of safe characters; anything else is replaced by a generated id.
_SANE_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")

_ARQ_LOGGERS = ("arq", "arq.worker", "arq.jobs", "arq.connections", "arq.cron")


def _request_id_from(scope) -> str:
    wanted = REQUEST_ID_HEADER.lower().encode("latin-1")
    for name, value in scope.get("headers") or ():
        if name.lower() == wanted:
            candidate = value.decode("latin-1")
            if _SANE_REQUEST_ID.fullmatch(candidate):
                return candidate
            break
    return uuid.uuid4().hex


class RequestIdMiddleware:
    """Bind a ``request_id`` to the log context for the life of each HTTP request.

    Pure ASGI (not BaseHTTPMiddleware) so the contextvar is visible to the
    endpoint and to streaming/mounted apps. Register it OUTERMOST (added last)
    so responses produced by inner middleware, such as auth 401s, carry the id.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from(scope)
        # Kept on the scope too: the 500 handler runs outside this middleware,
        # after the contextvar is gone, and reads the id from here.
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)

        async def send_with_id(message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"
                ]
                headers.append((REQUEST_ID_HEADER.lower().encode("latin-1"), request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")


def request_id_of(request) -> str | None:
    """The id RequestIdMiddleware assigned to this request, if it ran."""
    return (request.scope.get("state") or {}).get("request_id")


def configure_logging(*, worker: bool = False) -> None:
    """Configure structured logging with structlog.

    Uses JSON rendering for production (LOG_FORMAT=json or default)
    and console rendering for development (LOG_FORMAT=console).
    LOG_LEVEL / LOG_FORMAT come from ``config.settings``, so the environment
    and the service ``.env`` both count (the environment wins).

    ``worker=True`` (ARQ processes) also drops ARQ's own log handlers so its
    lines flow through the same renderer once, with the bound job id.
    """
    log_level = settings.log_level.strip().upper()
    log_format = settings.log_format.strip().lower()

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "console":
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # JSON has no traceback rendering of its own; the console renderer does.
    exc_processors: list[structlog.types.Processor] = (
        [] if log_format == "console" else [structlog.processors.format_exc_info]
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # Plain ``logging.getLogger`` records (most of the service) get the
        # level, logger name, timestamp and the bound request/job ids too.
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *exc_processors,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    if worker:
        for name in _ARQ_LOGGERS:
            logging.getLogger(name).handlers.clear()

    # Reduce noise from third-party loggers
    for noisy in ("httpcore", "httpx", "neo4j", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # uvicorn access logs (request method/path/status) are silenced by default
    # to cut noise. But when the OAuth connector is enabled they're the only
    # window into the Claude Cowork handshake — the 307/406/200 on /mcp that
    # broke the connector and were invisible in `docker logs` (we needed a
    # tcpdump sidecar to see them). Surface them at INFO in that case so
    # connector issues are debuggable without packet capture.
    #
    # Read from settings, not os.environ: config is loaded via pydantic-settings
    # (incl. the .env file), which does NOT populate os.environ. Checking
    # os.environ here would miss a .env-only deployment and keep the logs
    # silenced exactly when they're needed.
    oauth_enabled = bool(
        settings.neuralscape_public_url.strip()
        and settings.neuralscape_user_token_secret.strip()
    )
    logging.getLogger("uvicorn.access").setLevel(
        logging.INFO if oauth_enabled else logging.WARNING
    )
