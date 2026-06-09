"""Structured logging configuration for neuralscape-service.

Configures structlog for JSON output in production with key event correlation.
Import and call configure_logging() once at startup (in main.py lifespan).
"""

import logging
import os
import sys

import structlog


def configure_logging() -> None:
    """Configure structured logging with structlog.

    Uses JSON rendering for production (LOG_FORMAT=json or default)
    and console rendering for development (LOG_FORMAT=console).
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("LOG_FORMAT", "json")

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

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Reduce noise from third-party loggers
    for noisy in ("httpcore", "httpx", "neo4j", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # uvicorn access logs (request method/path/status) are silenced by default
    # to cut noise. But when the OAuth connector is enabled they're the only
    # window into the Claude Cowork handshake — the 307/406/200 on /mcp that
    # broke the connector and were invisible in `docker logs` (we needed a
    # tcpdump sidecar to see them). Surface them at INFO in that case so
    # connector issues are debuggable without packet capture.
    oauth_enabled = bool(
        os.environ.get("NEURALSCAPE_PUBLIC_URL", "").strip()
        and os.environ.get("NEURALSCAPE_USER_TOKEN_SECRET", "").strip()
    )
    logging.getLogger("uvicorn.access").setLevel(
        logging.INFO if oauth_enabled else logging.WARNING
    )
