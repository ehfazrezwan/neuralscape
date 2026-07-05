"""Outbound webhooks (roadmap C4 — queue visibility).

``queue.empty``: when a worker finishes a job and its queue is left empty,
it POSTs a small JSON event to ``WEBHOOK_QUEUE_EMPTY_URL`` so
ingest-then-query flows can stop polling per task (see
``worker._make_after_job_end``). Empty URL = feature off.

SSRF posture (deliberately conservative):

- only absolute ``http(s)`` URLs with a hostname are ever contacted —
  ``file:``, ``ftp:``, scheme-less, and host-less URLs are rejected;
- redirects are NEVER followed (``follow_redirects=False`` — a 3xx is
  terminal), so a public URL can't bounce the worker onto an internal
  address;
- the request is capped at :data:`WEBHOOK_TIMEOUT_S` seconds;
- delivery runs on a fire-and-forget daemon thread, so a slow or
  malicious endpoint can never block a worker's event loop.
"""

from __future__ import annotations

import logging
import threading
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT_S = 5.0


def webhook_url_allowed(url: str) -> bool:
    """Whether ``url`` is an absolute http(s) URL with a hostname."""
    try:
        parsed = urlparse(url or "")
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.hostname)


def _post(url: str, payload: dict) -> None:
    """Deliver the event (runs on the webhook thread). Best-effort."""
    import httpx

    try:
        response = httpx.post(
            url,
            json=payload,
            timeout=WEBHOOK_TIMEOUT_S,
            follow_redirects=False,
        )
        logger.debug(
            "queue.empty webhook delivered: %s -> HTTP %s", url, response.status_code
        )
    except Exception as e:  # noqa: BLE001 — delivery is fire-and-forget
        logger.warning(f"queue.empty webhook delivery failed: {e}")


def fire_queue_empty(url: str, payload: dict) -> bool:
    """Dispatch a ``queue.empty`` event on a daemon thread.

    Returns whether dispatch was attempted (False when the URL fails the
    SSRF guard). Never raises and never blocks beyond thread start.
    """
    if not webhook_url_allowed(url):
        logger.warning(
            "WEBHOOK_QUEUE_EMPTY_URL rejected — only absolute http(s) URLs "
            "with a hostname are allowed (got %r)", url,
        )
        return False
    threading.Thread(
        target=_post, args=(url, payload), daemon=True, name="ns-queue-webhook"
    ).start()
    return True
