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
- link-local and cloud-metadata targets are ALWAYS refused (169.254.0.0/16,
  fe80::/10, the metadata endpoints outside them, and IPv4 forms embedded in
  IPv6): the host is resolved at send time, on the webhook thread, and the
  event is not sent when ANY resolved address is one of them. Private and
  loopback targets stay allowed (local consumers and test sinks);
- the connection goes to the address that was validated, never to a second
  lookup of the name, so DNS rebinding cannot swap in a metadata address.
  TLS and the Host header still use the original hostname;
- the request is capped at :data:`WEBHOOK_TIMEOUT_S` seconds;
- delivery runs on a fire-and-forget daemon thread, so a slow or
  malicious endpoint can never block a worker's event loop.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import threading
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT_S = 5.0

# At most this many validated addresses are tried per delivery.
MAX_CONNECT_ATTEMPTS = 4


def webhook_url_allowed(url: str) -> bool:
    """Whether ``url`` is an absolute http(s) URL with a hostname."""
    try:
        parsed = urlparse(url or "")
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.hostname)


def _host_and_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").rstrip(".")
    return host, parsed.port or (443 if parsed.scheme == "https" else 80)


def _literal_never_allowed(host: str) -> bool:
    """Whether ``host`` is a literal link-local / cloud-metadata IP (the connector guard's rules)."""
    from connectors.destination import _literal_ip, _never_allowed

    ip = _literal_ip(host)
    return ip is not None and _never_allowed(ip)


def _resolve(host: str, port: int) -> list[str]:
    """Every distinct address ``host`` resolves to, bounded by the webhook timeout.

    ``getaddrinfo`` has no timeout of its own, so it runs on a throwaway daemon
    thread that is abandoned when it overruns (the caller already runs on the
    webhook's own daemon thread, never on a worker's event loop).
    """
    box: dict = {}

    def lookup() -> None:
        try:
            box["infos"] = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            box["error"] = exc

    thread = threading.Thread(target=lookup, daemon=True, name="ns-queue-webhook-dns")
    thread.start()
    thread.join(WEBHOOK_TIMEOUT_S)
    if thread.is_alive():
        raise TimeoutError(f"name lookup exceeded {WEBHOOK_TIMEOUT_S}s")
    if "error" in box:
        raise box["error"]
    return list(dict.fromkeys(info[4][0].split("%", 1)[0] for info in box["infos"]))


def _validated_addresses(url: str) -> list[str] | None:
    """Resolve ``url``'s host and return the addresses it may connect to, or None to refuse.

    Refuses (with a WARNING naming only the host) when ANY address is link-local or
    cloud metadata, or when the host does not resolve.
    """
    from connectors.destination import _literal_ip, _never_allowed

    host, port = _host_and_port(url)
    literal = _literal_ip(host)
    try:
        addresses = [str(literal)] if literal is not None else _resolve(host, port)
    except (OSError, TimeoutError) as exc:
        logger.warning("queue.empty webhook not sent: could not resolve host %s (%s)", host, type(exc).__name__)
        return None
    if not addresses:
        logger.warning("queue.empty webhook not sent: host %s did not resolve", host)
        return None
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            logger.warning("queue.empty webhook not sent: host %s resolved to an unusable address", host)
            return None
        if _never_allowed(ip):
            logger.warning(
                "queue.empty webhook refused: host %s is or resolves to a link-local or "
                "cloud-metadata address", host,
            )
            return None
    return addresses


def _pinned_transport(addresses: list[str]):
    """An httpx transport whose sockets go to ``addresses`` whatever name the URL carries."""
    import httpcore
    import httpx

    class PinnedBackend(httpcore.SyncBackend):
        def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
            last: Exception | None = None
            for address in addresses[:MAX_CONNECT_ATTEMPTS]:
                try:
                    return super().connect_tcp(
                        address, port, timeout=timeout, local_address=local_address, socket_options=socket_options
                    )
                except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                    last = exc
            raise last or httpcore.ConnectError("no address to connect to")

    transport = httpx.HTTPTransport()
    pool = getattr(transport, "_pool", None)
    if not isinstance(pool, httpcore.ConnectionPool) or not hasattr(pool, "_network_backend"):
        # An httpx/httpcore change we do not understand: fail closed, never unpinned.
        raise RuntimeError("cannot pin webhook connections to validated addresses with this httpx/httpcore")
    pool._network_backend = PinnedBackend()
    return transport


def _post(url: str, payload: dict) -> None:
    """Deliver the event (runs on the webhook thread). Best-effort."""
    import httpx

    try:
        addresses = _validated_addresses(url)
        if addresses is None:
            return
        # trust_env=False: a proxy would resolve the name itself and bypass the pin.
        with httpx.Client(
            transport=_pinned_transport(addresses),
            timeout=WEBHOOK_TIMEOUT_S,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = client.post(url, json=payload)
        logger.debug(
            "queue.empty webhook delivered: %s -> HTTP %s", _host_and_port(url)[0], response.status_code
        )
    except Exception as e:  # noqa: BLE001 — delivery is fire-and-forget
        logger.warning(f"queue.empty webhook delivery failed: {type(e).__name__}")


def fire_queue_empty(url: str, payload: dict) -> bool:
    """Dispatch a ``queue.empty`` event on a daemon thread.

    Returns whether dispatch was attempted (False when the URL fails the
    SSRF guard, including a literal link-local / metadata IP). Never raises
    and never blocks beyond thread start; name resolution happens on the
    webhook thread at send time.
    """
    if not webhook_url_allowed(url):
        logger.warning(
            "WEBHOOK_QUEUE_EMPTY_URL rejected — only absolute http(s) URLs "
            "with a hostname are allowed (got %r)", url,
        )
        return False
    host, _ = _host_and_port(url)
    if _literal_never_allowed(host):
        logger.warning(
            "WEBHOOK_QUEUE_EMPTY_URL rejected — host %s is a link-local or "
            "cloud-metadata address", host,
        )
        return False
    threading.Thread(
        target=_post, args=(url, payload), daemon=True, name="ns-queue-webhook"
    ).start()
    return True
