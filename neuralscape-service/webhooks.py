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

# At most this many name lookups are in flight at once. A lookup that overruns the timeout is
# abandoned but keeps its slot until the C resolver gives up, so a name server that never
# answers cannot pile up threads: further lookups are refused while the slots are taken.
MAX_RESOLVER_THREADS = 4
_resolver_slots = threading.BoundedSemaphore(MAX_RESOLVER_THREADS)


def _parse(url: str) -> tuple[str, str, int] | None:
    """``(scheme, host, port)`` as httpx will send it, or None when it is not a usable URL.

    Uses the parser the connector guard (and the sender) uses, so the host judged here is
    the host contacted: userinfo, backslashes and Unicode dots cannot make two parsers disagree.
    """
    import httpx

    try:
        parsed = httpx.URL(url or "")
        port = parsed.port
    except (httpx.InvalidURL, ValueError):
        return None
    host = parsed.raw_host.decode("ascii", "replace").lower().rstrip(".")
    if parsed.scheme not in ("http", "https") or not host:
        return None
    port = port or (443 if parsed.scheme == "https" else 80)
    if not 0 < port < 65536:
        return None
    return parsed.scheme, host, port


def _log_host(url: str) -> str:
    """A best-effort host for log lines (never the path, query or userinfo)."""
    try:
        return urlparse(url or "").hostname or "?"
    except ValueError:
        return "?"


def webhook_url_allowed(url: str) -> bool:
    """Whether ``url`` is an absolute http(s) URL with a hostname (and a valid port)."""
    return _parse(url) is not None


def _host_and_port(url: str) -> tuple[str, int]:
    parsed = _parse(url)
    if parsed is None:
        raise ValueError("not a usable webhook URL")
    return parsed[1], parsed[2]


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
    if not _resolver_slots.acquire(blocking=False):
        raise OSError("resolver busy")

    def lookup() -> None:
        try:
            box["infos"] = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except (OSError, UnicodeError) as exc:  # UnicodeError: an IDNA-invalid name
            box["error"] = exc
        finally:
            _resolver_slots.release()

    thread = threading.Thread(target=lookup, daemon=True, name="ns-queue-webhook-dns")
    try:
        thread.start()
    except BaseException:
        _resolver_slots.release()
        raise
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
    except (OSError, TimeoutError, UnicodeError) as exc:
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
    parsed = _parse(url)
    if parsed is None:
        logger.warning(
            "WEBHOOK_QUEUE_EMPTY_URL rejected — only absolute http(s) URLs with a hostname "
            "and a valid port are allowed (host %s)", _log_host(url),
        )
        return False
    host = parsed[1]
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
