"""Destination policy for connectors that fetch a caller-supplied URL.

A connector's ``config`` (``generic_rest``: ``list_url`` / ``item_url_template``;
``mcp`` over http: ``url``) is chosen by whoever registers it, and the ingest
worker fetches it with the connector's stored credentials attached. Without a
check that is server-side request forgery: a registered connector can aim the
worker at loopback (the API, Redis, Neo4j on the compose network), link-local
cloud metadata (``169.254.169.254``, ``fd00:ec2::254``) or any private range.

The rules (one module, used by every adapter that takes a URL):

* scheme ``https`` only. ``http`` is allowed only for an allowlisted host.
* the host is resolved (every A and AAAA record) and the URL is refused when ANY
  address is loopback, link-local, private, unique-local, CGNAT, unspecified,
  multicast or reserved, including IPv4 forms embedded in an IPv6 address
  (``::ffff:127.0.0.1``, ``::127.0.0.1``, 6to4, Teredo, NAT64). Literal IPs and
  the odd IPv4 spellings (``2130706433``, ``0x7f.1``, ``0177.0.0.1``) are parsed
  the way the C library will read them, so they cannot slip through as names.
* an operator who really needs an internal target lists exact hostnames
  (optionally ``host:port``) in ``CONNECTOR_ALLOWED_HOSTS``. Allowlisted hosts
  skip the address refusal and may use ``http``. Default: empty.

Enforcement runs twice. At registration (``validate_connector_destinations``,
called by ``POST /v1/connectors``) so a bad config is a 422 and never stored.
At fetch time (``guarded_async_client``) so connectors stored earlier, and DNS
that changed since registration, are still refused. The fetch-time client:

* checks every request, including each redirect hop (a redirect to a private
  address is refused; the adapters also leave ``follow_redirects`` off);
* connects to the address it validated, not to a second lookup of the name
  (a custom ``httpcore`` network backend), so DNS rebinding between the check
  and the connect cannot swap in a private address. TLS still verifies the
  certificate against the original hostname.

Direct connections only: a client built here ignores ``HTTP(S)_PROXY`` (a proxy
would resolve the name itself and bypass the pin).
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from typing import Any

import httpcore
import httpx

from connectors.stdio_policy import ConnectorConfigError

Resolver = Callable[[str, int], Awaitable[list[str]]]

# glibc/BSD ``inet_aton`` also reads decimal ("2130706433"), hex ("0x7f.1"),
# octal ("0177.0.0.1") and short ("127.1") IPv4 forms. A client library hands the
# name to the same resolver, so those must be judged as the IPs they are.
_IPV4_LOOSE = re.compile(r"^[0-9a-fA-FxX.]+$")


class DestinationRefused(ConnectorConfigError):
    """The URL points somewhere connectors may not fetch (message is safe to show the caller)."""


async def resolve_host(host: str, port: int) -> list[str]:
    """Every distinct address ``host`` resolves to (A and AAAA), as strings."""
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(info[4][0].split("%", 1)[0] for info in infos))


def _norm_host(host: str) -> str:
    return host.strip().lower().rstrip(".")


def parse_allowed_hosts(csv: str) -> frozenset[str]:
    """Normalise ``CONNECTOR_ALLOWED_HOSTS`` into exact ``host`` / ``host:port`` entries."""
    return frozenset(_norm_host(e) for e in (csv or "").split(",") if _norm_host(e))


def _allowlisted(host: str, port: int, allowed: frozenset[str]) -> bool:
    return bool(allowed) and (
        host in allowed
        or f"{host}:{port}" in allowed
        or f"[{host}]" in allowed
        or f"[{host}]:{port}" in allowed
    )


def _literal_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    if _IPV4_LOOSE.match(host):
        try:
            return ipaddress.IPv4Address(socket.inet_aton(host))
        except OSError:
            return None
    return None


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> list[ipaddress.IPv4Address]:
    """IPv4 addresses an IPv6 address carries (they decide where traffic really goes)."""
    found: list[ipaddress.IPv4Address] = []
    if ip.ipv4_mapped is not None:
        found.append(ip.ipv4_mapped)
    if ip.sixtofour is not None:
        found.append(ip.sixtofour)
    if ip.teredo is not None:
        found.extend(ip.teredo)
    packed = ip.packed
    if packed[:12] == bytes(12) and ip not in (ipaddress.IPv6Address("::"), ipaddress.IPv6Address("::1")):
        found.append(ipaddress.IPv4Address(packed[12:]))  # IPv4-compatible ::a.b.c.d
    if packed[:12] == bytes.fromhex("0064ff9b0000000000000000"):
        found.append(ipaddress.IPv4Address(packed[12:]))  # NAT64 64:ff9b::/96
    return found


def _non_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and any(_non_public(v4) for v4 in _embedded_ipv4(ip)):
        return True
    return bool(
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_unspecified
        or ip.is_multicast
        or ip.is_reserved
        or not ip.is_global
    )


async def validate_target(
    scheme: str,
    host: str,
    port: int | None,
    allowed: frozenset[str],
    resolver: Resolver | None = None,
) -> list[str]:
    """Raise :class:`DestinationRefused` unless ``scheme://host:port`` may be fetched.

    Returns the validated addresses (the fetch-time client connects to one of them).
    """
    scheme = (scheme or "").lower()
    host = _norm_host(host or "")
    if scheme not in ("https", "http"):
        raise DestinationRefused("connector URL must use https")
    if not host:
        raise DestinationRefused("connector URL has no host")
    port = port or (443 if scheme == "https" else 80)
    internal_ok = _allowlisted(host, port, allowed)
    if scheme == "http" and not internal_ok:
        raise DestinationRefused("connector URL must use https (http is allowed only for CONNECTOR_ALLOWED_HOSTS)")

    literal = _literal_ip(host)
    if literal is not None:
        addresses = [str(literal)]
    else:
        try:
            addresses = await (resolver or resolve_host)(host, port)
        except OSError:
            raise DestinationRefused("connector host could not be resolved") from None
        if not addresses:
            raise DestinationRefused("connector host could not be resolved")
    if not internal_ok:
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address)
            except ValueError:
                raise DestinationRefused("connector host resolved to an unusable address") from None
            if _non_public(ip):
                raise DestinationRefused(
                    "connector host resolves to a non-public address "
                    "(loopback, private, link-local or reserved); an operator can allow an internal "
                    "host with CONNECTOR_ALLOWED_HOSTS"
                )
    return addresses


async def validate_url(url: Any, allowed: frozenset[str] | None = None, resolver: Resolver | None = None) -> list[str]:
    """:func:`validate_target` for a URL string, parsed with the parser httpx will use to send it."""
    if allowed is None:
        from config import settings

        allowed = settings.connector_allowed_hosts_set()
    if not isinstance(url, str) or not url.strip():
        raise DestinationRefused("connector URL must be a non-empty string")
    try:
        parsed = httpx.URL(url.strip())
    except (httpx.InvalidURL, ValueError):
        raise DestinationRefused("connector URL is not a valid URL") from None
    return await validate_target(parsed.scheme, parsed.raw_host.decode("ascii", "replace"), parsed.port, allowed, resolver)


async def validate_connector_destinations(
    connector_type: str,
    config: dict | None,
    allowed: frozenset[str] | None = None,
    resolver: Resolver | None = None,
) -> None:
    """Registration-time check of every caller-supplied URL in a connector config.

    ``notion`` and ``google_drive`` talk to fixed vendor hosts and take no URL, so
    they have nothing to check. ``mcp`` checks its http ``url`` (stdio is
    :mod:`connectors.stdio_policy`'s). ``generic_rest`` checks ``list_url`` and
    ``item_url_template``.
    """
    config = config or {}
    if connector_type == "mcp":
        if config.get("transport", "http") == "http":
            await validate_url(config.get("url"), allowed, resolver)
    elif connector_type == "generic_rest":
        await validate_url(config.get("list_url"), allowed, resolver)
        template = config.get("item_url_template")
        if template:
            if not isinstance(template, str):
                raise DestinationRefused("item_url_template must be a string")
            try:
                sample = template.format(id="0")
            except (KeyError, IndexError, ValueError, AttributeError):
                raise DestinationRefused("item_url_template is not a valid template (only {id} is available)") from None
            await validate_url(sample, allowed, resolver)


class _GuardedBackend(httpcore.AsyncNetworkBackend):
    """Resolve, validate, then connect to the validated address (not to a second lookup)."""

    def __init__(self, allowed: frozenset[str], resolver: Resolver | None, inner: Any = None):
        self._allowed = allowed
        self._resolver = resolver
        self._inner = inner or httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        # Scheme rules are the request hook's job (it sees the URL); here the question
        # is only "which address", so judge as https: the allowlist still applies.
        addresses = await validate_target("https", host, port, self._allowed, self._resolver)
        last: Exception | None = None
        for address in addresses:
            try:
                return await self._inner.connect_tcp(
                    address, port, timeout=timeout, local_address=local_address, socket_options=socket_options
                )
            except httpcore.ConnectError as exc:
                last = exc
        raise last or httpcore.ConnectError("no address to connect to")

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise DestinationRefused("unix sockets are not permitted for connectors")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def guarded_async_client(
    *,
    allowed: frozenset[str] | None = None,
    resolver: Resolver | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    follow_redirects: bool = False,
    **client_kwargs: Any,
) -> httpx.AsyncClient:
    """An ``httpx.AsyncClient`` that refuses forbidden destinations on every request.

    ``transport`` is a test seam (the request hook still runs; only the connect pin
    is skipped because there is no socket). Production callers leave it unset.
    """
    if allowed is None:
        from config import settings

        allowed = settings.connector_allowed_hosts_set()

    async def check_request(request: httpx.Request) -> None:
        url = request.url
        await validate_target(url.scheme, url.raw_host.decode("ascii", "replace"), url.port, allowed, resolver)

    if transport is None:
        transport = httpx.AsyncHTTPTransport()
        pool = getattr(transport, "_pool", None)
        if not isinstance(pool, httpcore.AsyncConnectionPool) or not hasattr(pool, "_network_backend"):
            # An httpx/httpcore change we do not understand: fail closed, never unpinned.
            raise RuntimeError("cannot pin connector connections to validated addresses with this httpx/httpcore")
        pool._network_backend = _GuardedBackend(allowed, resolver)
    hooks = dict(client_kwargs.pop("event_hooks", None) or {})
    hooks["request"] = [check_request, *hooks.get("request", [])]
    return httpx.AsyncClient(
        transport=transport, follow_redirects=follow_redirects, event_hooks=hooks, trust_env=False, **client_kwargs
    )


def mcp_client_factory(allowed: frozenset[str] | None = None, resolver: Resolver | None = None):
    """``httpx_client_factory`` for ``streamablehttp_client``: same defaults as the SDK's, guarded.

    The SDK's client follows redirects; the guard re-checks each hop.
    """
    from mcp.shared._httpx_utils import MCP_DEFAULT_SSE_READ_TIMEOUT, MCP_DEFAULT_TIMEOUT

    def factory(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
        kwargs: dict[str, Any] = {
            "timeout": timeout or httpx.Timeout(MCP_DEFAULT_TIMEOUT, read=MCP_DEFAULT_SSE_READ_TIMEOUT),
        }
        if headers is not None:
            kwargs["headers"] = headers
        if auth is not None:
            kwargs["auth"] = auth
        return guarded_async_client(allowed=allowed, resolver=resolver, follow_redirects=True, **kwargs)

    return factory
