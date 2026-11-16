"""SSRF guard for connectors that fetch a caller-supplied URL (connectors.destination).

Covers the policy (fake resolver, no network), the registration route, the
fetch-time backstop in both URL-taking adapters, redirects, and the connect pin.
"""

from __future__ import annotations

import ipaddress

import httpcore
import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

import connectors.destination as destination
import main
from connectors.adapters.generic_rest import GenericRestAdapter
from connectors.adapters.mcp_generic import MCPConnectorAdapter
from connectors.destination import (
    DestinationRefused,
    _GuardedBackend,
    guarded_async_client,
    parse_allowed_hosts,
    validate_connector_destinations,
    validate_url,
)
from connectors.stdio_policy import ConnectorConfigError
from connectors.vault import ConnectorVault
from tests.fake_vault_store import FakeVaultStore

PUBLIC = "93.184.216.34"
NO_ALLOW: frozenset[str] = frozenset()


def fake_dns(table: dict[str, list[str]]):
    async def resolver(host: str, port: int) -> list[str]:
        if host not in table:
            raise OSError("no such host")
        return list(table[host])

    return resolver


DNS = fake_dns(
    {
        "api.example.com": [PUBLIC],
        "two-faced.example.com": [PUBLIC, "10.0.0.7"],
        "sneaky.example.com": ["127.0.0.1"],
        "meta.example.com": ["169.254.169.254"],
        "v6.example.com": ["fd00:ec2::254"],
        "mapped.example.com": ["::ffff:10.1.2.3"],
        "internal-mcp.corp": ["10.9.9.9"],
        "localhost": ["127.0.0.1", "::1"],
    }
)


@pytest.fixture(autouse=True)
def system_dns(monkeypatch):
    """Every test sees the fake DNS, never the network."""
    monkeypatch.setattr(destination, "resolve_host", DNS)


# ── policy ──────────────────────────────────────────────────────────────────

REFUSED_LITERALS = [
    "127.0.0.1", "127.1", "10.0.0.5", "172.16.0.1", "192.168.1.1", "169.254.169.254",
    "0.0.0.0", "100.64.0.1", "224.0.0.1", "240.0.0.1", "198.18.0.1",
    "[::1]", "[::]", "[fd00:ec2::254]", "[fc00::1]", "[fe80::1]", "[ff02::1]",
    "[::ffff:127.0.0.1]", "[::ffff:169.254.169.254]", "[::127.0.0.1]", "[::10.0.0.1]",
    "[64:ff9b::7f00:1]", "[2002:7f00:1::]",
    # odd IPv4 spellings the C resolver reads as 127.0.0.1
    "2130706433", "0x7f.1", "0177.0.0.1", "0x7f000001",
]


@pytest.mark.parametrize("host", REFUSED_LITERALS)
@pytest.mark.asyncio
async def test_non_public_literal_refused(host):
    with pytest.raises(DestinationRefused):
        await validate_url(f"https://{host}/mcp", NO_ALLOW)


@pytest.mark.parametrize("host", ["8.8.8.8", PUBLIC, "[2606:4700:4700::1111]"])
@pytest.mark.asyncio
async def test_public_literal_ok(host):
    assert await validate_url(f"https://{host}/x", NO_ALLOW)


@pytest.mark.asyncio
async def test_public_hostname_ok_and_returns_validated_addresses():
    assert await validate_url("https://api.example.com/v1/items", NO_ALLOW) == [PUBLIC]


@pytest.mark.parametrize("host", ["sneaky", "meta", "v6", "mapped", "two-faced", "localhost"])
@pytest.mark.asyncio
async def test_hostname_resolving_to_non_public_refused(host):
    name = "localhost" if host == "localhost" else f"{host}.example.com"
    with pytest.raises(DestinationRefused, match="not permitted") as exc:
        await validate_url(f"https://{name}/", NO_ALLOW)
    assert exc.value.reason in ("non_public", "link_local_or_metadata")


@pytest.mark.asyncio
async def test_mixed_public_and_private_answer_refused():
    # One private record among public ones is enough: the client may pick it.
    with pytest.raises(DestinationRefused):
        await validate_url("https://two-faced.example.com/", NO_ALLOW)


@pytest.mark.asyncio
async def test_unresolvable_host_refused():
    with pytest.raises(DestinationRefused, match="not permitted") as unresolved:
        await validate_url("https://nowhere.example.com/", NO_ALLOW)
    with pytest.raises(DestinationRefused) as private:
        await validate_url("https://sneaky.example.com/", NO_ALLOW)
    # The caller cannot tell "does not resolve" from "is private": same text, category only in .reason
    assert str(unresolved.value) == str(private.value)
    assert (unresolved.value.reason, private.value.reason) == ("unresolvable", "non_public")


@pytest.mark.parametrize("url", ["http://api.example.com/x", "ftp://api.example.com/", "file:///etc/passwd",
                                 "gopher://api.example.com/", "//api.example.com/x", "https:///x", "", "not a url"])
@pytest.mark.asyncio
async def test_non_https_and_malformed_refused(url):
    with pytest.raises(DestinationRefused):
        await validate_url(url, NO_ALLOW)


@pytest.mark.asyncio
async def test_non_string_url_refused():
    for bad in (None, 5, ["https://api.example.com"]):
        with pytest.raises(DestinationRefused):
            await validate_url(bad, NO_ALLOW)


@pytest.mark.asyncio
async def test_userinfo_trick_judged_by_the_real_host():
    # Whatever follows '@' is the host httpx will connect to.
    with pytest.raises(DestinationRefused):
        await validate_url("https://api.example.com@127.0.0.1/", NO_ALLOW)
    with pytest.raises(DestinationRefused):
        await validate_url("https://api.example.com:443@sneaky.example.com/", NO_ALLOW)


@pytest.mark.asyncio
async def test_trailing_dot_does_not_dodge_the_check():
    with pytest.raises(DestinationRefused):
        await validate_url("https://localhost./", NO_ALLOW)


class TestAllowlist:
    def test_parse(self):
        assert parse_allowed_hosts(" Internal-MCP.corp:8080 ,, db.corp. ") == {"internal-mcp.corp:8080", "db.corp"}
        assert parse_allowed_hosts("") == frozenset()

    @pytest.mark.asyncio
    async def test_allowlisted_internal_host_accepted_over_https_and_http(self):
        allow = parse_allowed_hosts("internal-mcp.corp")
        assert await validate_url("https://internal-mcp.corp/mcp", allow) == ["10.9.9.9"]
        assert await validate_url("http://internal-mcp.corp/mcp", allow) == ["10.9.9.9"]

    @pytest.mark.asyncio
    async def test_bare_entry_means_the_default_ports_only(self):
        allow = parse_allowed_hosts("internal-mcp.corp")
        for url in ("http://internal-mcp.corp:9000/mcp", "https://internal-mcp.corp:8443/mcp"):
            with pytest.raises(DestinationRefused):
                await validate_url(url, allow)
        assert await validate_url("http://internal-mcp.corp:9000/mcp", parse_allowed_hosts("internal-mcp.corp:9000"))

    @pytest.mark.parametrize("addr,host", [
        ("169.254.169.254", "meta.example.com"), ("168.63.129.16", "azure.example.com"),
        ("fe80::1", "ll.example.com"), ("fd00:ec2::254", "v6.example.com"), ("::ffff:169.254.169.254", "mapped-meta.example.com"),
    ])
    @pytest.mark.asyncio
    async def test_allowlisted_name_still_cannot_reach_link_local_or_metadata(self, addr, host):
        dns = fake_dns({host: [addr]})
        with pytest.raises(DestinationRefused) as exc:
            await validate_url(f"https://{host}/", parse_allowed_hosts(host), dns)
        assert exc.value.reason == "link_local_or_metadata"
        with pytest.raises(DestinationRefused):
            await validate_url(f"https://[{addr}]/" if ":" in addr else f"https://{addr}/", parse_allowed_hosts(addr))

    @pytest.mark.asyncio
    async def test_allowlisted_name_may_still_be_private_or_loopback(self):
        dns = fake_dns({"h.corp": ["10.1.1.1", "127.0.0.1"]})
        assert await validate_url("https://h.corp/", parse_allowed_hosts("h.corp"), dns)

    @pytest.mark.asyncio
    async def test_host_port_entry_is_exact(self):
        allow = parse_allowed_hosts("internal-mcp.corp:8080")
        assert await validate_url("http://internal-mcp.corp:8080/mcp", allow)
        with pytest.raises(DestinationRefused):
            await validate_url("http://internal-mcp.corp:9090/mcp", allow)

    @pytest.mark.asyncio
    async def test_allowlist_is_by_exact_name_only(self):
        allow = parse_allowed_hosts("internal-mcp.corp")
        with pytest.raises(DestinationRefused):
            await validate_url("https://sneaky.example.com/", allow)
        with pytest.raises(DestinationRefused):
            await validate_url("https://evil.internal-mcp.corp/", allow)
        with pytest.raises(DestinationRefused):
            await validate_url("http://api.example.com/", allow)  # http still needs the host listed

    @pytest.mark.asyncio
    async def test_allowlisted_literal_ip(self):
        allow = parse_allowed_hosts("127.0.0.1:8123,[::1]:8123")
        assert await validate_url("http://127.0.0.1:8123/x", allow)
        assert await validate_url("http://[::1]:8123/x", allow)
        with pytest.raises(DestinationRefused):
            await validate_url("http://127.0.0.1:8124/x", allow)

    def test_default_is_empty(self):
        from config import settings

        assert settings.connector_allowed_hosts == ""
        assert settings.connector_allowed_hosts_set() == frozenset()


def test_ipv4_mapped_classes_are_judged_by_the_embedded_address():
    for text in ("::ffff:127.0.0.1", "::ffff:10.0.0.1", "::ffff:169.254.169.254", "::192.168.0.1"):
        assert destination._non_public(ipaddress.ip_address(text)), text
    assert not destination._non_public(ipaddress.ip_address("::ffff:8.8.8.8"))


def test_embedded_ipv4_is_extracted_from_every_ipv6_carrier():
    """The stdlib's own flags differ across Python versions; the guard does not rely on them."""
    expect = {
        "::ffff:10.1.2.3": "10.1.2.3",
        "::10.1.2.3": "10.1.2.3",
        "64:ff9b::a01:203": "10.1.2.3",
        "2002:a01:203::": "10.1.2.3",
    }
    for text, v4 in expect.items():
        assert ipaddress.IPv4Address(v4) in destination._embedded_ipv4(ipaddress.IPv6Address(text)), text
    assert destination._embedded_ipv4(ipaddress.IPv6Address("::1")) == []
    assert destination._embedded_ipv4(ipaddress.IPv6Address("2606:4700::1111")) == []


# ── registration ────────────────────────────────────────────────────────────


def _body(ctype, config, cid="c1"):
    return {"connector_id": cid, "connector_type": ctype, "credentials": {"token": "t"}, "config": config}


@pytest.fixture
def client(monkeypatch):
    vault = ConnectorVault(FakeVaultStore(), Fernet(Fernet.generate_key()))
    monkeypatch.setattr(main, "_vault", vault)
    c = TestClient(main.app, raise_server_exceptions=False)
    c.vault = vault
    return c


BAD_CONFIGS = [
    ("mcp", {"url": "https://127.0.0.1/mcp"}),
    ("mcp", {"transport": "http", "url": "https://meta.example.com/mcp"}),
    ("mcp", {"url": "http://api.example.com/mcp"}),
    ("mcp", {"url": "https://[::ffff:127.0.0.1]/mcp"}),
    ("mcp", {}),  # http is the default transport and needs a url
    ("generic_rest", {"list_url": "https://169.254.169.254/latest/meta-data/"}),
    ("generic_rest", {"list_url": "https://[fd00:ec2::254]/"}),
    ("generic_rest", {"list_url": "https://api.example.com/items", "item_url_template": "https://10.0.0.5/{id}"}),
    ("generic_rest", {"list_url": "https://api.example.com/items", "item_url_template": "http://api.example.com/{id}"}),
    ("generic_rest", {"list_url": "https://api.example.com/items", "item_url_template": "{id}"}),
    ("generic_rest", {"list_url": "https://api.example.com/items", "item_url_template": "https://api.example.com/{nope}"}),
    ("generic_rest", {"list_url": "https://two-faced.example.com/items"}),
    ("generic_rest", {}),
]


class TestRegistration:
    @pytest.mark.parametrize("ctype,config", BAD_CONFIGS)
    def test_bad_destination_is_422_and_nothing_is_stored(self, client, ctype, config):
        r = client.post("/v1/connectors", json=_body(ctype, config))
        assert r.status_code == 422, r.text
        assert client.get("/v1/connectors/c1").status_code == 404

    def test_public_destinations_register(self, client):
        assert client.post("/v1/connectors", json=_body("mcp", {"url": "https://api.example.com/mcp"})).status_code == 201
        ok = {"list_url": "https://api.example.com/items", "item_url_template": "https://api.example.com/items/{id}"}
        assert client.post("/v1/connectors", json=_body("generic_rest", ok, "c2")).status_code == 201

    def test_stdio_and_fixed_host_types_are_not_url_checked(self, client):
        assert client.post("/v1/connectors", json=_body("notion", {"x": "http://127.0.0.1"}, "n1")).status_code == 201

    def test_allowlisted_internal_host_registers_over_http(self, client, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "connector_allowed_hosts", "internal-mcp.corp:9000")
        assert client.post("/v1/connectors", json=_body("mcp", {"url": "http://internal-mcp.corp:9000/mcp"})).status_code == 201
        # ...and only that host
        assert client.post("/v1/connectors", json=_body("mcp", {"url": "http://sneaky.example.com/mcp"}, "c3")).status_code == 422

    def test_refusal_is_logged_with_ids_only(self, client, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="main"):
            client.post("/v1/connectors", json=_body("mcp", {"url": "https://user:s3cr3t@127.0.0.1/mcp"}))
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "connector_id=c1" in text and "owner=" in text
        assert "s3cr3t" not in text and "127.0.0.1" not in text
        assert "127.0.0.1" not in client.post(
            "/v1/connectors", json=_body("mcp", {"url": "https://127.0.0.1/mcp"})).text


@pytest.mark.asyncio
async def test_validate_connector_destinations_direct():
    with pytest.raises(ConnectorConfigError):
        await validate_connector_destinations("generic_rest", {"list_url": "https://10.1.1.1/"}, NO_ALLOW)
    await validate_connector_destinations("notion", {}, NO_ALLOW)
    await validate_connector_destinations("mcp", {"transport": "stdio", "command": "x"}, NO_ALLOW)  # stdio_policy's job


# ── fetch-time backstop: generic_rest ───────────────────────────────────────


class Recorder:
    """MockTransport handler that counts the requests that actually reach the wire."""

    def __init__(self, response=None):
        self.requests: list[httpx.Request] = []
        self.response = response or (lambda req: httpx.Response(200, json=[{"id": "1"}]))

    def __call__(self, request):
        self.requests.append(request)
        return self.response(request)

    @property
    def transport(self):
        return httpx.MockTransport(self)


def _rest(config, rec, **kw):
    adapter = GenericRestAdapter("rest-1", {"bearer": "tok"}, config)
    adapter.owner_user_id = "u1"
    adapter._transport = rec.transport
    return adapter


@pytest.mark.asyncio
async def test_stored_bad_list_url_refused_at_fetch_time_and_nothing_is_sent():
    rec = Recorder()
    adapter = _rest({"list_url": "https://meta.example.com/latest/meta-data/"}, rec)
    with pytest.raises(DestinationRefused):
        await adapter.list_resources()
    assert rec.requests == []
    await adapter.aclose()


@pytest.mark.asyncio
async def test_stored_bad_item_template_refused_at_fetch_time_after_good_list():
    rec = Recorder()
    adapter = _rest({"list_url": "https://api.example.com/items", "item_url_template": "https://10.0.0.5/{id}"}, rec)
    resources, _ = await adapter.list_resources()
    assert len(rec.requests) == 1
    with pytest.raises(DestinationRefused):
        await adapter.fetch(resources[0])
    assert len(rec.requests) == 1  # the internal fetch never left
    await adapter.aclose()


@pytest.mark.asyncio
async def test_good_rest_connector_still_works_end_to_end():
    def respond(req):
        if req.url.path == "/items":
            return httpx.Response(200, json=[{"id": "7"}])
        return httpx.Response(200, json={"body": "hello"})

    rec = Recorder(respond)
    adapter = _rest({"list_url": "https://api.example.com/items", "item_url_template": "https://api.example.com/items/{id}",
                     "detail_content_field": "body"}, rec)
    resources, _ = await adapter.list_resources()
    assert await adapter.fetch(resources[0]) == "hello"
    assert [r.headers["authorization"] for r in rec.requests] == ["Bearer tok", "Bearer tok"]
    await adapter.aclose()


@pytest.mark.asyncio
async def test_rest_redirect_to_private_address_is_not_followed():
    rec = Recorder(lambda req: httpx.Response(302, headers={"location": "https://169.254.169.254/latest/meta-data/"}))
    adapter = _rest({"list_url": "https://api.example.com/items"}, rec)
    with pytest.raises(httpx.HTTPStatusError):
        await adapter.list_resources()
    assert len(rec.requests) == 1
    await adapter.aclose()


@pytest.mark.asyncio
async def test_every_redirect_hop_is_rechecked_when_following_is_on():
    rec = Recorder(lambda req: httpx.Response(302, headers={"location": "https://meta.example.com/creds"}))
    client = guarded_async_client(follow_redirects=True, transport=rec.transport, allowed=NO_ALLOW)
    with pytest.raises(DestinationRefused):
        await client.get("https://api.example.com/start")
    assert [str(r.url) for r in rec.requests] == ["https://api.example.com/start"]
    await client.aclose()


@pytest.mark.asyncio
async def test_redirect_to_plain_http_refused_when_following():
    rec = Recorder(lambda req: httpx.Response(302, headers={"location": "http://api.example.com/x"}))
    client = guarded_async_client(follow_redirects=True, transport=rec.transport, allowed=NO_ALLOW)
    with pytest.raises(DestinationRefused):
        await client.get("https://api.example.com/start")
    await client.aclose()


@pytest.mark.asyncio
async def test_rest_refusal_logs_ids_only(caplog):
    import logging

    rec = Recorder()
    adapter = _rest({"list_url": "https://127.0.0.1/secret-path?token=abc123"}, rec)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(DestinationRefused):
            await adapter.list_resources()
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "connector_id=rest-1" in text and "owner=u1" in text
    assert "abc123" not in text and "secret-path" not in text and "tok" not in text.replace("token", "")
    await adapter.aclose()


# ── fetch-time backstop: mcp http ───────────────────────────────────────────


@pytest.fixture
def sdk(monkeypatch):
    """Replace the SDK transport so no socket opens; record how it was called."""
    import mcp.client.streamable_http as sh

    calls: list[dict] = []

    class Boom(Exception):
        pass

    def fake(url, headers=None, httpx_client_factory=None, **kw):
        calls.append({"url": url, "headers": headers, "factory": httpx_client_factory})
        raise Boom("reached the SDK")

    monkeypatch.setattr(sh, "streamablehttp_client", fake)
    return calls, Boom


@pytest.mark.parametrize("url", ["https://127.0.0.1/mcp", "https://meta.example.com/mcp", "http://api.example.com/mcp"])
@pytest.mark.asyncio
async def test_stored_bad_mcp_url_refused_at_session_time_and_sdk_never_called(sdk, url):
    calls, _ = sdk
    adapter = MCPConnectorAdapter("mcp-1", {"token": "tok"}, {"url": url})
    with pytest.raises(DestinationRefused):
        await adapter.list_resources()
    assert calls == []


@pytest.mark.asyncio
async def test_good_mcp_url_reaches_sdk_with_a_guarded_client_factory(sdk):
    calls, Boom = sdk
    adapter = MCPConnectorAdapter("mcp-1", {"token": "tok"}, {"url": "https://api.example.com/mcp"})
    with pytest.raises(Boom):
        await adapter.list_resources()
    assert len(calls) == 1 and calls[0]["headers"]["Authorization"] == "Bearer tok"
    client = calls[0]["factory"]({"x": "y"}, None, None)
    try:
        assert client.follow_redirects is True  # the SDK's default; hops are checked by the hook
        assert any(h.__name__ == "check_request" for h in client.event_hooks["request"])
        rec = Recorder(lambda req: httpx.Response(302, headers={"location": "https://127.0.0.1/"}))
        client._transport = rec.transport
        client._mounts = {}
        with pytest.raises(DestinationRefused):
            await client.get("https://api.example.com/mcp")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_mcp_allowlisted_internal_host_reaches_sdk(sdk, monkeypatch):
    from config import settings

    calls, Boom = sdk
    monkeypatch.setattr(settings, "connector_allowed_hosts", "internal-mcp.corp:9000")
    adapter = MCPConnectorAdapter("mcp-1", {}, {"url": "http://internal-mcp.corp:9000/mcp"})
    with pytest.raises(Boom):
        await adapter.list_resources()
    assert calls[0]["url"] == "http://internal-mcp.corp:9000/mcp"


# ── connect pin (DNS rebinding) ─────────────────────────────────────────────


class FakeInner:
    def __init__(self):
        self.connected: list[tuple[str, int]] = []

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.connected.append((host, port))
        return object()

    async def sleep(self, seconds):
        return None


@pytest.mark.asyncio
async def test_connect_goes_to_the_validated_address_not_the_name():
    inner = FakeInner()
    backend = _GuardedBackend(NO_ALLOW, fake_dns({"api.example.com": [PUBLIC]}), inner)
    await backend.connect_tcp("api.example.com", 443)
    assert inner.connected == [(PUBLIC, 443)]


@pytest.mark.asyncio
async def test_rebinding_to_a_private_address_between_check_and_connect_is_refused():
    answers = iter([[PUBLIC], ["127.0.0.1"]])

    async def rebinding(host, port):
        return next(answers)

    inner = FakeInner()
    client = guarded_async_client(allowed=NO_ALLOW, resolver=rebinding, transport=None)
    pool = client._transport._pool
    pool._network_backend._inner = inner
    # first lookup (request hook) is public; the connect-time lookup now says loopback
    with pytest.raises(DestinationRefused):
        await client.get("https://api.example.com/x")
    assert inner.connected == []
    await client.aclose()


@pytest.mark.asyncio
async def test_backend_refuses_private_and_unix_sockets_and_honours_allowlist():
    inner = FakeInner()
    deny = _GuardedBackend(NO_ALLOW, DNS, inner)
    with pytest.raises(DestinationRefused):
        await deny.connect_tcp("sneaky.example.com", 443)
    with pytest.raises(DestinationRefused):
        await deny.connect_unix_socket("/var/run/docker.sock")
    assert inner.connected == []
    allow = _GuardedBackend(parse_allowed_hosts("internal-mcp.corp:9000"), DNS, inner)
    await allow.connect_tcp("internal-mcp.corp", 9000)
    assert inner.connected == [("10.9.9.9", 9000)]


@pytest.mark.asyncio
async def test_connect_falls_through_addresses_on_connect_error():
    class Flaky(FakeInner):
        async def connect_tcp(self, host, port, **kw):
            self.connected.append((host, port))
            if host == "8.8.8.8":
                raise httpcore.ConnectError("down")
            return object()

    inner = Flaky()
    backend = _GuardedBackend(NO_ALLOW, fake_dns({"h.example.com": ["8.8.8.8", PUBLIC]}), inner)
    await backend.connect_tcp("h.example.com", 443)
    assert inner.connected == [("8.8.8.8", 443), (PUBLIC, 443)]


def test_guarded_client_ignores_proxy_environment(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://10.0.0.1:3128")
    client = guarded_async_client(allowed=NO_ALLOW)
    assert client._mounts == {}


# ── review fixes: DNS timeout, connect timeout, mutation killers ────────────


@pytest.mark.asyncio
async def test_hung_dns_lookup_is_refused_not_awaited_forever(monkeypatch):
    import asyncio

    async def hangs(host, port):
        await asyncio.sleep(3600)

    monkeypatch.setattr(destination, "RESOLVE_TIMEOUT_S", 0.05)
    with pytest.raises(DestinationRefused) as exc:
        await validate_url("https://slow.example.com/", NO_ALLOW, hangs)
    assert exc.value.reason == "resolve_timeout"


@pytest.mark.asyncio
async def test_lookup_is_bounded_by_the_clients_connect_timeout():
    import asyncio

    async def hangs(host, port):
        await asyncio.sleep(3600)

    with pytest.raises(DestinationRefused) as exc:
        await destination.validate_target("https", "slow.example.com", 443, NO_ALLOW, hangs, timeout=0.05)
    assert exc.value.reason == "resolve_timeout"


@pytest.mark.asyncio
async def test_connect_falls_through_addresses_on_connect_timeout():
    class Slow(FakeInner):
        async def connect_tcp(self, host, port, **kw):
            self.connected.append((host, port))
            if host == "8.8.8.8":
                raise httpcore.ConnectTimeout("slow")
            return object()

    inner = Slow()
    backend = _GuardedBackend(NO_ALLOW, fake_dns({"h.example.com": ["8.8.8.8", PUBLIC]}), inner)
    await backend.connect_tcp("h.example.com", 443)
    assert inner.connected == [("8.8.8.8", 443), (PUBLIC, 443)]


def test_loose_ipv4_spellings_parse_to_the_address_libc_will_connect_to():
    """Direct parser test: independent of how the stdlib classifies the result."""
    lit = destination._literal_ip
    for text in ("2130706433", "0x7f.1", "0177.0.0.1", "0x7f000001", "127.1"):
        assert lit(text) == ipaddress.IPv4Address("127.0.0.1"), text
    assert lit("abc.def") is None and lit("api.example.com") is None and lit("deadbeef") is None
    assert lit("::1") == ipaddress.IPv6Address("::1")


@pytest.mark.asyncio
async def test_loose_ipv4_literal_is_refused_even_if_dns_would_say_public():
    async def says_public(host, port):
        return [PUBLIC]

    for host in ("2130706433", "0x7f.1", "0177.0.0.1"):
        with pytest.raises(DestinationRefused):
            await validate_url(f"https://{host}/", NO_ALLOW, says_public)


def test_embedded_ipv4_decides_even_when_the_stdlib_calls_the_ipv6_address_global(monkeypatch):
    carrier = ipaddress.IPv6Address("2606:4700::1111")  # public, carries nothing
    assert not destination._non_public(carrier)
    monkeypatch.setattr(destination, "_embedded_ipv4", lambda ip: [ipaddress.IPv4Address("10.0.0.1")])
    assert destination._non_public(carrier)
    assert destination._never_allowed(carrier) is False  # 10.0.0.1 is private, not link-local
    monkeypatch.setattr(destination, "_embedded_ipv4", lambda ip: [ipaddress.IPv4Address("169.254.169.254")])
    assert destination._never_allowed(carrier)


def test_azure_wire_server_refused_for_everyone():
    assert destination._non_public(ipaddress.ip_address("168.63.129.16"))
