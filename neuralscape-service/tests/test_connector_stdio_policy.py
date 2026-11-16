"""Stdio MCP connector allowlist: policy, registration route, adapter backstop.

A stdio connector spawns a subprocess on the worker, so it is off unless the
operator allowlists the exact command line (CONNECTOR_STDIO_COMMANDS).
"""

import sys

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

import main
from config import settings
from connectors.stdio_policy import ConnectorConfigError, stdio_command_line, validate_mcp_config
from connectors.vault import ConnectorVault
from tests.fake_vault_store import FakeVaultStore

ALLOWED = "/opt/ok/server --readonly"


@pytest.fixture
def stdio_setting():
    saved = settings.connector_stdio_commands
    yield lambda v: setattr(settings, "connector_stdio_commands", v)
    settings.connector_stdio_commands = saved


@pytest.fixture
def client(monkeypatch):
    vault = ConnectorVault(FakeVaultStore(), Fernet(Fernet.generate_key()))
    monkeypatch.setattr(main, "_vault", vault)
    c = TestClient(main.app, raise_server_exceptions=False)
    c.vault = vault
    return c


def _body(config, cid="mcp-1", ctype="mcp"):
    return {"connector_id": cid, "connector_type": ctype, "credentials": {}, "config": config}


class TestPolicy:
    def test_defaults_disable_stdio(self):
        assert settings.connector_stdio_commands == ""
        assert settings.connector_stdio_commands_set() == set()

    def test_csv_parsing(self, stdio_setting):
        stdio_setting(" /a/x , /b/y --z ,, ")
        assert settings.connector_stdio_commands_set() == {"/a/x", "/b/y --z"}

    def test_entries_are_normalised(self, stdio_setting):
        stdio_setting("  /opt/ok/server    --readonly , '/opt/my dir/srv' \"--a b\" ,'unbalanced")
        assert settings.connector_stdio_commands_set() == {
            "/opt/ok/server --readonly",
            "'/opt/my dir/srv' '--a b'",
        }
        validate_mcp_config({"transport": "stdio", "command": "/opt/my dir/srv", "args": ["--a b"]},
                            settings.connector_stdio_commands_set())

    def test_http_default_and_explicit_ok(self):
        validate_mcp_config(None, set())
        validate_mcp_config({"url": "https://x/mcp"}, set())
        validate_mcp_config({"transport": "http", "url": "https://x/mcp"}, set())

    def test_unknown_transport_rejected(self):
        with pytest.raises(ConnectorConfigError, match="unknown MCP transport"):
            validate_mcp_config({"transport": "Stdio", "command": "x"}, {"x"})
        with pytest.raises(ConnectorConfigError):
            validate_mcp_config({"transport": "websocket"}, set())

    def test_stdio_disabled_when_allowlist_empty(self):
        with pytest.raises(ConnectorConfigError, match="not enabled"):
            validate_mcp_config({"transport": "stdio", "command": "/bin/true"}, set())

    def test_exact_match_only(self):
        allowed = {ALLOWED}
        validate_mcp_config({"transport": "stdio", "command": "/opt/ok/server", "args": ["--readonly"]}, allowed)
        for cfg in (
            {"command": "/opt/ok/server"},  # missing approved args
            {"command": "/opt/ok/server", "args": ["--readonly", "--extra"]},
            {"command": "server", "args": ["--readonly"]},  # no PATH resolution
            {"command": "/opt/ok/../ok/server", "args": ["--readonly"]},  # no normalisation
            {"command": "/opt/ok/server --readonly"},  # split ambiguity: quoted differently
        ):
            with pytest.raises(ConnectorConfigError, match="allowlist"):
                validate_mcp_config({"transport": "stdio", **cfg}, allowed)

    def test_bad_shapes(self):
        for cfg in ({"transport": "stdio"}, {"transport": "stdio", "command": 3},
                    {"transport": "stdio", "command": "x", "args": "-c"},
                    {"transport": "stdio", "command": "x", "args": [1]}):
            with pytest.raises(ConnectorConfigError):
                validate_mcp_config(cfg, {"x"})

    def test_env_rejected_even_when_allowlisted(self):
        cfg = {"transport": "stdio", "command": "/opt/ok/server", "args": ["--readonly"],
               "env": {"LD_PRELOAD": "/tmp/evil.so"}}
        with pytest.raises(ConnectorConfigError, match="env"):
            validate_mcp_config(cfg, {ALLOWED})

    def test_command_line_quotes_unambiguously(self):
        assert stdio_command_line({"command": "a b", "args": []}) != stdio_command_line({"command": "a", "args": ["b"]})


class TestRegisterRoute:
    def test_stdio_refused_by_default(self, client):
        r = client.post("/v1/connectors", json=_body({"transport": "stdio", "command": "/bin/sh", "args": ["-c", "id"]}))
        assert r.status_code == 422
        assert "not enabled" in r.json()["detail"]
        assert client.get("/v1/connectors").json() == {"connectors": []}

    def test_non_allowlisted_command_refused(self, client, stdio_setting):
        stdio_setting(ALLOWED)
        r = client.post("/v1/connectors", json=_body({"transport": "stdio", "command": "/bin/sh", "args": ["-c", "id"]}))
        assert r.status_code == 422
        assert "allowlist" in r.json()["detail"]

    def test_allowlisted_command_registers(self, client, stdio_setting):
        stdio_setting(ALLOWED)
        r = client.post("/v1/connectors", json=_body({"transport": "stdio", "command": "/opt/ok/server", "args": ["--readonly"]}))
        assert r.status_code == 201

    def test_refusal_logged_with_ids_only(self, client, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            client.post("/v1/connectors", json=_body(
                {"transport": "stdio", "command": "/bin/sh", "args": ["-c", "SECRETARG"], "env": {"K": "SECRETENV"}},
                cid="logged-1"))
        text = caplog.text
        assert "connector_id=logged-1" in text and "owner=" in text
        assert "/bin/sh" not in text and "SECRETARG" not in text and "SECRETENV" not in text

    def test_env_refused_on_route(self, client, stdio_setting):
        stdio_setting(ALLOWED)
        r = client.post("/v1/connectors", json=_body(
            {"transport": "stdio", "command": "/opt/ok/server", "args": ["--readonly"], "env": {"PATH": "/tmp"}}))
        assert r.status_code == 422

    def test_unknown_transport_refused_and_http_unaffected(self, client, monkeypatch):
        import connectors.destination as destination

        async def public(host, port):
            return ["93.184.216.34"]

        monkeypatch.setattr(destination, "resolve_host", public)
        assert client.post("/v1/connectors", json=_body({"transport": "pipe"})).status_code == 422
        assert client.post("/v1/connectors", json=_body({"url": "https://x/mcp"})).status_code == 201

    def test_other_connector_types_unaffected(self, client):
        r = client.post("/v1/connectors", json=_body({"foo": 1}, cid="n1", ctype="notion"))
        assert r.status_code == 201


class TestAdapterBackstop:
    """A stdio config stored before the allowlist existed must not spawn."""

    @pytest.mark.asyncio
    async def test_session_refuses_and_spawns_nothing(self, monkeypatch):
        from connectors.adapters.mcp_generic import MCPConnectorAdapter
        import mcp.client.stdio as stdio_mod

        spawned = []

        def boom(*a, **k):
            spawned.append(a)
            raise AssertionError("must not spawn")

        monkeypatch.setattr(stdio_mod, "stdio_client", boom)
        adapter = MCPConnectorAdapter(
            "legacy", {}, {"transport": "stdio", "command": sys.executable, "args": ["-c", "print(1)"]}
        )
        with pytest.raises(ConnectorConfigError):
            await adapter.list_resources()
        assert spawned == []

    @pytest.mark.asyncio
    async def test_session_refusal_logged_with_ids_only(self, caplog):
        import logging

        from connectors.registry import build_adapter

        adapter = build_adapter({"connector_id": "legacy-2", "connector_type": "mcp", "owner_user_id": "alice",
                                 "credentials": {},
                                 "config": {"transport": "stdio", "command": "/bin/sh", "args": ["-c", "SECRETARG"]}})
        with caplog.at_level(logging.WARNING), pytest.raises(ConnectorConfigError):
            await adapter.list_resources()
        assert "connector_id=legacy-2 owner=alice" in caplog.text
        assert "/bin/sh" not in caplog.text and "SECRETARG" not in caplog.text

    @pytest.mark.asyncio
    async def test_allowlisted_session_reaches_spawn(self, monkeypatch, stdio_setting):
        from connectors.adapters.mcp_generic import MCPConnectorAdapter
        import mcp.client.stdio as stdio_mod

        stdio_setting(ALLOWED)
        seen = []

        def fake(params):
            seen.append(params)
            raise RuntimeError("spawn reached")

        monkeypatch.setattr(stdio_mod, "stdio_client", fake)
        adapter = MCPConnectorAdapter("ok", {}, {"transport": "stdio", "command": "/opt/ok/server", "args": ["--readonly"]})
        with pytest.raises(RuntimeError, match="spawn reached"):
            await adapter.list_resources()
        assert seen[0].command == "/opt/ok/server" and seen[0].env is None
