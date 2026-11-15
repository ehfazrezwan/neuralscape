"""Policy for the MCP connector's stdio transport.

A stdio MCP connector makes the ingest worker spawn a local subprocess, so a
caller-supplied command would be code execution on the worker. The transport is
therefore **off unless the operator allowlists exact command lines** in
``CONNECTOR_STDIO_COMMANDS``. The same check runs at registration (the API
rejects the config) and again when the adapter opens a session (connectors
stored before the allowlist existed may still hold a stdio config).

An allowlist entry is a full command line in shell-quoted form (``shlex``), e.g.
``/usr/bin/my-mcp-server --readonly``. It is compared for exact equality with
``shlex.join([command, *args])`` of the config: no PATH lookup, no path
normalisation, and the caller cannot append arguments the operator did not
approve. Caller-supplied ``env`` is rejected outright (it is a code-loading
channel: ``LD_PRELOAD``, ``PYTHONPATH``, ``NODE_OPTIONS`` ...); an operator who
needs environment for the server sets it on the worker.
"""

from __future__ import annotations

import shlex

_TRANSPORTS = ("http", "stdio")


class ConnectorConfigError(ValueError):
    """The connector config is not permitted (message is safe to show the caller)."""


def stdio_command_line(config: dict) -> str:
    command = config.get("command")
    args = config.get("args", [])
    if not isinstance(command, str) or not command:
        raise ConnectorConfigError("stdio connector requires config.command (a non-empty string)")
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ConnectorConfigError("stdio connector config.args must be a list of strings")
    return shlex.join([command, *args])


def validate_mcp_config(config: dict | None, allowed: set[str]) -> None:
    """Raise :class:`ConnectorConfigError` unless an ``mcp`` connector config is permitted."""
    config = config or {}
    transport = config.get("transport", "http")
    if transport not in _TRANSPORTS:
        raise ConnectorConfigError(
            f"unknown MCP transport {transport!r}; must be one of {list(_TRANSPORTS)}"
        )
    if transport != "stdio":
        return
    if not allowed:
        raise ConnectorConfigError(
            "stdio transport is not enabled on this server (CONNECTOR_STDIO_COMMANDS is empty)"
        )
    if config.get("env"):
        raise ConnectorConfigError("config.env is not permitted for stdio connectors")
    line = stdio_command_line(config)
    if line not in allowed:
        raise ConnectorConfigError("stdio command is not in the operator allowlist (CONNECTOR_STDIO_COMMANDS)")
