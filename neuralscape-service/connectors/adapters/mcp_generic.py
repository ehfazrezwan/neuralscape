"""Generic MCP-client connector — pulls from any remote MCP server.

This is the universal adapter: it makes Neuralscape an MCP *client* of another
server and ingests that server's exposed **resources** (the standard MCP
data primitive). It covers Notion/Drive *via their MCP servers* — and any
future server — with no bespoke code.

config:
  - ``transport``: "http" (Streamable HTTP, default) or "stdio"
  - http: ``url`` (required), optional ``headers``
  - stdio: ``command`` (required), optional ``args`` (list). Disabled unless the
    operator allowlists the exact command line in CONNECTOR_STDIO_COMMANDS; see
    ``connectors.stdio_policy``. ``env`` is not permitted.
  - ``server_name``: label used in the retrieval handle (default = connector_id)
credentials (optional): merged into HTTP headers as ``{"headers": {...}}`` or a
  bearer token ``{"token": "..."}`` (sent as ``Authorization: Bearer``).

Requires a live MCP server at runtime; in tests a fake session is injected via
``session_factory`` so the adapter logic is exercised without a network.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from connectors.base import ConnectorAdapter, ConnectorResource

logger = logging.getLogger(__name__)


class MCPConnectorAdapter(ConnectorAdapter):
    connector_type = "mcp"

    def __init__(self, connector_id: str, credentials: dict, config: dict | None = None):
        super().__init__(connector_id, credentials, config)
        # Test seam: an async-contextmanager factory yielding an initialized
        # ClientSession-like object exposing list_resources()/read_resource().
        self.session_factory = None

    def _headers(self) -> dict:
        headers = dict(self.config.get("headers", {}))
        headers.update(self.credentials.get("headers", {}))
        token = self.credentials.get("token")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @asynccontextmanager
    async def _session(self):
        """Yield an initialized MCP client session for the configured transport."""
        # Backstop for configs stored before the allowlist existed (the API
        # validates at registration); runs before anything is spawned.
        from config import settings
        from connectors.stdio_policy import validate_mcp_config

        from connectors.stdio_policy import ConnectorConfigError

        try:
            validate_mcp_config(self.config, settings.connector_stdio_commands_set())
        except ConnectorConfigError:
            # Ids only: never the command, args or env (they are caller-chosen).
            logger.warning(
                "MCP connector config refused at session time: connector_id=%s owner=%s",
                self.connector_id, getattr(self, "owner_user_id", None),
            )
            raise

        if self.session_factory is not None:
            async with self.session_factory() as session:
                yield session
            return

        from mcp import ClientSession

        transport = self.config.get("transport", "http")
        if transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=self.config["command"],
                args=self.config.get("args", []),
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        else:
            from mcp.client.streamable_http import streamablehttp_client

            url = self.config.get("url")
            if not url:
                raise ValueError("MCP connector (http) requires config.url")
            async with streamablehttp_client(url, headers=self._headers()) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    async def list_resources(
        self, cursor: str | None = None
    ) -> tuple[list[ConnectorResource], str | None]:
        async with self._session() as session:
            result = await session.list_resources()
        resources: list[ConnectorResource] = []
        for r in getattr(result, "resources", []) or []:
            uri = str(getattr(r, "uri", "") or "")
            if not uri:
                continue
            resources.append(
                ConnectorResource(
                    external_id=uri,
                    title=getattr(r, "name", None) or getattr(r, "title", None),
                    url=uri if uri.startswith("http") else None,
                    revision=None,  # MCP resources don't carry a standard version marker
                    raw={"mime_type": getattr(r, "mimeType", None)},
                )
            )
        # The MCP SDK paginates internally; a single call returns the full set.
        return resources, None

    async def fetch(self, resource: ConnectorResource) -> str:
        async with self._session() as session:
            result = await session.read_resource(resource.external_id)
        parts: list[str] = []
        for content in getattr(result, "contents", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)

    def retrieval_handle(self, resource: ConnectorResource) -> dict:
        return {
            "mcp_server": self.config.get("server_name", self.connector_id),
            "tool": "read_resource",
            "args": {"uri": resource.external_id},
        }
