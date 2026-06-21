"""Tests for connector registry + adapters (Notion via mocked httpx, generic MCP)."""

from contextlib import asynccontextmanager

import httpx
import pytest

from connectors.base import ConnectorAdapter, ConnectorResource
from connectors.registry import build_adapter, get_adapter_class


class TestRegistry:
    def test_resolves_known_types(self):
        from connectors.adapters.notion import NotionAdapter
        from connectors.adapters.gdrive import GoogleDriveAdapter
        from connectors.adapters.mcp_generic import MCPConnectorAdapter
        from connectors.adapters.generic_rest import GenericRestAdapter

        assert get_adapter_class("notion") is NotionAdapter
        assert get_adapter_class("google_drive") is GoogleDriveAdapter
        assert get_adapter_class("mcp") is MCPConnectorAdapter
        assert get_adapter_class("generic_rest") is GenericRestAdapter

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            get_adapter_class("dropbox")

    def test_build_adapter_from_record(self):
        adapter = build_adapter({
            "connector_id": "notion-personal",
            "connector_type": "notion",
            "credentials": {"token": "x"},
            "config": {},
        })
        assert isinstance(adapter, ConnectorAdapter)
        assert adapter.connector_type == "notion"


# ── Notion adapter against a mocked httpx transport ──

_SEARCH_RESPONSE = {
    "results": [
        {
            "id": "page-1",
            "url": "https://notion.so/page-1",
            "last_edited_time": "2026-06-20T10:00:00.000Z",
            "properties": {"Name": {"type": "title", "title": [{"plain_text": "Q3 Plan"}]}},
        }
    ],
    "has_more": False,
    "next_cursor": None,
}

_BLOCKS_RESPONSE = {
    "results": [
        {"type": "heading_1", "heading_1": {"rich_text": [{"plain_text": "Goals"}]}},
        {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Ship the connector."}]}},
        {"type": "image", "image": {}},  # non-text block → ignored
    ],
    "has_more": False,
    "next_cursor": None,
}


def _notion_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/v1/search":
        return httpx.Response(200, json=_SEARCH_RESPONSE)
    if path.startswith("/v1/blocks/") and path.endswith("/children"):
        return httpx.Response(200, json=_BLOCKS_RESPONSE)
    return httpx.Response(404, json={"error": "unexpected path", "path": path})


def _notion_adapter():
    from connectors.adapters.notion import NotionAdapter

    adapter = NotionAdapter("notion-personal", {"token": "secret"}, {})
    adapter._http = httpx.AsyncClient(
        base_url="https://api.notion.com/v1",
        transport=httpx.MockTransport(_notion_handler),
    )
    return adapter


class TestNotionAdapter:
    @pytest.mark.asyncio
    async def test_list_resources(self):
        adapter = _notion_adapter()
        try:
            resources, cursor = await adapter.list_resources()
        finally:
            await adapter.aclose()
        assert cursor is None
        assert len(resources) == 1
        r = resources[0]
        assert r.external_id == "page-1"
        assert r.title == "Q3 Plan"
        assert r.url == "https://notion.so/page-1"
        assert r.revision == "2026-06-20T10:00:00.000Z"

    @pytest.mark.asyncio
    async def test_fetch_concatenates_text_blocks(self):
        adapter = _notion_adapter()
        try:
            text = await adapter.fetch(ConnectorResource(external_id="page-1"))
        finally:
            await adapter.aclose()
        assert "Goals" in text
        assert "Ship the connector." in text
        # The image block contributed no text.
        assert text == "Goals\nShip the connector."

    def test_retrieval_handle(self):
        from connectors.adapters.notion import NotionAdapter

        adapter = NotionAdapter("notion-personal", {"token": "x"}, {})
        handle = adapter.retrieval_handle(ConnectorResource(external_id="page-1"))
        assert handle == {
            "mcp_server": "claude_ai_Notion",
            "tool": "notion-fetch",
            "args": {"id": "page-1"},
        }

    def test_source_descriptor_includes_handle(self):
        from connectors.adapters.notion import NotionAdapter

        adapter = NotionAdapter("notion-personal", {"token": "x"}, {})
        sd = adapter.source_descriptor(
            ConnectorResource(external_id="page-1", title="T", url="u", revision="rev")
        )
        assert sd["connector_id"] == "notion-personal"
        assert sd["connector_type"] == "notion"
        assert sd["parent_id"] == "page-1"
        assert sd["retrieval"]["tool"] == "notion-fetch"


# ── Generic MCP adapter against a fake injected session ──


class _FakeResource:
    def __init__(self, uri, name):
        self.uri = uri
        self.name = name
        self.mimeType = "text/markdown"


class _FakeListResult:
    def __init__(self, resources):
        self.resources = resources


class _FakeContent:
    def __init__(self, text):
        self.text = text


class _FakeReadResult:
    def __init__(self, text):
        self.contents = [_FakeContent(text)]


class _FakeSession:
    async def list_resources(self):
        return _FakeListResult([_FakeResource("notion://page-1", "Q3 Plan")])

    async def read_resource(self, uri):
        assert uri == "notion://page-1"
        return _FakeReadResult("full page content")


class TestMCPGenericAdapter:
    def _adapter(self):
        from connectors.adapters.mcp_generic import MCPConnectorAdapter

        adapter = MCPConnectorAdapter("mcp-notion", {}, {"server_name": "claude_ai_Notion"})

        @asynccontextmanager
        async def factory():
            yield _FakeSession()

        adapter.session_factory = factory
        return adapter

    @pytest.mark.asyncio
    async def test_list_resources(self):
        resources, cursor = await self._adapter().list_resources()
        assert cursor is None
        assert resources[0].external_id == "notion://page-1"
        assert resources[0].title == "Q3 Plan"

    @pytest.mark.asyncio
    async def test_fetch(self):
        text = await self._adapter().fetch(ConnectorResource(external_id="notion://page-1"))
        assert text == "full page content"

    def test_retrieval_handle_uses_server_name(self):
        handle = self._adapter().retrieval_handle(ConnectorResource(external_id="notion://page-1"))
        assert handle["mcp_server"] == "claude_ai_Notion"
        assert handle["tool"] == "read_resource"
        assert handle["args"] == {"uri": "notion://page-1"}
