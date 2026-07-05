"""Native Notion connector — pulls pages via the Notion REST API.

Auth is a single bearer token (an internal-integration secret), so this
adapter is fully exercisable against a mocked ``httpx`` transport with no OAuth
dance — it's the reference native adapter.

credentials: ``{"token": "secret_xxx"}``
config (optional):
  - ``notion_version``: API version header (default "2022-06-28")
  - ``page_size``: search page size (default 100)
"""

from __future__ import annotations

import logging

import httpx

from connectors.base import ConnectorAdapter, ConnectorResource

logger = logging.getLogger(__name__)

_API_BASE = "https://api.notion.com/v1"
_DEFAULT_VERSION = "2022-06-28"
# Block types whose rich_text we treat as document content.
_TEXT_BLOCK_TYPES = {
    "paragraph", "heading_1", "heading_2", "heading_3",
    "bulleted_list_item", "numbered_list_item", "to_do",
    "toggle", "quote", "callout", "code",
}


class NotionAdapter(ConnectorAdapter):
    connector_type = "notion"

    def __init__(self, connector_id: str, credentials: dict, config: dict | None = None):
        super().__init__(connector_id, credentials, config)
        self._http: httpx.AsyncClient | None = None

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            token = self.credentials.get("token")
            if not token:
                raise ValueError("Notion connector requires credentials.token")
            self._http = httpx.AsyncClient(
                base_url=_API_BASE,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Notion-Version": self.config.get("notion_version", _DEFAULT_VERSION),
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._http

    async def list_resources(
        self, cursor: str | None = None
    ) -> tuple[list[ConnectorResource], str | None]:
        http = self._get_http()
        body: dict = {
            "filter": {"property": "object", "value": "page"},
            "page_size": self.config.get("page_size", 100),
        }
        if cursor:
            body["start_cursor"] = cursor
        resp = await http.post("/search", json=body)
        resp.raise_for_status()
        data = resp.json()

        resources: list[ConnectorResource] = []
        for page in data.get("results", []):
            pid = page.get("id")
            if not pid:
                continue
            resources.append(
                ConnectorResource(
                    external_id=pid,
                    title=_extract_title(page),
                    url=page.get("url"),
                    revision=page.get("last_edited_time"),
                    raw={},
                )
            )
        next_cursor = data.get("next_cursor") if data.get("has_more") else None
        return resources, next_cursor

    async def fetch(self, resource: ConnectorResource) -> str:
        http = self._get_http()
        texts: list[str] = []
        cursor: str | None = None
        # Paginate through the page's block children.
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            resp = await http.get(f"/blocks/{resource.external_id}/children", params=params)
            resp.raise_for_status()
            data = resp.json()
            for block in data.get("results", []):
                txt = _block_text(block)
                if txt:
                    texts.append(txt)
            if data.get("has_more") and data.get("next_cursor"):
                cursor = data["next_cursor"]
            else:
                break
        return "\n".join(texts)

    def retrieval_handle(self, resource: ConnectorResource) -> dict:
        return {
            "mcp_server": "claude_ai_Notion",
            "tool": "notion-fetch",
            "args": {"id": resource.external_id},
        }

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


def _extract_title(page: dict) -> str | None:
    """Pull the page title out of its properties (the property of type 'title')."""
    props = page.get("properties", {}) or {}
    for prop in props.values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            parts = [t.get("plain_text", "") for t in prop.get("title", [])]
            title = "".join(parts).strip()
            if title:
                return title
    return None


def _block_text(block: dict) -> str:
    """Concatenate the plain_text of a block's rich_text array, if it has one."""
    btype = block.get("type")
    if btype not in _TEXT_BLOCK_TYPES:
        return ""
    payload = block.get(btype, {}) or {}
    rich = payload.get("rich_text", []) or payload.get("text", [])
    return "".join(t.get("plain_text", "") for t in rich).strip()
