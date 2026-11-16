"""Generic REST connector — ingest a JSON API by describing its shape in config.

For sources that expose a list endpoint of items and (optionally) a per-item
detail endpoint. Field paths are dotted (``a.b.c``). Everything is driven by
``config`` so no code change is needed per API.

config:
  - ``list_url`` (required): endpoint returning the item list
  - ``items_path``: dotted path to the array in the list response (default: root if it's a list)
  - ``id_field`` (default "id"), ``title_field``, ``url_field``, ``revision_field``
  - ``content_field``: field on the list item holding text (skip detail fetch)
  - ``item_url_template``: e.g. "https://api/items/{id}" — fetched when content isn't inline
  - ``detail_content_field``: dotted path to text in the detail response
  - ``retrieval``: optional explicit retrieval handle to stamp on memories
credentials (optional): ``{"bearer": "..."}`` or ``{"headers": {...}}``.

Every URL fetched (``list_url``, ``item_url_template``) must pass
``connectors.destination`` (public https host unless ``CONNECTOR_ALLOWED_HOSTS``).
``url_field`` is only stored as the resource's link, never fetched here.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from connectors.base import ConnectorAdapter, ConnectorResource
from connectors.destination import guarded_async_client
from connectors.stdio_policy import ConnectorConfigError

logger = logging.getLogger(__name__)


def _dig(obj: Any, path: str | None):
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


class GenericRestAdapter(ConnectorAdapter):
    connector_type = "generic_rest"

    def __init__(self, connector_id: str, credentials: dict, config: dict | None = None):
        super().__init__(connector_id, credentials, config)
        self._http: httpx.AsyncClient | None = None
        # Test seam: a fake transport (the destination checks still run).
        self._transport: httpx.AsyncBaseTransport | None = None

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            headers = dict(self.config.get("headers", {}))
            headers.update(self.credentials.get("headers", {}))
            bearer = self.credentials.get("bearer")
            if bearer:
                headers["Authorization"] = f"Bearer {bearer}"
            # Guarded client: every request (and redirect hop) is checked against the
            # destination policy and connects to the address it validated.
            # follow_redirects stays off: a 3xx is an error here, never a new target.
            self._http = guarded_async_client(
                headers=headers,
                timeout=30.0,
                follow_redirects=False,
                transport=self._transport,
            )
        return self._http

    async def _get(self, http: httpx.AsyncClient, url: str) -> httpx.Response:
        try:
            return await http.get(url)
        except ConnectorConfigError as exc:
            # Ids and the refusal category only: never the URL or headers (caller-chosen).
            logger.warning(
                "REST connector destination refused at fetch time: connector_id=%s owner=%s reason=%s",
                self.connector_id, getattr(self, "owner_user_id", None), getattr(exc, "reason", None),
            )
            raise

    async def list_resources(
        self, cursor: str | None = None
    ) -> tuple[list[ConnectorResource], str | None]:
        http = self._get_http()
        resp = await self._get(http, self.config["list_url"])
        resp.raise_for_status()
        items = _dig(resp.json(), self.config.get("items_path")) or []
        id_field = self.config.get("id_field", "id")
        resources: list[ConnectorResource] = []
        for item in items:
            ext_id = item.get(id_field)
            if ext_id is None:
                continue
            resources.append(
                ConnectorResource(
                    external_id=str(ext_id),
                    title=_dig(item, self.config.get("title_field")),
                    url=_dig(item, self.config.get("url_field")),
                    revision=_dig(item, self.config.get("revision_field")) and str(_dig(item, self.config.get("revision_field"))),
                    raw=item,
                )
            )
        return resources, None  # single-shot; add pagination per-API as needed

    async def fetch(self, resource: ConnectorResource) -> str:
        content_field = self.config.get("content_field")
        if content_field:
            inline = _dig(resource.raw, content_field)
            if inline is not None:
                return str(inline)
        template = self.config.get("item_url_template")
        if not template:
            # Nothing more to fetch; serialize the raw item as a fallback.
            return str(resource.raw)
        http = self._get_http()
        resp = await self._get(http, template.format(id=resource.external_id))
        resp.raise_for_status()
        body = resp.json()
        text = _dig(body, self.config.get("detail_content_field"))
        return str(text) if text is not None else str(body)

    def retrieval_handle(self, resource: ConnectorResource) -> dict:
        handle = dict(self.config.get("retrieval", {}))
        handle.setdefault("mcp_server", None)
        handle.setdefault("tool", None)
        handle.setdefault("args", {"url": resource.url or resource.external_id})
        return handle

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
