"""Google Drive connector — native OAuth2, via httpx (no google SDK dep).

CREDENTIAL-GATED: the adapter logic (token refresh, file listing, content
export) is complete and unit-testable against a mocked transport, but live
calls require a real OAuth2 refresh token and are NOT exercised in CI. Provide
``client_id`` / ``client_secret`` / ``refresh_token`` to use it for real.

credentials: ``{"client_id": "...", "client_secret": "...", "refresh_token": "..."}``
config (optional):
  - ``query``: Drive ``q`` filter (default: non-trashed Docs + plain text)
  - ``page_size`` (default 100)
"""

from __future__ import annotations

import logging

import httpx

from connectors.base import ConnectorAdapter, ConnectorResource

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_FILES_URL = "https://www.googleapis.com/drive/v3/files"
_DEFAULT_QUERY = (
    "trashed = false and ("
    "mimeType = 'application/vnd.google-apps.document' or "
    "mimeType = 'text/plain' or mimeType = 'text/markdown')"
)
# Google Workspace docs must be exported; binary/text files are downloaded.
_EXPORTABLE = {"application/vnd.google-apps.document"}


class GoogleDriveAdapter(ConnectorAdapter):
    connector_type = "google_drive"

    def __init__(self, connector_id: str, credentials: dict, config: dict | None = None):
        super().__init__(connector_id, credentials, config)
        self._http: httpx.AsyncClient | None = None
        self._access_token: str | None = None

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=60.0)
        return self._http

    async def _ensure_token(self) -> str:
        """Exchange the refresh token for a short-lived access token (cached)."""
        if self._access_token:
            return self._access_token
        for key in ("client_id", "client_secret", "refresh_token"):
            if not self.credentials.get(key):
                raise ValueError(f"Google Drive connector requires credentials.{key}")
        http = self._get_http()
        resp = await http.post(
            _TOKEN_URL,
            data={
                "client_id": self.credentials["client_id"],
                "client_secret": self.credentials["client_secret"],
                "refresh_token": self.credentials["refresh_token"],
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        self._access_token = resp.json()["access_token"]
        return self._access_token

    async def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {await self._ensure_token()}"}

    async def list_resources(
        self, cursor: str | None = None
    ) -> tuple[list[ConnectorResource], str | None]:
        http = self._get_http()
        params = {
            "q": self.config.get("query", _DEFAULT_QUERY),
            "pageSize": self.config.get("page_size", 100),
            "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink)",
        }
        if cursor:
            params["pageToken"] = cursor
        resp = await http.get(_FILES_URL, params=params, headers=await self._auth_headers())
        resp.raise_for_status()
        data = resp.json()
        resources: list[ConnectorResource] = []
        for f in data.get("files", []):
            resources.append(
                ConnectorResource(
                    external_id=f["id"],
                    title=f.get("name"),
                    url=f.get("webViewLink"),
                    revision=f.get("modifiedTime"),
                    raw={"mime_type": f.get("mimeType")},
                )
            )
        return resources, data.get("nextPageToken")

    async def fetch(self, resource: ConnectorResource) -> str:
        http = self._get_http()
        headers = await self._auth_headers()
        mime = resource.raw.get("mime_type")
        if mime in _EXPORTABLE:
            resp = await http.get(
                f"{_FILES_URL}/{resource.external_id}/export",
                params={"mimeType": "text/plain"},
                headers=headers,
            )
        else:
            resp = await http.get(
                f"{_FILES_URL}/{resource.external_id}",
                params={"alt": "media"},
                headers=headers,
            )
        resp.raise_for_status()
        return resp.text

    def retrieval_handle(self, resource: ConnectorResource) -> dict:
        return {
            "mcp_server": "claude_ai_Google_Drive",
            "tool": "get_file",
            "args": {"file_id": resource.external_id, "url": resource.url},
        }

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
