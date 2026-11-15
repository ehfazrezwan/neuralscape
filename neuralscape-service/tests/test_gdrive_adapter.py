"""Google Drive adapter against a mocked httpx transport (no live Google calls)."""

import httpx
import pytest

from connectors.adapters.gdrive import GoogleDriveAdapter

CREDS = {"client_id": "cid", "client_secret": "sec", "refresh_token": "rt"}
DOC = "application/vnd.google-apps.document"


def _adapter(handler, creds=CREDS, config=None):
    a = GoogleDriveAdapter("gd", creds, config or {})
    a._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return a


def _handler(calls):
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        p = req.url.path
        if req.url.host == "oauth2.googleapis.com":
            body = dict(x.split("=", 1) for x in req.content.decode().split("&"))
            assert body["grant_type"] == "refresh_token" and body["refresh_token"] == "rt"
            return httpx.Response(200, json={"access_token": "AT"})
        assert req.headers["authorization"] == "Bearer AT"
        if p == "/drive/v3/files":
            if req.url.params.get("pageToken") == "p2":
                return httpx.Response(200, json={"files": [{"id": "f3", "name": "c.txt", "mimeType": "text/plain",
                                                            "modifiedTime": "2026-01-03T00:00:00Z"}]})
            return httpx.Response(200, json={"nextPageToken": "p2", "files": [
                {"id": "f1", "name": "a", "mimeType": DOC, "modifiedTime": "2026-01-01T00:00:00Z",
                 "webViewLink": "https://docs.google.com/f1"},
                {"id": "f2", "name": "b.md", "mimeType": "text/markdown", "modifiedTime": "2026-01-02T00:00:00Z"}]})
        if p == "/drive/v3/files/f1/export":
            assert req.url.params["mimeType"] == "text/plain"
            return httpx.Response(200, text="exported doc text")
        if p == "/drive/v3/files/f2":
            assert req.url.params["alt"] == "media"
            return httpx.Response(200, text="# plain markdown")
        return httpx.Response(404)
    return handler


@pytest.mark.asyncio
async def test_refresh_token_exchange_and_cache():
    calls = []
    a = _adapter(_handler(calls))
    try:
        assert await a._ensure_token() == "AT"
        assert await a._ensure_token() == "AT"
    finally:
        await a.aclose()
    assert len([c for c in calls if c.url.host == "oauth2.googleapis.com"]) == 1


@pytest.mark.asyncio
async def test_listing_pages_through_cursor():
    calls = []
    a = _adapter(_handler(calls), config={"page_size": 2})
    try:
        page1, cur = await a.list_resources()
        assert cur == "p2"
        assert [r.external_id for r in page1] == ["f1", "f2"]
        assert page1[0].url == "https://docs.google.com/f1"
        assert page1[0].revision == "2026-01-01T00:00:00Z"
        assert page1[0].raw["mime_type"] == DOC
        page2, cur2 = await a.list_resources(cur)
        assert cur2 is None and [r.external_id for r in page2] == ["f3"]
    finally:
        await a.aclose()
    first = [c for c in calls if c.url.path == "/drive/v3/files"][0]
    assert first.url.params["pageSize"] == "2"


@pytest.mark.asyncio
async def test_export_doc_and_download_text_file():
    a = _adapter(_handler([]))
    try:
        resources, _ = await a.list_resources()
        assert await a.fetch(resources[0]) == "exported doc text"
        assert await a.fetch(resources[1]) == "# plain markdown"
    finally:
        await a.aclose()


@pytest.mark.asyncio
async def test_auth_failure_from_token_endpoint_raises():
    def handler(req):
        return httpx.Response(400, json={"error": "invalid_grant"})
    a = _adapter(handler)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await a.list_resources()
    finally:
        await a.aclose()


@pytest.mark.asyncio
async def test_api_401_raises():
    def handler(req):
        if req.url.host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "AT"})
        return httpx.Response(401, json={"error": "unauthorized"})
    a = _adapter(handler)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await a.list_resources()
    finally:
        await a.aclose()


@pytest.mark.asyncio
async def test_missing_credentials_rejected_before_any_request():
    calls = []
    a = _adapter(_handler(calls), creds={"client_id": "cid"})
    try:
        with pytest.raises(ValueError, match="client_secret"):
            await a.list_resources()
    finally:
        await a.aclose()
    assert calls == []
