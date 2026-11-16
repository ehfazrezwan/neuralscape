"""Multipart upload helper for ``POST /v1/ingest/files`` + a zip builder."""

from __future__ import annotations

import io
import zipfile
from typing import Any

from .client import NSClient


def build_zip(members: dict[str, str | bytes]) -> bytes:
    """A zip archive in memory: ``{member path: text or bytes}``."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data if isinstance(data, bytes) else data.encode("utf-8"))
    return buf.getvalue()


def upload_files(
    client: NSClient,
    files: dict[str, bytes],
    *,
    expect: int | None = 202,
    mime: str = "application/octet-stream",
    **form: Any,
):
    """POST ``files`` ({filename: bytes}) with ``form`` fields to /v1/ingest/files.

    Booleans are sent as ``true``/``false``, ``None`` fields are dropped. When
    ``expect`` is set, asserts the status. Returns the httpx response.
    """
    data = {
        k: (str(v).lower() if isinstance(v, bool) else str(v))
        for k, v in form.items()
        if v is not None
    }
    multipart = [("files", (name, content, mime)) for name, content in files.items()]
    resp = client.post("/v1/ingest/files", files=multipart, data=data)
    if expect is not None:
        assert resp.status_code == expect, (
            f"POST /v1/ingest/files -> {resp.status_code}: {resp.text[:200]}"
        )
    return resp
