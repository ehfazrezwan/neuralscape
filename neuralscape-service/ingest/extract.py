"""Convert an uploaded file's raw bytes into ingestible plain text / Markdown.

Tiered strategy (see the ingest design doc):

1. **Plain** (``.md/.markdown/.txt/.text/.rst``) — decoded directly; never
   touches an external parser.
2. **Rich** (PDF, MS Office ``docx/xlsx/pptx``, HTML, EPUB, images) — converted
   to Markdown by the **``docling-serve`` container** (the preferred, AI-grade
   path), with an in-process **MarkItDown** fallback when Docling is disabled or
   unreachable, so an upload never hard-fails on a transient outage.
3. **Unknown extension** — best-effort UTF-8 decode; binary that won't decode
   raises :class:`UnsupportedFile`.

Everything here returns text that the existing ``ingest_document`` pipeline then
chunks into passages + distils into graph facts.
"""

from __future__ import annotations

import io
import logging
import mimetypes
import os

logger = logging.getLogger(__name__)

# Read directly — already text, conversion would only add noise.
PLAIN_EXTS = {".md", ".markdown", ".txt", ".text", ".rst", ".log"}

# Sent through Docling (preferred) / MarkItDown (fallback).
RICH_EXTS = {
    ".pdf",
    ".docx", ".doc",
    ".xlsx", ".xls",
    ".pptx", ".ppt",
    ".html", ".htm",
    ".epub",
    ".csv",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif",
}


class UnsupportedFile(Exception):
    """Raised when a file can't be turned into text by any available parser."""


def _ext(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


def _decode_text(data: bytes) -> str:
    """Decode bytes as UTF-8, falling back to latin-1 (never fails on bytes)."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def _looks_textual(data: bytes) -> bool:
    """Heuristic: does this decode as UTF-8 without NUL bytes (i.e. not binary)?"""
    if b"\x00" in data[:8192]:
        return False
    try:
        data[:8192].decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _docling_convert(data: bytes, filename: str, settings) -> str | None:
    """Convert via docling-serve. Returns Markdown, or None if unavailable/failed.

    POSTs the file to ``{docling_url}/v1/convert/file`` requesting Markdown and
    reads ``document.md_content`` from the response. Any error (disabled, network,
    non-200, empty result) returns None so the caller falls back to MarkItDown.
    """
    if not settings.docling_enabled or not settings.docling_url:
        return None

    import httpx

    mimetype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    url = settings.docling_url.rstrip("/") + "/v1/convert/file"
    try:
        with httpx.Client(timeout=settings.docling_timeout_s) as client:
            resp = client.post(
                url,
                files={"files": (os.path.basename(filename), data, mimetype)},
                data={"to_formats": "md"},
            )
            resp.raise_for_status()
            payload = resp.json()
    except Exception as e:  # noqa: BLE001 — any failure → fallback path
        logger.warning("Docling convert failed for %s (%s); falling back", filename, e)
        return None

    # Single-file responses use `document`; some builds wrap in `documents`.
    doc = payload.get("document")
    if doc is None:
        docs = payload.get("documents") or []
        doc = docs[0] if docs else {}
    md = (doc or {}).get("md_content") or (doc or {}).get("text_content")
    if not md or not md.strip():
        logger.warning("Docling returned no content for %s; falling back", filename)
        return None
    return md


def _markitdown_convert(data: bytes, ext: str) -> str | None:
    """In-process fallback conversion via MarkItDown. Returns text or None."""
    try:
        from markitdown import MarkItDown
    except ImportError:
        logger.warning("markitdown not installed; cannot parse %s file", ext)
        return None
    try:
        md = MarkItDown(enable_plugins=False)
        result = md.convert_stream(io.BytesIO(data), file_extension=ext or None)
    except Exception as e:  # noqa: BLE001
        logger.warning("MarkItDown failed for %s file: %s", ext, e)
        return None
    text = getattr(result, "text_content", None) or getattr(result, "markdown", None)
    return text if text and text.strip() else None


def extract_text(filename: str, data: bytes, settings) -> tuple[str, str]:
    """Turn an uploaded file's bytes into text. Returns ``(text, doc_type)``.

    ``doc_type`` records which tier produced the text (``"plain"``,
    ``"docling"``, ``"markitdown"``, or ``"decoded"``) for observability.

    Raises:
        UnsupportedFile: If the file is empty, or no parser can extract text.
    """
    if not data:
        raise UnsupportedFile(f"'{filename}' is empty")

    ext = _ext(filename)

    if ext in PLAIN_EXTS:
        return _decode_text(data), "plain"

    if ext in RICH_EXTS or ext == "":
        md = _docling_convert(data, filename, settings)
        if md is not None:
            return md, "docling"
        md = _markitdown_convert(data, ext)
        if md is not None:
            return md, "markitdown"
        # No parser succeeded — last resort: if the bytes are actually text
        # (a .csv/.html/extensionless plain file that both parsers choked on),
        # decode directly so uploads never hard-fail. Otherwise give up.
        if _looks_textual(data):
            return _decode_text(data), "decoded"
        raise UnsupportedFile(
            f"Could not extract text from '{filename}' "
            f"(Docling unavailable and MarkItDown could not parse it)"
        )

    # Unknown extension: accept it only if it's genuinely textual.
    if _looks_textual(data):
        return _decode_text(data), "decoded"
    raise UnsupportedFile(f"Unsupported file type '{ext}' for '{filename}'")
