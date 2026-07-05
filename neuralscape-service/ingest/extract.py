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
import re

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


def _docling_post(data: bytes, filename: str, settings, embed_images: bool = False) -> dict | None:
    """POST a file to docling-serve; return the parsed JSON payload or None.

    One shared round-trip for both the text and image paths — a book PDF takes
    minutes to parse, so callers that want both must not convert twice. With
    ``embed_images`` the response carries figures as base64 data-URIs. Any error
    (disabled, network, non-200) returns None so callers fall back.
    """
    if not settings.docling_enabled or not settings.docling_url:
        return None

    import httpx

    mimetype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    url = settings.docling_url.rstrip("/") + "/v1/convert/file"
    form: dict = {"to_formats": "md"}
    if embed_images:
        form["image_export_mode"] = "embedded"
    try:
        with httpx.Client(timeout=settings.docling_timeout_s) as client:
            resp = client.post(
                url,
                files={"files": (os.path.basename(filename), data, mimetype)},
                data=form,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:  # noqa: BLE001 — any failure → fallback path
        logger.warning("Docling convert failed for %s (%s); falling back", filename, e)
        return None


def _md_from_payload(payload: dict) -> str | None:
    """Pull the Markdown text out of a docling-serve response payload."""
    # Single-file responses use `document`; some builds wrap in `documents`.
    doc = payload.get("document")
    if doc is None:
        docs = payload.get("documents") or []
        doc = docs[0] if docs else {}
    md = (doc or {}).get("md_content") or (doc or {}).get("text_content")
    return md if md and md.strip() else None


def _docling_convert(data: bytes, filename: str, settings) -> str | None:
    """Convert via docling-serve. Returns Markdown, or None if unavailable/failed."""
    payload = _docling_post(data, filename, settings)
    if payload is None:
        return None
    md = _md_from_payload(payload)
    if md is None:
        logger.warning("Docling returned no content for %s; falling back", filename)
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


def _data_uri_to_bytes(uri: str) -> tuple[bytes, str] | None:
    """Decode a ``data:image/…;base64,…`` URI into ``(bytes, ext)``; None if not one."""
    import base64
    import re as _re

    m = _re.match(r"^data:image/(?P<sub>[a-zA-Z0-9.+-]+);base64,(?P<b64>.+)$", uri, _re.DOTALL)
    if not m:
        return None
    sub = m.group("sub").lower()
    ext = {"jpeg": "jpg", "svg+xml": "svg"}.get(sub, sub)
    try:
        return base64.b64decode(m.group("b64")), ext
    except Exception:  # noqa: BLE001
        return None


# An image data-URI, either standing alone or inline inside Markdown/JSON text.
# Base64 charset only — stops cleanly at the closing ')' / quote / whitespace.
_DATA_URI_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+")


def _walk_data_uris(obj) -> list[tuple[str, dict]]:
    """Recursively collect ``(data_uri, containing_dict)`` pairs from a JSON blob.

    Robust to docling-serve schema drift: we don't hard-code the pictures path,
    just find every embedded image data-URI and keep its parent object so we can
    read a nearby caption/page number when present. Handles BOTH shapes seen in
    the wild (verified live during E2E):

    - a standalone string value (a ``pictures[].image.uri`` entry), and
    - data-URIs **inlined inside a larger string** — with
      ``image_export_mode=embedded`` docling-serve embeds figures directly into
      ``md_content`` as ``![Image](data:image/png;base64,…)``.
    """
    found: list[tuple[str, dict]] = []

    def _rec(node, container):
        if isinstance(node, str):
            if "data:image/" in node:
                ctr = container if isinstance(container, dict) else {}
                for m in _DATA_URI_RE.finditer(node):
                    found.append((m.group(0), ctr))
        elif isinstance(node, dict):
            # A dict carrying provenance/caption is the picture container for its
            # whole subtree (the data-URI usually sits under a nested `image` key).
            new_container = node if ("prov" in node or "caption" in node) else container
            for v in node.values():
                _rec(v, new_container)
        elif isinstance(node, list):
            for v in node:
                _rec(v, container)

    _rec(obj, {})
    return found


def _strip_inline_image_uris(md: str) -> str:
    """Replace inline image data-URIs in Markdown with a plain placeholder.

    With ``image_export_mode=embedded`` the figures arrive base64-inlined in the
    Markdown itself; left in place they would flow into passage chunking as
    megabytes of base64 noise. The images are harvested separately — the text
    keeps only a marker where each figure sat.
    """
    return _DATA_URI_RE.sub("image", md)


def _page_ref_from(container: dict, page_offset: int = 0) -> str | None:
    """Best-effort page/caption extraction from a docling picture container.

    ``page_offset`` is added to the parsed page number — a document ingested
    as slices of a larger original (a chaptered book split to fit the ingest
    job timeout) reports slice-relative pages, and the offset restores the
    original's numbering.
    """
    prov = container.get("prov")
    page = None
    if isinstance(prov, list) and prov and isinstance(prov[0], dict):
        page = prov[0].get("page_no") or prov[0].get("page")
    caption = container.get("caption")
    if isinstance(caption, dict):
        caption = caption.get("text")
    bits = []
    if page is not None:
        try:
            page = int(page)
        except (TypeError, ValueError):
            pass  # non-numeric page marker — keep as-is
        else:
            page = page + page_offset
        bits.append(f"p.{page}")
    if isinstance(caption, str) and caption.strip():
        bits.append(caption.strip())
    return " — ".join(bits) if bits else None


def _harvest_images(payload: dict, filename: str, page_offset: int = 0) -> list[dict]:
    """Collect embedded figures from a docling payload as ``{"bytes","ext","page_ref"}``."""
    import hashlib

    images: list[dict] = []
    seen: set[bytes] = set()
    for uri, container in _walk_data_uris(payload):
        decoded = _data_uri_to_bytes(uri)
        if decoded is None:
            continue
        img_bytes, ext = decoded
        # Dedup on a real content digest — a byte-prefix would collide for
        # distinct images sharing format headers (PNG signature + IHDR),
        # silently dropping exemplars.
        digest = hashlib.sha256(img_bytes).digest()
        if digest in seen:
            continue
        seen.add(digest)
        images.append(
            {"bytes": img_bytes, "ext": ext, "page_ref": _page_ref_from(container, page_offset)}
        )
    logger.info("Docling image extraction: %d image(s) from %s", len(images), filename)
    return images


def extract_images(data: bytes, filename: str, settings, page_offset: int = 0) -> list[dict]:
    """Extract embedded figures/pictures from a document via docling-serve.

    Returns a list of ``{"bytes", "ext", "page_ref"}`` (empty when disabled,
    unavailable, or the doc has no images). Best-effort: any failure returns
    ``[]`` so image extraction never breaks a text ingest.

    NOTE: this is a standalone Docling round-trip. Callers that also need the
    document text should use :func:`extract_text_and_images` — one conversion
    for both — instead of pairing this with ``extract_text``.
    """
    if not settings.docling_extract_images:
        return []
    payload = _docling_post(data, filename, settings, embed_images=True)
    if payload is None:
        return []
    return _harvest_images(payload, filename, page_offset)


def extract_text_and_images(
    filename: str, data: bytes, settings, page_offset: int = 0
) -> tuple[str, str, list[dict]]:
    """Turn a file into text AND its embedded figures with ONE Docling conversion.

    Returns ``(text, doc_type, images)``. A book PDF takes minutes per Docling
    parse, so the text+images case must not convert twice. Fallback order
    mirrors :func:`extract_text`; images are only available on the Docling path
    (MarkItDown/decode fallbacks return whatever images the failed-but-parsed
    Docling payload yielded, usually ``[]``). ``page_offset`` rebases figure
    page refs when the file is a slice of a larger original.

    Raises:
        UnsupportedFile: same contract as :func:`extract_text`.
    """
    if not data:
        raise UnsupportedFile(f"'{filename}' is empty")

    ext = _ext(filename)
    if ext in PLAIN_EXTS:
        return _decode_text(data), "plain", []

    if ext in RICH_EXTS or ext == "":
        images: list[dict] = []
        payload = _docling_post(data, filename, settings, embed_images=True)
        if payload is not None:
            images = _harvest_images(payload, filename, page_offset)
            md = _md_from_payload(payload)
            if md is not None:
                # Embedded mode inlines figures into the Markdown as base64 —
                # strip them so passage chunking sees prose, not image bytes.
                return _strip_inline_image_uris(md), "docling", images
            logger.warning("Docling returned no content for %s; falling back", filename)
        md = _markitdown_convert(data, ext)
        if md is not None:
            return md, "markitdown", images
        if _looks_textual(data):
            return _decode_text(data), "decoded", images
        raise UnsupportedFile(
            f"Could not extract text from '{filename}' "
            f"(Docling unavailable and MarkItDown could not parse it)"
        )

    if _looks_textual(data):
        return _decode_text(data), "decoded", []
    raise UnsupportedFile(f"Unsupported file type '{ext}' for '{filename}'")


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
