"""Visual setup exemplars — object store + vision-describe + ingest.

Implements the NS side of ``VISUAL_EXEMPLARS_SPEC``: turn a chart image from a
trading book into (a) stored bytes in an object store, (b) a structured visual
description from a multimodal model, and (c) a normal ``setup``/``visual_exemplar``
memory whose body IS the description (so it embeds into Qdrant — the v1 text-proxy
recall index) plus a ``VisualExemplar`` graph node (via the trading ontology).

v1 object store = a local dir addressed by a ``file://`` URI written into
``source_ref.stored_path``; swap to S3/MinIO later by changing only the store
helpers here (call sites are URI-agnostic).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

_EXT_RE = re.compile(r"^[a-z0-9]{1,8}$")
_MIME_BY_EXT = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "gif": "image/gif", "webp": "image/webp",
}


# ── Object store (local, v1) ────────────────────────────────────────


def _exemplar_dir(settings) -> Path:
    return Path(settings.exemplar_store_dir).expanduser()


def store_exemplar_image(data: bytes, ext: str, settings) -> str:
    """Persist image bytes keyed by content hash; return a ``file://`` URI.

    Idempotent (content-addressed): re-storing identical bytes returns the same
    URI without rewriting. Atomic (temp + os.replace).
    """
    d = _exemplar_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()[:32]
    ext = (ext or "png").lower().lstrip(".")
    if not _EXT_RE.match(ext):
        ext = "png"
    path = (d / f"{digest}.{ext}").resolve()
    # Path-traversal guard: the resolved path must stay under the store dir.
    if not str(path).startswith(str(d.resolve())):
        raise ValueError("exemplar path escaped the store directory")
    if not path.exists():
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return path.as_uri()


def read_exemplar_image(uri: str, settings) -> bytes:
    """Read image bytes back from a ``file://`` exemplar URI (store-scoped)."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"unsupported exemplar URI scheme: {parsed.scheme!r}")
    path = Path(unquote(parsed.path)).resolve()
    store = _exemplar_dir(settings).resolve()
    if not str(path).startswith(str(store)):
        raise ValueError("exemplar URI escaped the store directory")
    return path.read_bytes()


def _mime_for(ext: str) -> str:
    return _MIME_BY_EXT.get((ext or "png").lower().lstrip("."), "image/png")


# ── Vision describe (multimodal, via the LLM gateway) ───────────────

EXEMPLAR_DESCRIBE_PROMPT = """\
You are a trading chart-vision analyst. This image is a labeled setup example from a trading book.
Describe it using CHECKABLE visual features (the same vocabulary you would use to read a live chart), so a later reader can match a live chart against this exemplar.

Return ONLY a JSON object:
{
  "setup_name": "the setup this image shows, e.g. 'Kangaroo Tail' (or null)",
  "direction": "bullish | bearish | null",
  "visual_description": "the structured read: candle body/tail position and proportions, where it sits relative to a support/resistance zone, 'room to the left', relative size vs neighbors, any trend context",
  "key_levels": "annotated prices if legible, else null",
  "chart_context": "timeframe/instrument if shown, else null",
  "caption": "nearby caption text if visible in the image, else null"
}
No commentary outside the JSON.
"""


def _vision_client(settings):
    """Build an OpenAI-compatible client pointed at the LLM gateway (fronts Opus 4.8)."""
    from openai import OpenAI

    if not settings.llm_gateway_enabled:
        raise RuntimeError(
            "Visual exemplar description requires the LLM gateway (multimodal). "
            "Set LLM_GATEWAY_ENABLED=true."
        )
    return OpenAI(
        base_url=settings._gateway_openai_base(),
        api_key=settings.llm_gateway_api_key,
    )


def _parse_description(text: str) -> dict:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        data = json.loads(t)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        logger.warning("exemplar describe: non-JSON response")
        return {}


def describe_exemplar_image(data: bytes, ext: str, settings, client=None) -> dict:
    """Vision-describe a setup image → structured dict. Returns {} on failure.

    ``client`` is injectable for tests; production builds one against the gateway.
    A missing/misconfigured gateway degrades to {} — the caller still stores the
    image + a minimal memory, so nothing is lost and the description can be
    backfilled once the gateway is available.
    """
    if client is None:
        try:
            client = _vision_client(settings)
        except Exception as e:  # noqa: BLE001 — no gateway ⇒ undescribed, not lost
            logger.warning("exemplar vision client unavailable: %s", e)
            return {}
    model = settings.exemplar_vision_model or settings.llm_gateway_llm_model
    b64 = base64.b64encode(data).decode()
    data_uri = f"data:{_mime_for(ext)};base64,{b64}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXEMPLAR_DESCRIBE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        )
        text = resp.choices[0].message.content
    except Exception as e:
        logger.warning("exemplar vision call failed: %s", e)
        return {}
    return _parse_description(text)


# ── Ingest an exemplar (store + describe + memory) ──────────────────


def image_hash(data: bytes) -> str:
    """Stable id for an exemplar image — sha256 prefix of the raw bytes."""
    return hashlib.sha256(data).hexdigest()[:16]


def _find_exemplar_point(service, *, external_id: str, user_id: str, with_payload: bool):
    """Scroll for one exemplar memory by image hash + owner. None on miss/error."""
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        from config import settings as core_settings

        # _get_memory() lazily initializes; the raw attribute is None on a
        # service that hasn't served a request yet (same fix as the strategy
        # synthesizer's scroll — never deref ._memory directly).
        client = service._get_memory().vector_store.client
        points, _ = client.scroll(
            collection_name=core_settings.qdrant_collection,
            scroll_filter=Filter(must=[
                FieldCondition(
                    key="metadata.source_ref.external_id",
                    match=MatchValue(value=external_id),
                ),
                FieldCondition(
                    key="metadata.owner_user_id", match=MatchValue(value=user_id)
                ),
            ]),
            limit=1,
            with_payload=with_payload,
            with_vectors=False,
        )
        return points[0] if points else None
    except Exception:  # noqa: BLE001 — lookups are best-effort
        logger.warning("exemplar lookup failed for %s", external_id, exc_info=True)
        return None


def find_existing_exemplar(service, *, image_bytes: bytes, user_id: str) -> str | None:
    """Return the memory id of an already-ingested exemplar for these image bytes.

    The vision description is nondeterministic, so re-ingesting the same book
    would produce a *different* memory body each time and slip past store_raw's
    content-hash dedup. The image bytes are the stable identity: look up an
    existing exemplar memory by ``source_ref.external_id`` (the image hash) +
    owner before describing/storing. Best-effort — errors return None (caller
    proceeds with a normal ingest).
    """
    point = _find_exemplar_point(
        service, external_id=image_hash(image_bytes), user_id=user_id, with_payload=False
    )
    return str(point.id) if point is not None else None


def find_exemplar_uri(service, *, image_id: str, user_id: str) -> str | None:
    """Resolve an exemplar image id to its stored ``file://`` URI, owner-scoped.

    Backs the ``GET /v1/ingest/exemplars/{image_id}`` download endpoint: the
    caller may only fetch images referenced by an exemplar memory THEY own —
    a miss (or another owner's image) returns None, so the endpoint 404s
    without acting as an existence oracle.
    """
    point = _find_exemplar_point(
        service, external_id=image_id, user_id=user_id, with_payload=True
    )
    if point is None:
        return None
    payload = getattr(point, "payload", None) or {}
    source_ref = (payload.get("metadata") or {}).get("source_ref") or {}
    return source_ref.get("stored_path") or None


def _exemplar_body(desc: dict, page_ref: str | None) -> str:
    """The memory body — the visual description plus a citation, embeddable text."""
    parts: list[str] = []
    name = (desc.get("setup_name") or "").strip()
    direction = (desc.get("direction") or "").strip()
    header = " ".join(p for p in (direction, name) if p) or "Setup exemplar"
    parts.append(f"Visual exemplar — {header}.")
    vis = (desc.get("visual_description") or "").strip()
    if vis:
        parts.append(vis)
    for label, key in (("Key levels", "key_levels"), ("Chart", "chart_context"), ("Caption", "caption")):
        v = (desc.get(key) or "").strip() if isinstance(desc.get(key), str) else None
        if v:
            parts.append(f"{label}: {v}")
    if page_ref:
        parts.append(f"Source: {page_ref}")
    return "\n".join(parts)


def ingest_exemplar(
    service,
    *,
    image_bytes: bytes,
    ext: str,
    settings,
    strategy_name: str | None = None,
    page_ref: str | None = None,
    user_id: str,
    project_id: str | None = None,
    scope: str = "global",
    visibility: str | None = None,
    describe_client=None,
    graph_ontology: dict | None = None,
    add_to_graph: bool = True,
) -> dict:
    """Store an exemplar image, describe it, and index it as a memory + graph node.

    Returns ``{"image_uri", "setup_name", "memory_ids", "described", "graph_job"}``.
    Best-effort on the vision step: if description fails, the image is still
    stored and a minimal memory is written so nothing is lost.

    With ``add_to_graph=True`` (default — standalone/script usage) the graph
    write runs inline using ``graph_ontology``. The ingest worker passes
    ``add_to_graph=False`` instead and enqueues the returned ``graph_job`` onto
    the graph queue, so a book's worth of exemplars can't blow the ingest job
    timeout. ``graph_job`` is ``None`` for dedup hits (already in the graph) —
    the vision description is nondeterministic, so re-ingest dedup is handled
    upstream by :func:`find_existing_exemplar`, not by the body content hash.
    """
    image_uri = store_exemplar_image(image_bytes, ext, settings)
    desc = describe_exemplar_image(image_bytes, ext, settings, client=describe_client)
    body = _exemplar_body(desc, page_ref)
    setup_name = (desc.get("setup_name") or "").strip() or None

    tags = ["visual_exemplar"]
    if strategy_name:
        tags.append(f"strategy:{strategy_name}")

    title = f"{setup_name or 'setup'} exemplar"
    if page_ref:
        title += f" ({page_ref})"
    source_ref = {
        "connector_id": "file_upload",
        "connector_type": "file_upload",
        "external_id": image_hash(image_bytes),
        "title": title,
        "stored_path": image_uri,
    }

    stored, created = service.store_raw(
        content=body,
        user_id=user_id,
        category="setup",
        scope=scope,
        project_id=project_id,
        tags=tags,
        source_type="imported",
        visibility=visibility,
        memory_kind="fact",
        source_ref=source_ref,
        add_to_graph=add_to_graph,
        graph_ontology=graph_ontology,
        return_created=True,
    )
    graph_job = None
    if created and not add_to_graph and stored:
        m = stored[0]
        graph_job = {
            "memory_id": m.id,
            "content": body,
            "user_id": user_id,
            "project_id": project_id,
            "visibility": getattr(m, "visibility", None),
            "source_ref": source_ref,
        }
    return {
        "image_uri": image_uri,
        "setup_name": setup_name,
        "memory_ids": [m.id for m in stored],
        "described": bool(desc),
        "graph_job": graph_job,
    }
