"""Visual setup exemplars with ``EXEMPLAR_STORE_ENABLED`` + ``DOCLING_EXTRACT_IMAGES`` on.

Drives the real ingest worker function ``process_ingest_file`` for a trading book
upload (``adapter=trading_strategy``) against a real Qdrant collection, a real
exemplar store directory, and the real download route.

What is real: the worker job, ``extract_text_and_images`` (including figure
harvesting from the converter's response), the chunking/passage pipeline and
``store_raw`` with real embeddings (``llm_key``), ``ingest_exemplar`` (content-
addressed file store, memory shape, source_ref), the owner-scoped
``GET /v1/ingest/exemplars/{image_id}`` route with real per-user tokens.

What is stubbed, at its network boundary only: the Docling sidecar's HTTP
response (``ingest.extract._docling_post``: the sidecar is not part of the gate's
stores, so the converter output is a Markdown body with one embedded PNG, exactly
the shape docling-serve returns with ``image_export_mode=embedded``) and the
multimodal vision model (``adapters.trading.exemplars._vision_client``, which in
production is the LLM gateway). With the flags off the same upload stores no
exemplar and the route 404s.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import base64
import json
import struct
import zlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

SECRET = "exemplar-it-token-secret"


def _png(rgb: tuple[int, int, int]) -> bytes:
    """A valid 1x1 PNG of the given colour (distinct colours = distinct image hashes)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"\x00" + bytes(rgb)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class _FakeRedis:
    def __init__(self):
        self.jobs: list[tuple] = []

    async def enqueue_job(self, *args, **kwargs):
        self.jobs.append((args, kwargs))
        return SimpleNamespace(job_id=f"job-{len(self.jobs)}")


class _StubVision:
    """OpenAI-shaped client: returns a fixed structured description."""

    def __init__(self):
        self.calls = 0
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls += 1
                desc = {
                    "setup_name": "Kangaroo Tail", "direction": "bullish",
                    "visual_description": "Long lower tail rejecting a support zone with room to the left.",
                    "key_levels": None, "chart_context": "H4", "caption": "Figure 3.1",
                }
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(desc)))])

        self.chat = SimpleNamespace(completions=_Completions())


@pytest.fixture
def env(flags, stores, llm_key, unique_ns, tmp_path, monkeypatch):
    from qdrant_client import QdrantClient

    flags(
        qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url,
        exemplar_store_dir=str(tmp_path / "exemplars"),
        neuralscape_user_token_secret=SECRET, neuralscape_api_key="",
    )
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    svc = MemoryService()
    svc._get_memory()

    png = _png((200, 30, 30))
    md = (
        "# The Kangaroo Tail\n\nA kangaroo tail is a reversal bar with a long tail.\n\n"
        f"![Image](data:image/png;base64,{base64.b64encode(png).decode()})\n\n"
        "Enter on a break of the bar's body.\n"
    )
    monkeypatch.setattr(
        "ingest.extract._docling_post",
        lambda data, filename, settings, embed_images=False: {"document": {"md_content": md}},
    )
    vision = _StubVision()
    monkeypatch.setattr("adapters.trading.exemplars._vision_client", lambda settings: vision)
    import main

    monkeypatch.setattr(main, "_service", svc)
    return SimpleNamespace(svc=svc, png=png, vision=vision, ctx={"service": svc, "redis": _FakeRedis()},
                           user=unique_ns.user_id, tmp=tmp_path)


def _upload(env, *, user=None):
    import worker

    payload = {
        "filename": "kangaroo-book.pdf",
        "source_ref": {"connector_id": "file_upload", "connector_type": "file_upload",
                       "external_id": "kangaroo-book-sha", "title": "Kangaroo Book"},
        "options": {"adapter": "trading_strategy", "extract_facts": False, "tags": ["strategy:kangaroo"],
                    "scope": "global"},
        "user_id": user or env.user,
        "data_b64": base64.b64encode(b"%PDF-1.4 stub").decode(),
    }
    return asyncio.run(worker.process_ingest_file(env.ctx, payload))


def _exemplar_rows(env, uid):
    return [r for r in env.svc._scroll_all_user_memories(uid)
            if (r["payload"].get("metadata") or {}).get("category") == "setup"]


def _bearer(uid):
    from tokens import issue_user_token

    return {"Authorization": f"Bearer {issue_user_token(uid, SECRET, 3600)}"}


def test_flags_on_the_figure_is_stored_described_indexed_and_downloadable(flags, env):
    from adapters.trading.exemplars import image_hash

    flags(exemplar_store_enabled=True, docling_extract_images=True)
    out = _upload(env)

    assert out["doc_type"] == "docling" and len(out["exemplars"]) == 1, out
    summary = out["exemplars"][0]
    assert summary["described"] is True and summary["setup_name"] == "Kangaroo Tail"
    assert env.vision.calls == 1

    # the memory: setup category, tagged, body is the description, source_ref points at the stored image
    (row,) = _exemplar_rows(env, env.user)
    meta = row["payload"]["metadata"]
    assert "visual_exemplar" in (meta.get("tags") or []) and "strategy:kangaroo" in meta["tags"]
    assert "Kangaroo Tail" in row["payload"]["data"] and "Visual exemplar" in row["payload"]["data"]
    image_id = image_hash(env.png)
    assert meta["source_ref"]["external_id"] == image_id
    stored = meta["source_ref"]["stored_path"]
    assert stored.startswith("file://") and (env.tmp / "exemplars").exists()

    # the stored bytes are the figure
    from adapters.trading.exemplars import read_exemplar_image
    from config import settings

    assert read_exemplar_image(stored, settings) == env.png

    # the graph job for the exemplar was handed to the graph queue (not run inline)
    queued = [a for a, k in env.ctx["redis"].jobs if a and a[0] == "process_graph_enrichment"]
    assert any(a[1] == row["id"] for a in queued)

    # the owner downloads it; another user and an unknown id get 404 (no oracle)
    app_client = TestClient(__import__("main").app)
    ok = app_client.get(f"/v1/ingest/exemplars/{image_id}", headers=_bearer(env.user))
    assert ok.status_code == 200 and ok.content == env.png and ok.headers["content-type"] == "image/png"
    assert app_client.get(f"/v1/ingest/exemplars/{image_id}", headers=_bearer(env.user + "-other")).status_code == 404
    assert app_client.get("/v1/ingest/exemplars/0123456789abcdef", headers=_bearer(env.user)).status_code == 404


def test_reingesting_the_same_book_does_not_duplicate_the_exemplar(flags, env):
    flags(exemplar_store_enabled=True, docling_extract_images=True)
    _upload(env)
    again = _upload(env)
    assert again["exemplars"][0].get("deduped") is True
    assert len(_exemplar_rows(env, env.user)) == 1
    assert env.vision.calls == 1, "the figure must not be vision-described twice"


def test_flags_off_no_exemplar_is_stored_and_the_route_404s(flags, env):
    from adapters.trading.exemplars import image_hash

    flags(exemplar_store_enabled=False, docling_extract_images=False)
    out = _upload(env)
    assert "exemplars" not in out
    assert _exemplar_rows(env, env.user) == []
    assert env.vision.calls == 0 and not (env.tmp / "exemplars").exists()
    resp = TestClient(__import__("main").app).get(
        f"/v1/ingest/exemplars/{image_hash(env.png)}", headers=_bearer(env.user))
    assert resp.status_code == 404


def test_only_one_flag_on_is_still_off(flags, env):
    flags(exemplar_store_enabled=True, docling_extract_images=False)
    out = _upload(env)
    assert "exemplars" not in out and _exemplar_rows(env, env.user) == []
