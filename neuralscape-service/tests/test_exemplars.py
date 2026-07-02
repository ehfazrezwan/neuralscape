"""Visual setup exemplars: object store, vision-describe parsing, ingest wiring.

The live multimodal vision call is validated in E2E; here the describe client is
mocked so the store + parse + memory-shaping logic is unit-testable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from adapters.trading import exemplars as ex
from ingest.extract import _data_uri_to_bytes, _walk_data_uris


def _settings(tmp_path):
    return SimpleNamespace(
        exemplar_store_dir=str(tmp_path / "exemplars"),
        exemplar_vision_model="",
        llm_gateway_enabled=True,
        llm_gateway_llm_model="gw-opus",
        llm_gateway_api_key="k",
        _gateway_openai_base=lambda: "https://gw.example/v1",
    )


PNG = b"\x89PNG\r\n\x1a\n" + b"fake image bytes"


# ── Object store ───────────────────────────────────────────────────


def test_store_is_content_addressed_and_idempotent(tmp_path):
    s = _settings(tmp_path)
    uri1 = ex.store_exemplar_image(PNG, "png", s)
    uri2 = ex.store_exemplar_image(PNG, "png", s)
    assert uri1 == uri2  # same bytes → same content-addressed URI
    assert uri1.startswith("file://")
    assert ex.read_exemplar_image(uri1, s) == PNG


def test_store_sanitizes_extension(tmp_path):
    s = _settings(tmp_path)
    uri = ex.store_exemplar_image(PNG, "../evil", s)
    assert uri.endswith(".png")  # bad ext falls back to png


def test_read_rejects_uri_outside_store(tmp_path):
    s = _settings(tmp_path)
    with pytest.raises(ValueError):
        ex.read_exemplar_image("file:///etc/passwd", s)


# ── Describe parsing ───────────────────────────────────────────────


def test_parse_description_handles_fenced_json():
    out = ex._parse_description('```json\n{"setup_name": "Kangaroo Tail"}\n```')
    assert out["setup_name"] == "Kangaroo Tail"


def test_parse_description_bad_json_is_empty():
    assert ex._parse_description("not json") == {}


class _FakeVisionClient:
    def __init__(self, content):
        payload = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: payload))


def test_describe_exemplar_image_with_injected_client(tmp_path):
    client = _FakeVisionClient('{"setup_name":"Big Shadow","direction":"bearish","visual_description":"large engulfing candle at resistance"}')
    out = ex.describe_exemplar_image(PNG, "png", _settings(tmp_path), client=client)
    assert out["setup_name"] == "Big Shadow"
    assert out["direction"] == "bearish"


# ── Ingest wiring ──────────────────────────────────────────────────


class _FakeStored:
    def __init__(self, mid):
        self.id = mid


class _RecordingService:
    def __init__(self):
        self.calls = []

    def store_raw(self, **kwargs):
        self.calls.append(kwargs)
        responses = [_FakeStored("ex-1")]
        if kwargs.get("return_created"):
            return responses, True
        return responses


def test_ingest_exemplar_stores_memory_with_provenance(tmp_path):
    svc = _RecordingService()
    client = _FakeVisionClient('{"setup_name":"Kangaroo Tail","direction":"bullish","visual_description":"long lower tail on a support zone, room to the left"}')
    ontology = {"entity_types": {"VisualExemplar": object}}
    out = ex.ingest_exemplar(
        svc,
        image_bytes=PNG,
        ext="png",
        settings=_settings(tmp_path),
        strategy_name="naked-forex-reversal",
        page_ref="Ch8 p.142",
        user_id="u1",
        describe_client=client,
        graph_ontology=ontology,
    )
    assert out["setup_name"] == "Kangaroo Tail"
    assert out["image_uri"].startswith("file://")
    assert out["described"] is True
    assert out["memory_ids"] == ["ex-1"]

    call = svc.calls[0]
    assert call["category"] == "setup"
    assert "visual_exemplar" in call["tags"]
    assert "strategy:naked-forex-reversal" in call["tags"]
    assert call["memory_kind"] == "fact"
    assert call["source_ref"]["stored_path"] == out["image_uri"]
    assert call["graph_ontology"] is ontology
    # Inline mode (default): graph write happens in store_raw, no deferred job.
    assert call["add_to_graph"] is True
    assert out["graph_job"] is None
    # The body embeds the visual description + citation (v1 text-proxy recall).
    assert "room to the left" in call["content"]
    assert "Ch8 p.142" in call["content"]


def test_ingest_exemplar_deferred_mode_returns_graph_job(tmp_path):
    svc = _RecordingService()
    client = _FakeVisionClient('{"setup_name":"Kangaroo Tail","direction":"bullish","visual_description":"long tail on a zone"}')
    out = ex.ingest_exemplar(
        svc, image_bytes=PNG, ext="png", settings=_settings(tmp_path),
        user_id="u1", describe_client=client, add_to_graph=False,
    )
    call = svc.calls[0]
    assert call["add_to_graph"] is False
    job = out["graph_job"]
    assert job is not None
    assert job["memory_id"] == "ex-1"
    assert job["user_id"] == "u1"
    # The job's source_ref keeps the image backlink for the (:Source) attach.
    assert job["source_ref"]["external_id"] == ex.image_hash(PNG)


def test_find_existing_exemplar_matches_image_hash(tmp_path):
    class _Pt:
        id = "prior-mem"

    class _Client:
        def __init__(self):
            self.filters = None

        def scroll(self, scroll_filter=None, **kw):
            self.filters = scroll_filter
            return [_Pt()], None

    class _VS:
        pass

    class _Mem:
        pass

    svc = _RecordingService()
    vs = _VS()
    vs.client = _Client()
    mem = _Mem()
    mem.vector_store = vs
    svc._memory = mem
    # Mirror MemoryService: lookups go through _get_memory() (lazy init),
    # never the raw ._memory attribute.
    svc._get_memory = lambda: svc._memory

    found = ex.find_existing_exemplar(svc, image_bytes=PNG, user_id="u1")
    assert found == "prior-mem"
    # The filter keys on the image hash + owner (not the nondeterministic body).
    keys = {c.key for c in vs.client.filters.must}
    assert keys == {"metadata.source_ref.external_id", "metadata.owner_user_id"}


def test_ingest_exemplar_degrades_when_vision_fails(tmp_path):
    svc = _RecordingService()
    client = _FakeVisionClient("garbage not json")
    out = ex.ingest_exemplar(
        svc, image_bytes=PNG, ext="png", settings=_settings(tmp_path),
        user_id="u1", describe_client=client,
    )
    # Image still stored + a minimal memory written; described=False.
    assert out["described"] is False
    assert out["image_uri"].startswith("file://")
    assert svc.calls  # a memory was still written


# ── Docling image harvesting helpers ───────────────────────────────


def test_data_uri_decode():
    import base64
    uri = "data:image/png;base64," + base64.b64encode(PNG).decode()
    decoded = _data_uri_to_bytes(uri)
    assert decoded is not None
    data, ext = decoded
    assert data == PNG and ext == "png"
    assert _data_uri_to_bytes("not a data uri") is None


def test_walk_data_uris_finds_nested_images():
    import base64
    uri = "data:image/jpeg;base64," + base64.b64encode(PNG).decode()
    blob = {"document": {"pictures": [{"image": {"uri": uri}, "prov": [{"page_no": 8}]}]}}
    found = _walk_data_uris(blob)
    assert len(found) == 1
    assert found[0][0] == uri
    assert found[0][1]["prov"][0]["page_no"] == 8
