"""Endpoint tests for /v1/ingest/text, /v1/ingest/files, and artifact download.

The task queue is mocked (no Redis); these verify request handling, zip
expansion, artifact persistence, and that produced jobs reference a real source.
"""

import io
import zipfile
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import main
from main import app


@pytest.fixture(autouse=True)
def _tmp_storage(tmp_path):
    """Point artifact storage at a tmp dir for the duration of each test."""
    from config import settings
    saved_dir = settings.ingest_storage_dir
    saved_enabled = settings.ingest_storage_enabled
    settings.ingest_storage_dir = str(tmp_path)
    settings.ingest_storage_enabled = True
    yield
    settings.ingest_storage_dir = saved_dir
    settings.ingest_storage_enabled = saved_enabled


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


class TestIngestText:
    def test_persists_artifact_and_enqueues_with_manual_source(self, client, monkeypatch):
        enqueue = AsyncMock(return_value="ns-abc")
        monkeypatch.setattr(main._task_manager, "enqueue_ingest_document", enqueue)

        resp = client.post("/v1/ingest/text", json={
            "content": "some pasted context worth remembering",
            "title": "My Notes",
            "user_id": "alice",
            "category": "domain_knowledge",
        })
        assert resp.status_code == 202, resp.text
        enqueue.assert_awaited_once()
        doc = enqueue.await_args.args[0]
        # The produced job must reference a real, stored artifact — not sourceless.
        assert doc["source"]["connector_type"] == "manual"
        assert doc["source"]["url"].startswith("/v1/ingest/artifacts/")
        # Internal storage path must not leak into user-visible provenance.
        assert "stored_path" not in doc["source"]

    def test_project_scope_requires_project_id(self, client, monkeypatch):
        monkeypatch.setattr(main._task_manager, "enqueue_ingest_document", AsyncMock())
        resp = client.post("/v1/ingest/text", json={
            "content": "x", "scope": "project", "user_id": "alice",
        })
        assert resp.status_code == 400

    def test_invalid_scope_rejected(self, client, monkeypatch):
        monkeypatch.setattr(main._task_manager, "enqueue_ingest_document", AsyncMock())
        resp = client.post("/v1/ingest/text", json={
            "content": "x", "scope": "projcet", "user_id": "alice",  # typo
        })
        assert resp.status_code == 422  # schema validator rejects it

    def test_standard_text_ingest_requires_dictator(self, client, monkeypatch):
        from config import settings
        monkeypatch.setattr(main._task_manager, "enqueue_ingest_document", AsyncMock(return_value="ns-1"))
        monkeypatch.setattr(settings, "standards_enabled", True)
        monkeypatch.setattr(settings, "dictator_user_ids", "alice")
        base = {"content": "All PRs must be reviewed.", "visibility": "standard"}
        assert client.post("/v1/ingest/text", json={**base, "user_id": "bob"}).status_code == 403
        assert client.post("/v1/ingest/text", json={**base, "user_id": "alice"}).status_code == 202


class TestIngestFiles:
    def test_multi_file_and_zip_expansion(self, client, monkeypatch):
        enqueue = AsyncMock(side_effect=lambda p: f"ns-{p['filename']}")
        monkeypatch.setattr(main._task_manager, "enqueue_ingest_file", enqueue)

        zip_bytes = _zip({"inner1.md": b"a", "inner2.md": b"b", "__MACOSX/._x": b"j"})
        resp = client.post(
            "/v1/ingest/files",
            data={"user_id": "alice", "category": "domain_knowledge"},
            files=[
                ("files", ("plain.md", b"top-level", "text/markdown")),
                ("files", ("bundle.zip", zip_bytes, "application/zip")),
            ],
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        # plain.md + 2 zip members (macosx junk skipped) = 3.
        assert body["count"] == 3
        names = {f["filename"] for f in body["files"]}
        assert names == {"plain.md", "inner1.md", "inner2.md"}
        # Each enqueued job carries a stored_path + file-upload source_ref.
        for call in enqueue.await_args_list:
            payload = call.args[0]
            assert payload["stored_path"]
            assert payload["source_ref"]["connector_type"] == "file_upload"

    def test_page_offset_reaches_worker_options(self, client, monkeypatch):
        enqueue = AsyncMock(return_value="ns-1")
        monkeypatch.setattr(main._task_manager, "enqueue_ingest_file", enqueue)
        resp = client.post(
            "/v1/ingest/files",
            data={"user_id": "alice", "page_offset": "60"},
            files=[("files", ("slice.pdf", b"%PDF-1.4 fake", "application/pdf"))],
        )
        assert resp.status_code == 202, resp.text
        assert enqueue.await_args.args[0]["options"]["page_offset"] == 60

    def test_page_offset_default_omitted_and_negative_rejected(self, client, monkeypatch):
        enqueue = AsyncMock(return_value="ns-1")
        monkeypatch.setattr(main._task_manager, "enqueue_ingest_file", enqueue)
        ok = client.post(
            "/v1/ingest/files",
            data={"user_id": "alice"},
            files=[("files", ("doc.md", b"x", "text/markdown"))],
        )
        assert ok.status_code == 202
        # Default 0 stays out of options → job-id key unchanged for plain uploads.
        assert "page_offset" not in enqueue.await_args.args[0]["options"]
        bad = client.post(
            "/v1/ingest/files",
            data={"user_id": "alice", "page_offset": "-5"},
            files=[("files", ("doc.md", b"x", "text/markdown"))],
        )
        assert bad.status_code == 400

    def test_download_roundtrip(self, client, monkeypatch):
        monkeypatch.setattr(
            main._task_manager, "enqueue_ingest_file", AsyncMock(return_value="ns-1")
        )
        up = client.post(
            "/v1/ingest/files",
            data={"user_id": "alice"},
            files=[("files", ("doc.md", b"downloadable body", "text/markdown"))],
        )
        assert up.status_code == 202
        file_id = up.json()["files"][0]["file_id"]

        dl = client.get(f"/v1/ingest/artifacts/{file_id}", params={"user_id": "alice"})
        assert dl.status_code == 200
        assert dl.content == b"downloadable body"

    def test_download_missing_is_404(self, client):
        resp = client.get("/v1/ingest/artifacts/deadbeef", params={"user_id": "alice"})
        assert resp.status_code == 404

    def test_unknown_adapter_form_field_is_400(self, client, monkeypatch):
        # A typo'd adapter must fail loudly, not silently ingest with default.
        monkeypatch.setattr(
            main._task_manager, "enqueue_ingest_file", AsyncMock(return_value="ns-1")
        )
        resp = client.post(
            "/v1/ingest/files",
            data={"user_id": "alice", "adapter": "trading-strategy"},  # hyphen typo
            files=[("files", ("doc.md", b"x", "text/markdown"))],
        )
        assert resp.status_code == 400
        assert "Unknown adapter" in resp.json()["detail"]

    def test_known_adapter_form_field_accepted(self, client, monkeypatch):
        enqueue = AsyncMock(return_value="ns-1")
        monkeypatch.setattr(main._task_manager, "enqueue_ingest_file", enqueue)
        resp = client.post(
            "/v1/ingest/files",
            data={"user_id": "alice", "adapter": "trading_strategy"},
            files=[("files", ("doc.md", b"x", "text/markdown"))],
        )
        assert resp.status_code == 202
        assert enqueue.await_args.args[0]["options"]["adapter"] == "trading_strategy"


class TestExemplarDownload:
    def test_roundtrip_owner_scoped(self, client, monkeypatch, tmp_path):
        from config import settings as cfg
        from adapters.trading import exemplars as ex

        monkeypatch.setattr(cfg, "exemplar_store_enabled", True)
        monkeypatch.setattr(cfg, "exemplar_store_dir", str(tmp_path))
        png = b"\x89PNG\r\n\x1a\nimage-bytes"
        uri = ex.store_exemplar_image(png, "png", cfg)
        image_id = ex.image_hash(png)
        # Owner-scoped resolution: only alice's lookup resolves the URI.
        monkeypatch.setattr(
            ex, "find_exemplar_uri",
            lambda service, *, image_id, user_id: uri if user_id == "alice" else None,
        )
        ok = client.get(f"/v1/ingest/exemplars/{image_id}", params={"user_id": "alice"})
        assert ok.status_code == 200
        assert ok.content == png
        assert ok.headers["content-type"] == "image/png"
        other = client.get(f"/v1/ingest/exemplars/{image_id}", params={"user_id": "bob"})
        assert other.status_code == 404

    def test_disabled_store_is_404(self, client, monkeypatch):
        from config import settings as cfg

        monkeypatch.setattr(cfg, "exemplar_store_enabled", False)
        resp = client.get("/v1/ingest/exemplars/abc123", params={"user_id": "alice"})
        assert resp.status_code == 404

    def test_invalid_form_fields_rejected(self, client, monkeypatch):
        monkeypatch.setattr(
            main._task_manager, "enqueue_ingest_file", AsyncMock(return_value="ns-1")
        )
        f = [("files", ("a.md", b"hi", "text/markdown"))]
        # bad category
        r = client.post("/v1/ingest/files", data={"user_id": "alice", "category": "not_a_cat"}, files=f)
        assert r.status_code == 400
        # bad scope
        r = client.post("/v1/ingest/files", data={"user_id": "alice", "scope": "weird"}, files=f)
        assert r.status_code == 400
        # bad visibility
        r = client.post("/v1/ingest/files", data={"user_id": "alice", "visibility": "public"}, files=f)
        assert r.status_code == 400

    def test_total_request_cap(self, client, monkeypatch):
        from config import settings
        monkeypatch.setattr(
            main._task_manager, "enqueue_ingest_file", AsyncMock(return_value="ns-1")
        )
        monkeypatch.setattr(settings, "ingest_max_request_mb", 1)  # 1 MB total
        # Two 700KB files individually pass the per-file cap but together exceed 1MB.
        blob = b"x" * (700 * 1024)
        resp = client.post(
            "/v1/ingest/files",
            data={"user_id": "alice"},
            files=[
                ("files", ("a.md", blob, "text/markdown")),
                ("files", ("b.md", blob, "text/markdown")),
            ],
        )
        assert resp.status_code == 413

    def test_standard_ingest_requires_dictator(self, client, monkeypatch):
        from config import settings
        monkeypatch.setattr(main._task_manager, "enqueue_ingest_file", AsyncMock(return_value="ns-1"))
        monkeypatch.setattr(settings, "standards_enabled", True)
        monkeypatch.setattr(settings, "dictator_user_ids", "alice")
        f = [("files", ("std.md", b"the rule", "text/markdown"))]
        # non-dictator → 403 (rejected synchronously, no lost jobs)
        r = client.post("/v1/ingest/files", data={"user_id": "bob", "visibility": "standard"}, files=f)
        assert r.status_code == 403
        # dictator → accepted
        r = client.post("/v1/ingest/files", data={"user_id": "alice", "visibility": "standard"}, files=f)
        assert r.status_code == 202

    def test_standard_ingest_rejected_when_tier_disabled(self, client, monkeypatch):
        from config import settings
        monkeypatch.setattr(main._task_manager, "enqueue_ingest_file", AsyncMock(return_value="ns-1"))
        monkeypatch.setattr(settings, "standards_enabled", False)
        monkeypatch.setattr(settings, "dictator_user_ids", "alice")
        f = [("files", ("std.md", b"the rule", "text/markdown"))]
        r = client.post("/v1/ingest/files", data={"user_id": "alice", "visibility": "standard"}, files=f)
        assert r.status_code == 403

    def test_oversize_file_rejected(self, client, monkeypatch):
        from config import settings
        monkeypatch.setattr(
            main._task_manager, "enqueue_ingest_file", AsyncMock(return_value="ns-1")
        )
        big = b"x" * (settings.ingest_max_file_mb * 1024 * 1024 + 1)
        resp = client.post(
            "/v1/ingest/files",
            data={"user_id": "alice"},
            files=[("files", ("big.md", big, "text/markdown"))],
        )
        assert resp.status_code == 413
