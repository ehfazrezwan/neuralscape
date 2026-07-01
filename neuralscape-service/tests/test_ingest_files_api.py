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
    saved = settings.ingest_storage_dir
    settings.ingest_storage_dir = str(tmp_path)
    settings.ingest_storage_enabled = True
    yield
    settings.ingest_storage_dir = saved


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
        assert doc["source"]["stored_path"]
        assert doc["source"]["url"].startswith("/v1/ingest/artifacts/")

    def test_project_scope_requires_project_id(self, client, monkeypatch):
        monkeypatch.setattr(main._task_manager, "enqueue_ingest_document", AsyncMock())
        resp = client.post("/v1/ingest/text", json={
            "content": "x", "scope": "project", "user_id": "alice",
        })
        assert resp.status_code == 400


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
