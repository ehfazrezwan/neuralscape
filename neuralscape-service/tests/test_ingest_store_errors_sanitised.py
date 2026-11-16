"""store_errors / per-concept errors leaving the process are bounded and sanitised.

A store exception can carry a credentialed store URL and document text. What
reaches callers (ingest_document return, task status, REST/MCP sync fallbacks,
OKF failed_concepts) is only ``ClassName: <fixed message>``, capped; the raw
text goes to the server log.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import main
import mcp_server
import worker
from ingest.pipeline import (
    MAX_STORE_ERRORS,
    MAX_STORE_ERRORS_TOTAL_CHARS,
    IngestDoc,
    ingest_document,
)
from schemas import MemoryResponse

SECRET_URL = "redis://user:s3cret@host:6379"
DOC_TEXT = "Quarterly revenue forecast is confidential project Nightingale"
LEAKS = ("s3cret", SECRET_URL, "user:", "host:6379", "Nightingale", "confidential")

SOURCE = {"connector_id": "c", "connector_type": "manual", "external_id": "e1"}


class _BoomError(Exception):
    pass


class _LeakyService:
    """store_raw always raises, with the URL and the document text in the message."""

    def store_raw(self, **kw):
        raise _BoomError(f"cannot reach {SECRET_URL} while storing: {kw['content']}")

    def extract_facts_only(self, *a, **kw):
        return [("domain_knowledge", DOC_TEXT)]


def _doc(content=DOC_TEXT, **over):
    kw = dict(content=content, source=dict(SOURCE), user_id="alice", index_passages=True)
    kw.update(over)
    return IngestDoc(**kw)


def _assert_clean(payload) -> None:
    blob = json.dumps(payload, default=str)
    for leak in LEAKS:
        assert leak not in blob, f"{leak!r} leaked: {blob}"


class TestIngestDocumentStoreErrors:
    def test_clean_ingest_has_empty_errors_and_exact_counts(self):
        class Ok:
            def store_raw(self, **kw):
                r = [MemoryResponse(id="m", memory=kw["content"])]
                return (r, True) if kw.get("return_created") else r

            def extract_facts_only(self, *a, **kw):
                return []

        out = ingest_document(Ok(), _doc("short body"))
        assert out["store_errors"] == []
        assert out["stores_failed"] == 0
        assert out["stores_succeeded"] == 1  # one passage, no facts

    def test_failure_shape_is_class_name_plus_fixed_message(self):
        out = ingest_document(_LeakyService(), _doc())
        # one passage store + one fact store, both failed
        assert out["store_errors"] == [
            "_BoomError: passage store failed",
            "_BoomError: fact store failed",
        ]
        assert out["stores_failed"] == 2
        assert out["stores_succeeded"] == 0
        _assert_clean(out)

    def test_raw_text_goes_to_the_log_only(self, caplog):
        with caplog.at_level("WARNING"):
            ingest_document(_LeakyService(), _doc())
        assert "s3cret" in caplog.text  # operators still see it

    def test_entry_count_and_total_length_are_capped(self):
        body = "\n\n".join(f"Paragraph {i} " + "word " * 300 for i in range(40))
        out = ingest_document(_LeakyService(), _doc(body, max_chars=400, overlap=0))
        assert out["stores_failed"] > MAX_STORE_ERRORS  # many failures happened...
        assert len(out["store_errors"]) == MAX_STORE_ERRORS  # ...few are reported
        assert sum(map(len, out["store_errors"])) <= MAX_STORE_ERRORS_TOTAL_CHARS

    def test_long_exception_class_name_is_bounded(self):
        Huge = type("E" * 500, (Exception,), {})

        class Svc(_LeakyService):
            def store_raw(self, **kw):
                raise Huge(SECRET_URL)

        out = ingest_document(Svc(), _doc())
        assert all(len(e) <= 120 for e in out["store_errors"])
        _assert_clean(out)


class TestEveryCallerPathIsClean:
    @pytest.mark.asyncio
    async def test_worker_task_status_result(self):
        ctx = {"service": _LeakyService(), "redis": MagicMock(), "extension_registry": None}
        ctx["redis"].enqueue_job = AsyncMock()
        result = await worker.process_ingest_document(ctx, {
            "content": DOC_TEXT, "source": dict(SOURCE), "user_id": "alice",
        })
        assert result["store_errors"] and result["stores_failed"] == 2
        _assert_clean(result)

    def test_rest_sync_fallback_response(self, monkeypatch):
        monkeypatch.setattr(
            main._task_manager, "enqueue_ingest_document",
            AsyncMock(side_effect=ConnectionError("redis down")),
        )
        monkeypatch.setattr(main, "_service", _LeakyService())
        client = TestClient(main.app, raise_server_exceptions=False)
        resp = client.post("/v1/ingest", json={
            "content": DOC_TEXT, "source": dict(SOURCE), "user_id": "alice",
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["store_errors"]
        _assert_clean(resp.text)

    @pytest.mark.asyncio
    async def test_mcp_sync_fallback_result(self, monkeypatch):
        tm = MagicMock()
        tm.enqueue_ingest_document = AsyncMock(side_effect=ConnectionError("redis down"))
        monkeypatch.setattr(mcp_server, "_task_manager", tm)
        monkeypatch.setattr(mcp_server, "_service", _LeakyService())
        res = await mcp_server.call_tool("ingest_document", {
            "content": DOC_TEXT, "source": dict(SOURCE), "user_id": "alice",
        })
        text = res[0].text
        data = json.loads(text)
        assert data["fallback"] == "sync" and data["result"]["store_errors"]
        _assert_clean(text)

    def test_okf_bundle_failed_concepts(self):
        from ingest.okf_bundle import ingest_okf_bundle

        files = {
            "index.md": "---\nokf_version: 0.1\n---\n",
            "a.md": f"---\ntype: Topic\ntitle: A\n---\n\n{DOC_TEXT}\n",
        }
        out = ingest_okf_bundle(
            _LeakyService(), files=files, bundle_uri="b.zip", user_id="alice",
            extract_facts=False, llm_call=lambda p: "{}",
        )
        assert out["failed"] == 1
        assert out["failed_concepts"][0]["error"] == "_BoomError: passage store failed"
        _assert_clean(out)
