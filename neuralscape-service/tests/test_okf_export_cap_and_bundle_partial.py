"""OKF export is never silently cut; OKF bundle import reports per-concept failures.

* Export: a pool larger than the cap used to stop at 5000 rows with no marker.
  Pools are now paged to completion up to the cap, and a pool that hits the cap
  is reported (exact exported/total counts, response headers, the bundle's own
  ``log.md``).
* Import: ``ingest_okf_bundle`` swallowed per-concept failures and the worker
  job reported success. It now returns ``failed_concepts`` and a
  ``completed`` / ``partial`` / ``failed`` status, and the worker job fails when
  nothing could be ingested.
"""

import base64
import io
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from okf import export as okf_export
from okf.export import collect_memories, collect_memories_with_info, export_bundle


class _Pt:
    def __init__(self, pid, payload):
        self.id = pid
        self.payload = payload


class _PagingQdrant:
    """scroll() honours limit/offset like Qdrant; count() counts the filtered pool."""

    def __init__(self, n, *, user="alice"):
        self.points = [
            _Pt(f"id-{i:06d}", {
                "data": f"fact number {i}", "user_id": user, "created_at": "2026-07-01T00:00:00Z",
                "metadata": {"category": "decision", "visibility": "private", "scope": "global"},
            })
            for i in range(n)
        ]
        self.scrolls = 0

    @staticmethod
    def _get(payload, key):
        if key.startswith("metadata."):
            return (payload.get("metadata") or {}).get(key[len("metadata."):])
        return payload.get(key)

    def _match(self, flt):
        def ok(p):
            return all(self._get(p.payload, c.key) == c.match.value for c in flt.must or []) and not any(
                self._get(p.payload, c.key) == c.match.value for c in flt.must_not or []
            )
        return [p for p in self.points if ok(p)]

    def scroll(self, collection_name, scroll_filter, limit, offset, with_payload, with_vectors=False):
        self.scrolls += 1
        pool = self._match(scroll_filter)
        start = int(offset or 0)
        page = pool[start:start + limit]
        nxt = start + limit if start + limit < len(pool) else None
        return page, nxt

    def count(self, collection_name, count_filter, exact=True):
        return SimpleNamespace(count=len(self._match(count_filter)))


def _service(n, **kw):
    client = _PagingQdrant(n, **kw)
    vs = SimpleNamespace(client=client)
    svc = SimpleNamespace(_get_memory=lambda: SimpleNamespace(vector_store=vs))
    return svc, client


class TestExportIsNeverSilentlyCut:
    def test_pool_beyond_the_old_5000_cap_is_exported_whole(self):
        svc, client = _service(5400)
        rows, info = collect_memories_with_info(svc, user_id="alice", visibility="private")
        assert len(rows) == 5400 and info == {"truncated": False, "pools": {}}
        assert client.scrolls >= 11  # paged in bounded batches, not one big read

    def test_a_pool_over_the_cap_reports_exact_counts(self):
        svc, _ = _service(12)
        rows, info = collect_memories_with_info(svc, user_id="alice", visibility="private", limit=5)
        assert len(rows) == 5
        assert info == {"truncated": True, "pools": {"personal": {"exported": 5, "total": 12}}}

    def test_a_pool_exactly_at_the_cap_is_not_truncated(self):
        svc, _ = _service(5)
        rows, info = collect_memories_with_info(svc, user_id="alice", visibility="private", limit=5)
        assert len(rows) == 5 and info["truncated"] is False

    def test_collect_memories_keeps_returning_just_the_rows(self):
        svc, _ = _service(7)
        assert len(collect_memories(svc, user_id="alice", visibility="private", limit=3)) == 3

    def test_export_bundle_says_so_in_stats_and_in_the_bundle_itself(self):
        svc, _ = _service(12)
        data, stats = export_bundle(svc, user_id="alice", visibility="private", limit=5)
        assert stats["truncated"] is True
        assert stats["truncated_pools"] == {"personal": {"exported": 5, "total": 12}}
        log = zipfile.ZipFile(io.BytesIO(data)).read("log.md").decode()
        assert "INCOMPLETE" in log and "personal pool 5 of 12" in log

    def test_a_complete_export_carries_no_incomplete_marker(self):
        svc, _ = _service(3)
        data, stats = export_bundle(svc, user_id="alice", visibility="private")
        assert stats["truncated"] is False
        assert "INCOMPLETE" not in zipfile.ZipFile(io.BytesIO(data)).read("log.md").decode()

    def test_the_cap_is_a_setting_with_a_default_and_an_override(self, monkeypatch):
        from config import Settings, settings

        assert Settings(_env_file=None).okf_export_max_memories == 50_000
        assert okf_export.export_cap() == settings.okf_export_max_memories
        monkeypatch.setenv("OKF_EXPORT_MAX_MEMORIES", "7")
        assert Settings(_env_file=None).okf_export_max_memories == 7
        monkeypatch.setenv("OKF_EXPORT_MAX_MEMORIES", "0")
        with pytest.raises(ValueError):
            Settings(_env_file=None)

    def test_the_setting_is_what_an_export_reads(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "okf_export_max_memories", 5)
        svc, _ = _service(12)
        _, stats = export_bundle(svc, user_id="alice", visibility="private")  # no explicit limit
        assert stats["concepts"] == 5 and stats["truncated"] is True
        assert stats["truncated_pools"] == {"personal": {"exported": 5, "total": 12}}


class TestCountFailureNeverLooksComplete:
    def test_a_failing_count_reports_truncation_with_an_unknown_total(self):
        svc, client = _service(12)

        def boom(**kw):
            raise RuntimeError("qdrant count timeout")

        client.count = boom
        rows, info = collect_memories_with_info(svc, user_id="alice", visibility="private", limit=5)
        assert len(rows) == 5
        assert info == {"truncated": True, "pools": {"personal": {"exported": 5, "total": None}}}

    def test_export_bundle_still_says_incomplete_and_the_route_says_unknown(self):
        svc, client = _service(12)
        client.count = lambda **kw: (_ for _ in ()).throw(RuntimeError("down"))
        data, stats = export_bundle(svc, user_id="alice", visibility="private", limit=5)
        assert stats["truncated"] is True
        log = zipfile.ZipFile(io.BytesIO(data)).read("log.md").decode()
        assert "INCOMPLETE" in log and "more than 5" in log

    def test_route_header_total_is_unknown(self, monkeypatch):
        import main
        from fastapi.testclient import TestClient

        monkeypatch.setattr(main, "_service", MagicMock())
        monkeypatch.setattr(okf_export, "export_bundle", lambda *a, **k: (b"PK", {
            "concepts": 5, "files": 9, "truncated": True,
            "truncated_pools": {"personal": {"exported": 5, "total": None}},
        }))
        resp = TestClient(main.app).get("/v1/export/okf")
        assert resp.headers["X-OKF-Truncated"] == "true" and resp.headers["X-OKF-Personal-Total"] == "unknown"


class TestExportRoute:
    @pytest.fixture
    def client(self, monkeypatch):
        import main
        from fastapi.testclient import TestClient

        monkeypatch.setattr(main, "_service", MagicMock())
        return TestClient(main.app, raise_server_exceptions=False)

    def _call(self, client, monkeypatch, stats):
        monkeypatch.setattr(okf_export, "export_bundle", lambda *a, **k: (b"PK", stats))
        return client.get("/v1/export/okf")

    def test_headers_flag_a_truncated_export_with_counts(self, client, monkeypatch):
        resp = self._call(client, monkeypatch, {
            "concepts": 5, "files": 9, "truncated": True,
            "truncated_pools": {"personal": {"exported": 5, "total": 12}},
        })
        assert resp.status_code == 200
        assert resp.headers["X-OKF-Truncated"] == "true"
        assert resp.headers["X-OKF-Personal-Exported"] == "5" and resp.headers["X-OKF-Personal-Total"] == "12"

    def test_a_complete_export_says_false(self, client, monkeypatch):
        resp = self._call(client, monkeypatch, {"concepts": 2, "files": 4, "truncated": False, "truncated_pools": {}})
        assert resp.status_code == 200 and resp.headers["X-OKF-Truncated"] == "false"
        assert "X-OKF-Personal-Total" not in resp.headers

    def test_the_export_scroll_runs_off_the_event_loop(self):
        import inspect

        import main

        assert "asyncio.to_thread" in inspect.getsource(main.v1_export_okf)


# ── bundle import: per-concept failures ─────────────────────────────

FILES = {
    "index.md": "---\nokf_version: 0.1\n---\n",
    "a.md": "---\ntype: Playbook\ntitle: Alpha\n---\n\nAlpha body text.\n",
    "b.md": "---\ntype: Topic\ntitle: Beta\n---\n\nBeta body text.\n",
    "c.md": "---\ntype: Topic\ntitle: Gamma\n---\n\nGamma body text.\n",
}


def _patch_ingest(monkeypatch, fail_when):
    import ingest.pipeline as pipeline

    def fake(service, doc):
        if fail_when(doc.content):
            raise RuntimeError("embedding backend unavailable for this concept")
        return {"passages": 1, "facts": 1, "memory_ids": [f"m-{abs(hash(doc.content)) % 1000}"], "graph_jobs": []}

    monkeypatch.setattr(pipeline, "ingest_document", fake)


def _run(files=FILES):
    from ingest.okf_bundle import ingest_okf_bundle

    return ingest_okf_bundle(MagicMock(), files=files, bundle_uri="b.zip", user_id="alice", llm_call=lambda p: "{}")


class TestBundleImportReportsFailures:
    def test_all_concepts_ingested_is_completed(self, monkeypatch):
        _patch_ingest(monkeypatch, lambda c: False)
        out = _run()
        assert out["status"] == "completed" and out["failed"] == 0 and out["failed_concepts"] == []

    def test_one_failing_concept_is_partial_and_named(self, monkeypatch):
        _patch_ingest(monkeypatch, lambda c: "Beta" in c)
        out = _run()
        assert out["status"] == "partial"
        assert out["concepts"] == 2 and out["failed"] == 1
        (failure,) = out["failed_concepts"]
        assert failure["concept_id"] == "b"
        assert failure["error"] == "RuntimeError: concept ingest failed"  # class + fixed text, never the message

    def test_swallowed_store_failures_on_the_real_ingest_path_are_reported(self):
        """Real ingest_document (not mocked): store_raw raising is swallowed per
        chunk inside it; the bundle must still see and report it."""
        from ingest.okf_bundle import ingest_okf_bundle

        svc = MagicMock()

        def store_raw(*a, **kw):
            if "Beta" in kw["content"]:
                raise RuntimeError("store rejected this concept")
            return [MagicMock(id="m-ok")]

        svc.store_raw.side_effect = store_raw
        out = ingest_okf_bundle(
            svc, files=FILES, bundle_uri="b.zip", user_id="alice",
            extract_facts=False, llm_call=lambda p: "{}",
        )
        assert out["status"] == "partial"
        assert out["concepts"] == 2 and out["failed"] == 1
        assert [f["concept_id"] for f in out["failed_concepts"]] == ["b"]
        (fc,) = out["failed_concepts"]
        assert fc["stores_failed"] >= 1 and fc["stores_succeeded"] == 0

        # a concept where some chunks store and some fail reports both numbers
        calls = {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("one chunk rejected")
            return [MagicMock(id=f"m{calls['n']}")]

        svc.store_raw.side_effect = flaky
        long_body = "\n\n".join(f"Paragraph {i} " + "word " * 300 for i in range(6))
        out = ingest_okf_bundle(
            svc, files={"index.md": FILES["index.md"],
                        "z.md": f"---\ntype: Topic\ntitle: Z\n---\n\n{long_body}\n"},
            bundle_uri="b.zip", user_id="alice", extract_facts=False, llm_call=lambda p: "{}",
        )
        (fz,) = out["failed_concepts"]
        assert fz["stores_failed"] == 1 and fz["stores_succeeded"] == calls["n"] - 1 >= 1
        assert out["memory_ids"], "what did store is kept"

        svc.store_raw.side_effect = RuntimeError("down")
        out = ingest_okf_bundle(
            svc, files=FILES, bundle_uri="b.zip", user_id="alice",
            extract_facts=False, llm_call=lambda p: "{}",
        )
        assert out["status"] == "failed" and out["concepts"] == 0 and out["failed"] == 3

    def test_every_concept_failing_is_failed(self, monkeypatch):
        _patch_ingest(monkeypatch, lambda c: True)
        out = _run()
        assert out["status"] == "failed" and out["concepts"] == 0 and out["failed"] == 3

    def test_empty_concepts_are_not_failures(self, monkeypatch):
        _patch_ingest(monkeypatch, lambda c: False)
        out = _run({**FILES, "d.md": "---\ntype: Topic\ntitle: Empty\n---\n\n"})
        assert out["status"] == "completed" and out["failed"] == 0

    def test_failure_report_never_carries_exception_text(self, monkeypatch):
        import ingest.pipeline as pipeline

        def boom(s, d):
            raise RuntimeError(
                "connect to http://user:hunter2@qdrant.internal:6333/collections failed; "
                "Authorization: Bearer abc.def.ghi api_key=sk-12345 password: p4ss"
            )

        monkeypatch.setattr(pipeline, "ingest_document", boom)
        err = _run()["failed_concepts"][0]["error"]
        for leaked in ("hunter2", "qdrant.internal", "abc.def.ghi", "sk-12345", "p4ss", "http"):
            assert leaked not in err, err
        assert err == "RuntimeError: concept ingest failed"

    def test_error_text_is_bounded(self, monkeypatch):
        import ingest.pipeline as pipeline

        monkeypatch.setattr(pipeline, "ingest_document", lambda s, d: (_ for _ in ()).throw(RuntimeError("x" * 5000)))
        assert len(_run()["failed_concepts"][0]["error"]) < 130


def _zip(files) -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return base64.b64encode(buf.getvalue()).decode()


class TestWorkerJobSurfacesTheStatus:
    @pytest.mark.asyncio
    async def _job(self, monkeypatch, fail_when):
        import worker

        _patch_ingest(monkeypatch, fail_when)
        monkeypatch.setattr("ingest.okf_bundle.default_type_llm", lambda service: (lambda p: "{}"))
        ctx = {"service": MagicMock(), "extension_registry": None, "redis": AsyncMock()}
        return await worker.process_ingest_okf_bundle(
            ctx, {"filename": "b.zip", "data_b64": _zip(FILES), "user_id": "alice", "options": {}}
        )

    @pytest.mark.asyncio
    async def test_partial_import_is_reported_in_the_job_result(self, monkeypatch):
        out = await self._job(monkeypatch, lambda c: "Beta" in c)
        assert out["status"] == "partial" and out["failed"] == 1
        assert [f["concept_id"] for f in out["failed_concepts"]] == ["b"]

    @pytest.mark.asyncio
    async def test_a_fully_failed_import_fails_the_job(self, monkeypatch):
        with pytest.raises(RuntimeError, match="all 3 concepts failed"):
            await self._job(monkeypatch, lambda c: True)

    @pytest.mark.asyncio
    async def test_a_clean_import_is_completed(self, monkeypatch):
        out = await self._job(monkeypatch, lambda c: False)
        assert out["status"] == "completed" and out["failed"] == 0
