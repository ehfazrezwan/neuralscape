"""Flag-on proof for two connector-sync guarantees, through the real sync job on real Redis.

* A resource whose chunks did not all store is NOT recorded as synced: it stays eligible on the
  next sync, and the run's result carries a ``failed`` count.
* A page-based connector returns to the first page once the listing is exhausted (the stored
  cursor is reset), so a later sync sees items added at the front of the list.

Real Redis carries the encrypted connector vault, the ARQ queue, the job-result cache and the task
status. A real ``arq`` worker runs the service's own ``process_connector_sync`` job, the service's
own ``TaskManager`` enqueues it, and the connector is the service's own Notion adapter. There is no
real third-party service: the adapter's API base is pointed at a local fake endpoint inside this
test (paged ``POST /search`` and ``GET /blocks/<id>/children``). The service's real
``ingest_document`` runs; its memory service is a small in-test stand-in whose ``store_raw`` can be
told to fail for chosen content, so the "chunks did not all store" condition is produced by the
real pipeline's own failure accounting rather than hand-built.

Flag used: ``connectors_enabled``.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration


class _FakeNotion:
    """Minimal Notion-shaped API. ``pages`` is mutable between syncs; ``search_cursors`` records
    the ``start_cursor`` of every listing request (None for a first-page request)."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.search_cursors: list[str | None] = []
        source = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802 - http.server API
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                start = req.get("start_cursor")
                source.search_cursors.append(start)
                size = int(req.get("page_size", 100))
                offset = int(start) if start else 0
                chunk = source.pages[offset : offset + size]
                more = offset + size < len(source.pages)
                self._send(
                    {
                        "results": [
                            {"id": p["id"], "url": f"https://example.invalid/{p['id']}", "last_edited_time": p["rev"]}
                            for p in chunk
                        ],
                        "has_more": more,
                        "next_cursor": str(offset + size) if more else None,
                    }
                )

            def do_GET(self):  # noqa: N802 - http.server API
                page_id = self.path.split("/blocks/")[1].split("/children")[0]
                text = next(p["text"] for p in source.pages if p["id"] == page_id)
                self._send(
                    {
                        "results": [
                            {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": text}]}}
                        ],
                        "has_more": False,
                    }
                )

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class _StoreStandIn:
    """Memory-service stand-in for the real ``ingest_document``: records stored chunk text and
    raises for any chunk containing a marker in ``fail_for``."""

    def __init__(self):
        self.stored: list[str] = []
        self.fail_for: set[str] = set()

    def store_raw(self, content, **kwargs):
        if any(marker in content for marker in self.fail_for):
            raise RuntimeError("simulated store outage")
        self.stored.append(content)
        return [SimpleNamespace(id=f"mem-{len(self.stored)}", visibility=None)]

    def extract_facts_only(self, *args, **kwargs):
        return []


def _run_syncs(stores, unique_ns, flags, monkeypatch, fake, store, steps, *, page_size=None):
    """Register a Notion connector against ``fake`` and run ``steps`` syncs through a real arq
    worker. Each step is a callable run BEFORE its sync (to mutate the source or the store).
    Returns ([result per sync], [stored vault record after each sync])."""
    import redis.asyncio as aioredis
    from arq.worker import Worker
    from cryptography.fernet import Fernet

    import worker
    from config import parse_redis_settings
    from connectors.vault import ConnectorVault
    from task_manager import TaskManager

    queue = f"nsit-connector-sync-fc-{unique_ns.user_id[-10:]}"
    flags(connectors_enabled=True, redis_url=stores.redis_url, ingest_queue_name=queue)
    connector_id = f"nsit-conn-fc-{unique_ns.user_id[-10:]}"
    monkeypatch.setattr("connectors.adapters.notion._API_BASE", fake.url)

    async def no_graph_jobs(*args, **kwargs):
        return None

    monkeypatch.setattr(worker, "_enqueue_graph_jobs", no_graph_jobs)

    async def scenario():
        redis_store = aioredis.Redis.from_url(stores.redis_url, decode_responses=True)
        vault = ConnectorVault(redis_store, Fernet(Fernet.generate_key()))
        tm = TaskManager()
        await tm.connect()
        redis_settings = parse_redis_settings()
        job_ids: list[str] = []
        results, records = [], []
        try:
            config = {"page_size": page_size} if page_size else {}
            await vault.put(
                {
                    "connector_id": connector_id,
                    "connector_type": "notion",
                    "owner_user_id": unique_ns.user_id,
                    "credentials": {"token": "nsit-not-a-secret"},
                    "config": config,
                }
            )
            for before in steps:
                before()
                record = await vault.get(connector_id)
                task_id = await tm.enqueue_connector_sync(
                    connector_id, unique_ns.user_id, generation=record["generation"]
                )
                job_ids.append(task_id)
                w = Worker(
                    functions=[worker.process_connector_sync],
                    queue_name=queue,
                    redis_settings=redis_settings,
                    ctx={"vault": vault, "service": store},
                    burst=True,
                    poll_delay=0.1,
                    handle_signals=False,
                )
                try:
                    await w.main()
                finally:
                    await w.close()
                status = await tm.get_status(task_id)
                assert status["status"] == "completed", f"sync did not complete: {status}"
                results.append(status["result"])
                records.append(await vault.get(connector_id))
            return results, records
        finally:
            keys = [f"ns:connector:{connector_id}"] + [f"arq:result:{j}" for j in set(job_ids)]
            await redis_store.delete(*keys)
            await redis_store.aclose()
            await tm.pool.aclose()

    try:
        return asyncio.run(scenario())
    finally:
        fake.close()


def test_a_resource_whose_chunks_failed_to_store_is_not_marked_synced_and_is_retried(
    flags, stores, unique_ns, monkeypatch
):
    pages = [
        {"id": "page-ok", "rev": "r1", "text": "GOODTEXT the first page stores cleanly."},
        {"id": "page-bad", "rev": "r1", "text": "BADTEXT the second page fails to store."},
    ]
    fake = _FakeNotion(pages)
    store = _StoreStandIn()
    store.fail_for = {"BADTEXT"}

    def heal_store():
        store.fail_for = set()

    results, records = _run_syncs(
        stores, unique_ns, flags, monkeypatch, fake, store, [lambda: None, heal_store, lambda: None]
    )
    first, second, third = results

    # Run 1: one resource stored, one did not; the run says so and records only the stored one.
    assert (first["synced"], first["skipped"], first["failed"]) == (1, 0, 1), first
    assert records[0]["last_revision_by_id"] == {"page-ok": "r1"}, records[0]["last_revision_by_id"]
    assert any("GOODTEXT" in c for c in store.stored[:1]), store.stored

    # Run 2 (store healed): the failed resource is still eligible and now stores; the stored one is skipped.
    assert (second["synced"], second["skipped"], second["failed"]) == (1, 1, 0), second
    assert records[1]["last_revision_by_id"] == {"page-ok": "r1", "page-bad": "r1"}, records[1]["last_revision_by_id"]
    assert sum("BADTEXT" in c for c in store.stored) == 1, store.stored

    # Run 3: everything is recorded, so nothing is ingested again.
    assert (third["synced"], third["skipped"], third["failed"]) == (0, 2, 0), third


def test_a_page_based_connector_returns_to_the_first_page_after_the_end_of_the_listing(
    flags, stores, unique_ns, monkeypatch
):
    pages = [{"id": f"page-{n}", "rev": "r1", "text": f"NOTE{n} body"} for n in (1, 2, 3)]
    fake = _FakeNotion(pages)
    store = _StoreStandIn()

    def add_page_at_the_front():
        fake.pages.insert(0, {"id": "page-new", "rev": "r1", "text": "NOTENEW body"})

    results, records = _run_syncs(
        stores, unique_ns, flags, monkeypatch, fake, store, [lambda: None, lambda: None, add_page_at_the_front],
        page_size=2,
    )
    first, second, third = results

    # Sync 1 reads page 1 (two items) and stores a cursor for page 2.
    assert (first["synced"], first["skipped"], first["failed"]) == (2, 0, 0), first
    assert records[0]["cursor"] == "2", records[0]["cursor"]
    # Sync 2 reads the last page; the end of the listing resets the stored cursor.
    assert (second["synced"], second["skipped"], second["failed"]) == (1, 0, 0), second
    assert records[1]["cursor"] is None, f"cursor not reset at the end of the listing: {records[1]['cursor']!r}"
    # Sync 3 starts from the first page again and sees the item added at the front.
    assert fake.search_cursors == [None, "2", None], fake.search_cursors
    assert (third["synced"], third["skipped"], third["failed"]) == (1, 1, 0), third
    assert any("NOTENEW" in c for c in store.stored), store.stored
    assert records[2]["last_revision_by_id"].keys() == {"page-1", "page-2", "page-3", "page-new"}
