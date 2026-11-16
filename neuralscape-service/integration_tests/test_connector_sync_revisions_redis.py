"""Flag-on proof: a second connector sync over unchanged items ingests nothing.

Real Redis carries everything the feature touches: the encrypted connector vault,
the ARQ queue, the job-result cache and the task status. A real ``arq`` worker runs
the service's own ``process_connector_sync`` job, the service's own ``TaskManager``
enqueues each sync and reports its result, and the connector is the service's
generic REST adapter pointed at a local HTTP source whose items carry a ``rev``.

Only the document ingest itself is replaced by a recorder (the real one needs the
LLM and the vector store); the assertions are about the revision bookkeeping and
about the second sync really running instead of being answered with the first
run's cached result.

Flag used: ``connectors_enabled``.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytestmark = pytest.mark.integration


class _Source:
    """A tiny JSON API: ``GET /items`` -> ``{"items": [...]}``, mutable between syncs."""

    def __init__(self):
        self.items = [{"id": f"item-{n}", "rev": "1", "body": f"# Note {n}\n\nbody {n}"} for n in (1, 2)]
        source = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server API
                body = json.dumps({"items": source.items}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/items"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_second_sync_skips_unchanged_items_and_reingests_only_the_changed_one(flags, stores, unique_ns, monkeypatch):
    import redis.asyncio as aioredis
    from arq.worker import Worker
    from cryptography.fernet import Fernet

    import worker
    from config import parse_redis_settings
    from connectors.vault import ConnectorVault
    from task_manager import TaskManager

    queue = f"nsit-connector-sync-{unique_ns.user_id[-10:]}"
    flags(connectors_enabled=True, redis_url=stores.redis_url, ingest_queue_name=queue)
    connector_id = f"nsit-conn-{unique_ns.user_id[-10:]}"
    ingested: list[str] = []

    def record_ingest(service, doc):
        ingested.append(doc.source["external_id"])
        return {"graph_jobs": []}

    async def no_graph_jobs(*args, **kwargs):
        return None

    monkeypatch.setattr("ingest.pipeline.ingest_document", record_ingest)
    monkeypatch.setattr(worker, "_enqueue_graph_jobs", no_graph_jobs)

    source = _Source()
    # The sink is an in-test loopback http server: allow exactly its host:port, as the
    # destination policy requires for any internal or plain-http connector target.
    flags(connector_allowed_hosts=source.url.split("/")[2])

    async def scenario():
        store = aioredis.Redis.from_url(stores.redis_url, decode_responses=True)
        vault = ConnectorVault(store, Fernet(Fernet.generate_key()))
        tm = TaskManager()
        await tm.connect()
        redis_settings = parse_redis_settings()
        job_ids: list[str] = []

        async def run_sync() -> dict:
            record = await vault.get(connector_id)
            task_id = await tm.enqueue_connector_sync(
                connector_id, unique_ns.user_id, generation=record["generation"]
            )
            job_ids.append(task_id)
            w = Worker(
                functions=[worker.process_connector_sync],
                queue_name=queue,
                redis_settings=redis_settings,
                ctx={"vault": vault, "service": object()},
                burst=True,
                poll_delay=0.1,
                handle_signals=False,
            )
            try:
                await w.main()
            finally:
                await w.close()
            status = await tm.get_status(task_id)
            assert status["status"] == "completed", status
            return status["result"]

        try:
            await vault.put(
                {
                    "connector_id": connector_id,
                    "connector_type": "generic_rest",
                    "owner_user_id": unique_ns.user_id,
                    "credentials": {"bearer": "nsit-not-a-secret"},
                    "config": {
                        "list_url": source.url,
                        "items_path": "items",
                        "id_field": "id",
                        "revision_field": "rev",
                        "content_field": "body",
                    },
                }
            )
            first = await run_sync()
            after_first = list(ingested)
            second = await run_sync()
            after_second = list(ingested)
            source.items[1]["rev"] = "2"
            third = await run_sync()
            stored = await vault.get(connector_id)
            return first, after_first, second, after_second, third, stored, job_ids
        finally:
            keys = [f"ns:connector:{connector_id}"] + [f"arq:result:{j}" for j in job_ids]
            await store.delete(*keys)
            await store.aclose()
            await tm.pool.aclose()

    try:
        first, after_first, second, after_second, third, stored, job_ids = asyncio.run(scenario())
    finally:
        source.close()

    assert (first["synced"], first["skipped"]) == (2, 0), first
    assert sorted(after_first) == ["item-1", "item-2"]
    # The same job id serves every sync of this connector instance; each one must really run.
    assert len(set(job_ids)) == 1
    assert (second["synced"], second["skipped"]) == (0, 2), second
    assert after_second == after_first, "unchanged items were ingested again"
    assert (third["synced"], third["skipped"]) == (1, 1), third
    assert ingested[len(after_second):] == ["item-2"]
    assert stored["last_revision_by_id"] == {"item-1": "1", "item-2": "2"}
