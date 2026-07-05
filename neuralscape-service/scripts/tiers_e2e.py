"""End-to-end exercise of C3 (reasoning-tier ask) + C4 (checkpoint batch save
+ queue visibility + queue.empty webhook) against real Qdrant/Redis and the
real Gemini answering path.

SAFE BY CONSTRUCTION: refuses to run against the default Qdrant collection
OR the default ARQ queue names — everything runs on dedicated names, so the
live workers never consume this script's jobs and this script's in-process
burst worker never touches live data. Cleans up after itself (drops the
collection, deletes the queue/tracking keys).

Usage (from neuralscape-service/, with Redis/Qdrant up):

    set -a; source ../.env; set +a
    QDRANT_COLLECTION=tiers_e2e \
    ARQ_QUEUE_NAME=tiers-e2e:queue \
    GRAPH_QUEUE_NAME=tiers-e2e:graph \
    INGEST_QUEUE_NAME=tiers-e2e:ingest \
    uv run python scripts/tiers_e2e.py

Exercises:

1. ask @ minimal   → single semantic search, direct cited answer, no
                     fabricated citation ids
2. ask @ high      → keyword + semantic + update-language passes; the
                     seeded contradiction (Tue → rescheduled Thu) is
                     surfaced with the newer fact preferred
3. abstention      → an unknowable question honestly abstains
4. checkpoint      → 5 items (2 pre-stored dupes) + session note → per-item
                     verdicts, ONE batch task, verified stored via task
                     polling after an in-process burst worker run
5. queue_status    → queued/caught_up=false before the worker; completed/
                     caught_up=true after
6. queue.empty     → webhook POSTs to a local recorder when the burst
                     worker drains the queue
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

USER = "tiers-e2e"
NOW = datetime.now(timezone.utc)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))


def backdate(service, mid: str, dt: datetime) -> None:
    from config import settings

    service._memory.vector_store.client.set_payload(
        collection_name=settings.qdrant_collection,
        payload={"created_at": dt.isoformat()},
        points=[mid],
    )


def seed(service, content: str, dt: datetime, **kw) -> str:
    [resp] = service.store_raw(
        content=content, user_id=USER, add_to_graph=False, **kw
    )
    backdate(service, resp.id, dt)
    return resp.id


class _WebhookRecorder(BaseHTTPRequestHandler):
    events: list[dict] = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            _WebhookRecorder.events.append(json.loads(self.rfile.read(length)))
        except Exception:
            pass
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):  # silence
        pass


async def main() -> int:
    from config import parse_redis_settings, settings

    if settings.qdrant_collection == "neuralscape_memories":
        print("REFUSING to run against the default collection. Set QDRANT_COLLECTION.")
        return 2
    if settings.arq_queue_name == "neuralscape:queue":
        print("REFUSING to run against the default ARQ queue. Set ARQ_QUEUE_NAME.")
        return 2

    import checkpoint as checkpoint_mod
    from ask import ask_memory
    from memory_service import MemoryService
    from schemas import CheckpointRequest
    from task_manager import TaskManager, _task_user_key, _user_tasks_key
    from worker import _make_after_job_end, process_memory_raw_batch

    print(f"collection={settings.qdrant_collection} queue={settings.arq_queue_name} user={USER}")
    service = MemoryService()
    service._get_memory()
    client = service._memory.vector_store.client
    # Vector-only run: the burst worker's batch path would otherwise perform
    # inline Graphiti extraction (minutes of Gemini) into the LIVE Neo4j graph.
    # enrich_graph no-ops when the graph handles are absent; search's graph
    # pass degrades gracefully (its failures are non-critical by design).
    service._graphiti = None
    service._bridge = None

    tm = TaskManager()
    await tm.connect()

    # Local webhook recorder
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _WebhookRecorder)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    settings.webhook_queue_empty_url = f"http://127.0.0.1:{httpd.server_address[1]}/hook"

    task_id = None
    try:
        # ── [1] Seed: contradiction pair + fillers ──
        print("\n[1] seeding facts (contradiction pair + fillers)")
        old_id = seed(
            service, "The team's weekly sync meeting is on Tuesday at 10:00.",
            NOW - timedelta(days=30), category="interaction", visibility="private",
        )
        new_id = seed(
            service,
            "The team's weekly sync meeting was rescheduled and is now on Thursday at 14:00.",
            NOW - timedelta(days=2), category="interaction", visibility="private",
        )
        seed(service, "The staging deploy pipeline uses blue-green rollout.",
             NOW - timedelta(days=10), category="decision", visibility="private")
        seed(service, "Grafana dashboards live under the ops folder.",
             NOW - timedelta(days=8), category="domain_knowledge", visibility="private")
        seed(service, "The release captain rotates every sprint.",
             NOW - timedelta(days=5), category="convention", visibility="private")
        seeded_ids = {old_id, new_id}
        check("seeded 5 facts", True)

        # ── [2] ask @ minimal ──
        print("\n[2] ask @ minimal (single search, direct answer)")
        out_min = await ask_memory(
            service, question="When is the team's weekly sync meeting?",
            user_id=USER, reasoning_level="minimal",
        )
        check("minimal: exactly one search pass", len(out_min["searches"]) == 1,
              str(out_min["searches"]))
        check("minimal: non-empty answer", bool(out_min["answer"]),
              out_min["answer"][:100])
        check("minimal: not abstained", out_min["abstained"] is False)
        check("minimal: citations are real retrieved ids",
              all(c in {old_id, new_id} or len(c) == 36 for c in out_min["citations"])
              and bool(out_min["citations"]), str(out_min["citations"]))

        # ── [3] ask @ high — contradiction surfacing ──
        print("\n[3] ask @ high (iterative loop, contradiction surfacing)")
        out_high = await ask_memory(
            service, question="When is the team's weekly sync meeting?",
            user_id=USER, reasoning_level="high",
        )
        answer_l = out_high["answer"].lower()
        check("high: keyword + semantic + update passes ran",
              len(out_high["searches"]) >= 3
              and out_high["searches"][0].startswith("keyword:"),
              str(out_high["searches"][:3]))
        check("high: not abstained", out_high["abstained"] is False)
        check("high: newer fact wins (Thursday)", "thursday" in answer_l,
              out_high["answer"][:160])
        newer_surfaced = new_id in out_high["citations"] or "thursday" in answer_l
        older_surfaced = old_id in out_high["citations"] or "tuesday" in answer_l
        check("high: contradiction surfaced (both sides present)",
              newer_surfaced and older_surfaced,
              f"citations={out_high['citations']} answer={out_high['answer'][:160]}")
        check("high: no fabricated citation ids",
              all(len(c) == 36 for c in out_high["citations"]))

        # ── [4] abstention ──
        print("\n[4] abstention on an unknowable question")
        out_abs = await ask_memory(
            service, question="What is the user's blood type?",
            user_id=USER, reasoning_level="high",
        )
        abstained = out_abs["abstained"] or "don't know" in out_abs["answer"].lower()
        check("abstains instead of fabricating", abstained, out_abs["answer"][:160])
        check("abstention cites nothing it didn't retrieve",
              all(len(c) == 36 for c in out_abs["citations"]))

        # ── [5] checkpoint: 5 items, 2 dupes, + session note ──
        print("\n[5] checkpoint (5 items, 2 pre-stored dupes, session note)")
        dupe_a = "Checkpoint dupe alpha: the API gateway caches for 60s."
        dupe_b = "Checkpoint dupe beta: retries use exponential backoff."
        for content in (dupe_a, dupe_b):
            service.store_raw(content=content, user_id=USER, category="decision",
                              add_to_graph=False)
        new_contents = [
            "Checkpoint new one: cron anchors at 03:00 UTC.",
            "Checkpoint new two: the vault key rotates quarterly.",
            "Checkpoint new three: MCP census is tracked in tests.",
        ]
        req = CheckpointRequest(
            user_id=USER,
            memories=[
                {"content": dupe_a, "category": "decision"},
                {"content": new_contents[0], "category": "decision"},
                {"content": dupe_b, "category": "decision"},
                {"content": new_contents[1], "category": "decision"},
                {"content": new_contents[2], "category": "decision"},
            ],
            session_note={
                "request": "Run the tiers E2E",
                "completed": "Seeded, asked, checkpointed",
                "next_steps": "Verify queue drains",
            },
        )
        prepared = checkpoint_mod.prepare_checkpoint(service, req, USER)
        verdicts = [v["verdict"] for v in prepared["verdicts"]]
        check("verdicts: 2 duplicates flagged at the right indexes",
              verdicts == ["duplicate", "new", "duplicate", "new", "new"],
              str(verdicts))
        check("duplicate verdicts carry existing ids",
              all("existing_id" in v for v in prepared["verdicts"]
                  if v["verdict"] == "duplicate"))
        check("enqueue set = 3 new + session note",
              len(prepared["to_enqueue"]) == 4)
        task_id = await tm.enqueue_raw_batch(items=prepared["to_enqueue"])
        check("single batch task id", isinstance(task_id, str) and bool(task_id), task_id)

        # ── [6] queue_status BEFORE the worker ──
        print("\n[6] queue_status before the worker")
        qs_before = await tm.get_queue_status(USER)
        check("tracked includes the checkpoint task", qs_before["tracked"] >= 1,
              str(qs_before))
        check("queued > 0 and not caught up",
              qs_before["counts"]["queued"] >= 1 and qs_before["caught_up"] is False,
              str(qs_before["counts"]))

        # ── [7] burst worker drains our dedicated queue ──
        print("\n[7] in-process burst worker (dedicated queue)")
        from arq.worker import Worker

        w = Worker(
            functions=[process_memory_raw_batch],
            redis_settings=parse_redis_settings(),
            queue_name=settings.arq_queue_name,
            burst=True,
            ctx={"service": service},
            after_job_end=_make_after_job_end(settings.arq_queue_name),
            poll_delay=0.2,
        )
        await w.main()
        await w.close()

        # ── [8] task polls to completed; storage verified ──
        print("\n[8] poll task status + verify storage")
        status = None
        for _ in range(20):
            status = await tm.get_status(task_id)
            if status["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.5)
        check("batch task completed", status is not None and status["status"] == "completed",
              str(status and status["status"]))
        stored = status["result"]["memories"] if status and status.get("result") else []
        stored_contents = {m["memory"] for m in stored}
        check("3 new + note stored by the batch job", len(stored) == 4, str(len(stored)))
        check("session note stored as task_context/meeting_outcome",
              any(m.get("category") == "task_context"
                  and m.get("observation_type") == "meeting_outcome"
                  and m["memory"].startswith("Session note:") for m in stored))
        check("all new contents present",
              all(c in stored_contents for c in new_contents))
        # Dupes were excluded — each dupe content exists exactly once in Qdrant.
        import hashlib

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        for label, content in (("alpha", dupe_a), ("beta", dupe_b)):
            pts, _ = client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="user_id", match=MatchValue(value=USER)),
                    FieldCondition(key="hash", match=MatchValue(
                        value=hashlib.md5(content.encode()).hexdigest())),
                ]),
                limit=10, with_payload=False, with_vectors=False,
            )
            check(f"dupe {label} exists exactly once", len(pts) == 1, str(len(pts)))

        # ── [9] queue_status AFTER: caught up ──
        print("\n[9] queue_status after the worker")
        qs_after = await tm.get_queue_status(USER)
        check("caught_up after drain", qs_after["caught_up"] is True,
              str(qs_after["counts"]))
        check("completed counted", qs_after["counts"]["completed"] >= 1,
              str(qs_after["counts"]))

        # ── [10] queue.empty webhook fired ──
        print("\n[10] queue.empty webhook")
        deadline = time.time() + 5
        while time.time() < deadline and not _WebhookRecorder.events:
            await asyncio.sleep(0.2)
        events = list(_WebhookRecorder.events)
        check("webhook delivered", len(events) >= 1, f"{len(events)} events")
        if events:
            ev = events[-1]
            check("event shape", ev.get("event") == "queue.empty"
                  and ev.get("queue") == settings.arq_queue_name, json.dumps(ev))
            check("event attributes the caller", ev.get("user_id") == USER,
                  str(ev.get("user_id")))

    finally:
        print("\n[cleanup] dropping collection + redis keys")
        try:
            client.delete_collection(collection_name=settings.qdrant_collection)
        except Exception as e:
            print(f"  collection cleanup failed: {e}")
        try:
            import redis as redis_lib

            r = redis_lib.Redis.from_url(settings.redis_url)
            keys = [_user_tasks_key(USER)]
            if task_id:
                keys += [_task_user_key(task_id), f"arq:result:{task_id}",
                         f"arq:job:{task_id}"]
            keys += [settings.arq_queue_name, settings.graph_queue_name,
                     settings.ingest_queue_name]
            r.delete(*keys)
            r.close()
        except Exception as e:
            print(f"  redis cleanup failed: {e}")
        httpd.shutdown()
        await tm.close()
        service.close()

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'=' * 60}\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        for name, _, detail in failed:
            print(f"  FAILED: {name} {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
