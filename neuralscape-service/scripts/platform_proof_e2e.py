"""End-to-end exercise of E3 (session summarizer slots + context assembler)
and E4 (custom extraction instructions) against real Qdrant/Redis + Gemini.

SAFE BY CONSTRUCTION: runs against a dedicated Qdrant collection
(``QDRANT_COLLECTION`` must be overridden — the script refuses the default
collection), unique per-run user ids, and its own Redis session/settings
keys. Cleans up after itself (drops the collection, deletes its keys).

Usage (from neuralscape-service/, with Qdrant/Redis up):

    set -a; source ../.env; set +a
    QDRANT_COLLECTION=platform_e2e \
    LLM_GATEWAY_ENABLED=false LLM_GATEWAY_GRAPHITI_ENABLED=false \
    uv run python scripts/platform_proof_e2e.py

(The gateway is disabled for the run when the deployment's gateway fronts
google-vertex embeddings, which reject mem0's batched embed calls — the
E2E goes straight to AI Studio via GOOGLE_API_KEY.)

Exercises:

1. summarizer  → a simulated 25-message session crosses the short-slot
                 threshold via the worker trigger; the enqueued refresh
                 runs with real Gemini; the stored slot is ≤ its token
                 budget and reflects the conversation
2. assembler   → POST /v1/context/assemble at 2k and 8k budgets with
                 format=anthropic: shape ({system, messages}), budget
                 respected (used_tokens ≤ budget), savings line present
                 and ledgered
3. formatters  → openai + plain shapes at 2k
4. instructions→ PUT project-wide extraction instructions (dictator gate:
                 403 for non-dictator, 200 for dictator), then a real
                 extract_and_store conversation — the extracted decision
                 memory reflects the guidance (carries the ADR number)
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hermetic regardless of the sourced .env: no MCP HTTP app at import time,
# never the operator's real vault, auth bypassed (legacy body-user_id mode).
os.environ["MCP_TRANSPORT"] = "stdio"
os.environ["OBSIDIAN_VAULT_PATH"] = "/tmp/platform-e2e-vault"
os.environ["DREAMING_OBSIDIAN_VAULT_PATH"] = "/tmp/platform-e2e-vault"
os.environ["AUTH_PROVIDER"] = "token"
os.environ["NEURALSCAPE_API_KEY"] = ""
os.environ["NEURALSCAPE_USER_TOKEN_SECRET"] = ""

RUN = uuid.uuid4().hex[:8]
USER = f"platform-e2e-user-{RUN}"
DICTATOR = f"platform-e2e-dictator-{RUN}"
SESSION = f"platform-e2e-sess-{RUN}"
PROJECT = f"platform-e2e-proj-{RUN}"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))


def main() -> int:  # noqa: PLR0915 — linear E2E narrative
    from config import settings

    if settings.qdrant_collection == "neuralscape_memories":
        print("REFUSING to run against the default collection. Set QDRANT_COLLECTION.")
        return 2

    settings.neuralscape_api_key = ""
    settings.neuralscape_user_token_secret = ""
    settings.savings_meter_enabled = True
    settings.session_summary_enabled = True
    settings.extraction_instructions_enabled = True
    settings.dictator_user_ids = DICTATOR

    import redis as redis_lib
    from fastapi.testclient import TestClient

    import extraction_settings as es
    import main as main_mod
    import session_summarizer as ss
    import worker as worker_mod
    from memory_service import MemoryService

    print(f"collection={settings.qdrant_collection} run={RUN}")
    service = MemoryService()
    service._get_memory()
    main_mod._service = service
    client = TestClient(main_mod.app, raise_server_exceptions=False)
    r = redis_lib.Redis.from_url(settings.redis_url)

    try:
        # ═══ 1. Summarizer: 25-message session crosses the short threshold ═══
        print("\n[1] session summarizer (25-message session)")
        topics = [
            ("user", "Let's design the billing service. I want idempotent webhooks."),
            ("assistant", "Agreed — dedupe on the Stripe event id with a processed-events table."),
            ("user", "We also decided to keep invoices in Postgres, not Mongo."),
            ("assistant", "Postgres it is; JSONB for the line items."),
            ("user", "Remind me to add a retry backoff to the webhook consumer."),
        ]
        messages = [
            {"role": role, "content": f"({i}) {text}"}
            for i in range(5)
            for role, text in topics
        ]  # 25 messages
        assert len(messages) == 25

        enqueued: list[tuple] = []

        class _CaptureArq:
            async def enqueue_job(self, fn, *args, **kwargs):
                enqueued.append((fn, args, kwargs))

        ctx = {"redis": _CaptureArq()}
        # Simulate three conversation writes for the same session (the fast
        # worker's trigger path, verbatim).
        for batch in (messages[:10], messages[10:20], messages[20:]):
            asyncio.run(worker_mod._note_session_messages(ctx, USER, SESSION, batch))

        due = [(a[2], k.get("_job_id")) for f, a, k in enqueued if f == "process_session_summary"]
        check("threshold trigger enqueued a short refresh", ("short", f"sess-{USER}-{SESSION}-short-1") in due, str(due))

        # Run the enqueued refresh job(s) for real (Gemini). Dedupe by job id
        # first — ARQ coalesces identical _job_ids (that's exactly why the
        # trigger derives them from the (session, slot, threshold bucket)).
        seen_jobs: set[str] = set()
        for fn, args, kwargs in enqueued:
            if fn != "process_session_summary" or kwargs.get("_job_id") in seen_jobs:
                continue
            seen_jobs.add(kwargs.get("_job_id"))
            out = asyncio.run(worker_mod.process_session_summary({}, *args))
            check(f"refresh job ran ({args[2]})", out.get("status") == "refreshed", str(out))

        slot = ss.load_slot(USER, SESSION, "short")
        check("short slot exists", slot is not None)
        if slot:
            budget = settings.session_summary_short_max_tokens
            real = ss.text_tokens(slot["text"])
            check("short slot within budget", real <= budget, f"{real} ≤ {budget}")
            check("slot reflects the session",
                  any(w in slot["text"].lower() for w in ("billing", "webhook", "postgres", "invoice")),
                  slot["text"][:120].replace("\n", " "))
            check("through_count advanced", slot.get("through_count") == 25, str(slot.get("through_count")))

        # ═══ 2. Assembler at 2k / 8k, format=anthropic ═══
        print("\n[2] context assembler (2k / 8k, anthropic)")
        for budget in (2000, 8000):
            resp = client.post("/v1/context/assemble", json={
                "budget_tokens": budget, "user_id": USER, "session_id": SESSION,
                "format": "anthropic", "query": "webhook idempotency decisions",
            })
            ok = resp.status_code == 200
            check(f"assemble {budget} returns 200", ok, str(resp.status_code))
            if not ok:
                continue
            body = resp.json()
            check(f"assemble {budget} shape", set(body["bundle"]) == {"system", "messages"}
                  and all(m["role"] in ("user", "assistant") for m in body["bundle"]["messages"]))
            check(f"assemble {budget} within budget",
                  body["used_tokens"] <= budget, f"{body['used_tokens']} ≤ {budget}")
            check(f"assemble {budget} has summary + messages",
                  body["sections"]["summary_tokens"] > 0 and body["sections"]["messages_included"] > 0,
                  str(body["sections"]))
            check(f"assemble {budget} savings line present",
                  isinstance(body.get("savings"), str) and "net of overhead" in body["savings"],
                  str(body.get("savings")))
        # Ledger actually recorded context_assemble events for this user.
        entries = r.xrange(f"ns:savings:{USER}")
        ops = [fields.get(b"op", b"").decode() for _, fields in entries]
        check("savings ledger has context_assemble entries", ops.count("context_assemble") >= 2, str(ops))

        # ═══ 3. Other formatter shapes ═══
        print("\n[3] openai + plain shapes")
        body = client.post("/v1/context/assemble", json={
            "budget_tokens": 2000, "user_id": USER, "session_id": SESSION, "format": "openai",
        }).json()
        msgs = body["bundle"].get("messages", [])
        check("openai shape", bool(msgs) and msgs[0]["role"] == "system", str(msgs[:1]))
        body = client.post("/v1/context/assemble", json={
            "budget_tokens": 2000, "user_id": USER, "session_id": SESSION, "format": "plain",
        }).json()
        check("plain shape", isinstance(body["bundle"].get("text"), str)
              and "## Recent messages" in body["bundle"]["text"])

        # ═══ 4. Custom extraction instructions (E4) ═══
        print("\n[4] extraction instructions (dictator gate + guided extraction)")
        guidance = ("Always tag decisions with the ADR number when one is present, "
                    "phrasing the fact as 'ADR-<number>: <decision> because <why>'.")
        resp = client.put("/v1/settings/extraction-instructions", json={
            "instructions": guidance, "user_id": USER, "project_id": PROJECT,
        })
        check("non-dictator project PUT rejected", resp.status_code == 403, str(resp.status_code))
        resp = client.put("/v1/settings/extraction-instructions", json={
            "instructions": guidance, "user_id": DICTATOR, "project_id": PROJECT,
        })
        check("dictator project PUT accepted", resp.status_code == 200, str(resp.status_code))
        got = client.get("/v1/settings/extraction-instructions",
                         params={"user_id": USER, "project_id": PROJECT}).json()
        check("member reads project instructions", got.get("instructions") == guidance)

        conversation = [
            {"role": "user", "content":
                "For the payments revamp we compared Kafka and Redis Streams for event "
                "transport. We're going with Redis Streams because we already operate Redis "
                "and the volume is modest. I logged this in the decision record ADR-077."},
            {"role": "assistant", "content":
                "Makes sense — Redis Streams keeps the ops surface small. I'll note ADR-077."},
        ]
        stored = service.extract_and_store(
            messages=conversation, user_id=USER, project_id=PROJECT, run_id=SESSION
        )
        check("extraction stored memories", len(stored) > 0, f"{len(stored)} memories")
        decision_texts = [m.memory for m in stored]
        check("extracted memory reflects the guidance (ADR number attached)",
              any("ADR-077" in t for t in decision_texts),
              " | ".join(t[:90] for t in decision_texts))
    finally:
        # ── Cleanup: collection + all per-run Redis keys ──
        print("\n[cleanup]")
        try:
            service._memory.vector_store.client.delete_collection(settings.qdrant_collection)
            print("  dropped collection")
        except Exception as e:  # noqa: BLE001
            print(f"  collection drop failed: {e}")
        try:
            patterns = [
                f"ns:session:{USER}:*",
                f"ns:extraction-instructions:project:{PROJECT}",
                f"ns:extraction-instructions:user:{USER}",
                f"ns:savings:{USER}", f"ns:savings:totals:{USER}",
                f"ns:savings:{DICTATOR}", f"ns:savings:totals:{DICTATOR}",
                f"ns:savings:schema-charged:{USER}:*",
                f"ns:savings:schema-charged:{DICTATOR}:*",
                f"ns:user-tasks:{USER}", f"ns:user-tasks:{DICTATOR}",
            ]
            deleted = 0
            for pat in patterns:
                if "*" in pat:
                    for key in r.scan_iter(match=pat):
                        deleted += r.delete(key)
                else:
                    deleted += r.delete(pat)
            print(f"  deleted {deleted} redis keys")
        except Exception as e:  # noqa: BLE001
            print(f"  redis cleanup failed: {e}")

        failed = [c for c in CHECKS if not c[1]]
        print(f"\n{'='*60}\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
        for name, _, detail in failed:
            print(f"  FAILED: {name} — {detail}")

    return 1 if any(not c[1] for c in CHECKS) else 0


if __name__ == "__main__":
    sys.exit(main())
