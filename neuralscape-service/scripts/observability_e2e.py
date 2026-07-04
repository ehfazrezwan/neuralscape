"""End-to-end exercise of A5 (surprisal-targeted REM), E1 (SSE live stream)
and E2 (token-economics telemetry) against real Qdrant/Redis (+ Gemini for
the dream sweep's reflection pass).

SAFE BY CONSTRUCTION: runs against a dedicated Qdrant collection
(``QDRANT_COLLECTION`` must be overridden — the script refuses the default
collection), unique per-run user ids, and its own dream pool. Cleans up
after itself (drops the collection, deletes its Redis ledger/gate keys).

Usage (from neuralscape-service/, with Qdrant/Redis up):

    set -a; source ../.env; set +a
    QDRANT_COLLECTION=observability_e2e \
    uv run python scripts/observability_e2e.py

Exercises:

1. write path   → token_estimate is a REAL tiktoken count at write time
2. index recall → POST /v1/search index_only carries a positive measured
                  savings line with plausible numbers (baseline == sum of
                  stored counts; net = baseline − rows − line overhead)
3. metrics      → GET /v1/metrics totals equal the raw ledger-stream sum
                  (including the once-per-day tool_schema charge)
4. SSE          → a live /v1/stream client receives a memory_stored event
                  for its own write (emitted through the real extension-
                  registry hook) and NEVER one for another user's private
                  write; shared events arrive
5. surprisal    → a planted-anomaly memory ranks top-of-substrate in the
                  REM reflection pass of a real dream sweep (force+dry_run)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Keep the run hermetic regardless of the sourced .env: no MCP HTTP app at
# import time, and NEVER the operator's real vault (the sweep below is
# dry-run with all vault passes disabled anyway — this is belt and braces).
os.environ["MCP_TRANSPORT"] = "stdio"
os.environ["OBSIDIAN_VAULT_PATH"] = "/tmp/obs-e2e-vault"
os.environ["DREAMING_OBSIDIAN_VAULT_PATH"] = "/tmp/obs-e2e-vault"
# Auth is exercised elsewhere; this run bypasses it (legacy body-user_id mode).
os.environ["AUTH_PROVIDER"] = "token"
os.environ["NEURALSCAPE_API_KEY"] = ""
os.environ["NEURALSCAPE_USER_TOKEN_SECRET"] = ""

RUN = uuid.uuid4().hex[:8]
USER_A = f"obs-e2e-alice-{RUN}"
USER_B = f"obs-e2e-bob-{RUN}"
USER_S = f"obs-e2e-dreamer-{RUN}"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))


def seed(service, content: str, user_id: str, **kw) -> str:
    [resp] = service.store_raw(
        content=content, user_id=user_id, add_to_graph=False, **kw
    )
    return resp.id


def main() -> int:  # noqa: PLR0915 — linear E2E narrative
    from config import settings

    if settings.qdrant_collection == "neuralscape_memories":
        print("REFUSING to run against the default collection. Set QDRANT_COLLECTION.")
        return 2

    # Local-run determinism: bypass auth middleware, force both features on.
    settings.neuralscape_api_key = ""
    settings.neuralscape_user_token_secret = ""
    settings.savings_meter_enabled = True
    settings.event_stream_enabled = True

    import redis as redis_lib
    import tiktoken

    import event_stream as es
    import savings_meter as sm
    from memory_service import MemoryService

    print(f"collection={settings.qdrant_collection} run={RUN}")
    service = MemoryService()
    service._get_memory()
    client = service._memory.vector_store.client
    r = redis_lib.Redis.from_url(settings.redis_url)
    enc = tiktoken.get_encoding(settings.savings_tokenizer)

    try:
        # ── [1] write path: real token counts stamped at write ──
        print("\n[1] write path — token_estimate is a real tiktoken count")
        detail_para = (
            "The canary cohort held traffic for thirty minutes while p99 "
            "latency, error budget burn, saturation, GC pauses, connection "
            "pool exhaustion, replica lag, cache hit ratio and queue depth "
            "were compared against the seven-day baseline across all three "
            "regions. The release captain walked the full checklist: schema "
            "migrations verified reversible, feature flags staged dark, "
            "rollback artifact pinned, on-call briefed, dashboards annotated, "
            "and the incident channel pre-warmed. Postmortem notes from the "
            "previous rollout were re-read and the two regressions it caught "
            "were explicitly re-tested before ratification was recorded in "
            "the deploy journal with the full dashboard export attached. "
        )
        contents = [
            f"Deploy decision {i}: the blue-green rollout for service tier {i} "
            f"was ratified after canary analysis. " + detail_para * 2
            for i in range(8)
        ]
        ids_a = [
            seed(service, c, USER_A, category="decision", visibility="private",
                 confidence=0.9, tags=["obs-e2e"])
            for c in contents
        ]
        point = client.retrieve(
            collection_name=settings.qdrant_collection, ids=[ids_a[0]],
            with_payload=True, with_vectors=False,
        )[0]
        stored = point.payload["metadata"]["token_estimate"]
        real = len(enc.encode(contents[0]))
        check("token_estimate equals real tiktoken count", stored == real,
              f"stored={stored} real={real}")

        # ── Boot a REAL in-process API server (uvicorn) for the HTTP/E1/E2
        # surfaces: starlette's TestClient buffers streaming bodies, so an
        # infinite SSE response can only be exercised against a live server.
        print("\n[*] starting in-process API server")
        import httpx
        import uvicorn

        import main as main_mod

        port = 18471
        server = uvicorn.Server(uvicorn.Config(
            main_mod.app, host="127.0.0.1", port=port, log_level="warning",
        ))
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code in (200, 503):
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("in-process API server failed to start")
        tc = httpx.Client(base_url=base, timeout=30)

        # ── [2] index_only recall shows a positive measured savings line ──
        print("\n[2] index_only recall — positive measured savings line")
        # include_shared=False keeps this run hermetic: the shared pool of a
        # long-lived backing graph could otherwise inject foreign hits into
        # the measured baseline.
        resp = tc.post("/v1/search", json={
            "query": "blue-green rollout deploy decision",
            "user_id": USER_A, "index_only": True, "limit": 8,
            "include_shared": False,
        })
        body = resp.json()
        line = body.get("savings") or ""
        detail = body.get("savings_detail") or {}
        print(f"    savings line: {line!r}")
        print(f"    detail: {detail}")
        check("HTTP 200 + rows returned", resp.status_code == 200 and body["results"], "")
        check("savings line present and positive",
              line.startswith("saved ~") and line.endswith("net of overhead")
              and detail.get("net_tokens_saved", 0) > 0, line)
        # Baseline cross-check through the public surface: batch-get the same
        # hits and recompute exactly what hit_tokens uses (stored count, else
        # a real count of the content). NOTE: this batch-get itself appends a
        # net-zero get_memories ledger entry — [3] must still balance.
        hit_ids = [row["id"] for row in body["results"]]
        got = tc.post("/v1/memories/batch-get",
                      json={"ids": hit_ids, "user_id": USER_A}).json()
        expected_baseline = sum(
            (m.get("token_estimate") or len(enc.encode(m["memory"])))
            for m in got["results"]
        )
        check("baseline == sum of stored real counts",
              detail.get("baseline_tokens") == expected_baseline,
              f"baseline={detail.get('baseline_tokens')} expected={expected_baseline}")
        check("served is zero content, overhead covers rows+line",
              detail.get("served_tokens") == 0 and detail.get("overhead_tokens", 0) > 0
              and detail["net_tokens_saved"] == detail["baseline_tokens"] - detail["overhead_tokens"],
              "")
        check("estimated field separate from measured net",
              detail.get("rederivation_savings_estimate", 0) > 0
              and detail["net_tokens_saved"] < detail["rederivation_savings_estimate"], "")

        # ── [3] /v1/metrics totals match the raw ledger sum ──
        print("\n[3] metrics — totals match the append-only ledger")
        mresp = tc.get("/v1/metrics", params={"user_id": USER_A})
        meter = mresp.json()["savings_meter"]
        entries = r.xrange(sm.LEDGER_KEY.format(user_id=USER_A))
        ledger_net = sum(int(fields[b"net"]) for _, fields in entries)
        ledger_baseline = sum(int(fields[b"baseline"]) for _, fields in entries)
        ops = [fields[b"op"].decode() for _, fields in entries]
        check("ledger has tool_schema + search_index + get_memories entries",
              "tool_schema" in ops and "search_index" in ops and "get_memories" in ops,
              str(ops))
        check("metrics user net == ledger sum (signed, schema charge included)",
              meter["user"]["net_tokens_saved"] == ledger_net,
              f"metrics={meter['user']['net_tokens_saved']} ledger={ledger_net}")
        check("metrics user baseline == ledger baseline sum",
              meter["user"]["baseline_tokens"] == ledger_baseline, "")
        check("metrics exposes the per-release schema constant",
              meter["tool_schema_overhead_tokens"] > 0, str(meter["tool_schema_overhead_tokens"]))

        # ── [4] SSE: own event arrives, foreign private never does ──
        print("\n[4] SSE — visibility-filtered live stream")
        frames: list[str] = []
        stop = threading.Event()

        def consume() -> None:
            try:
                with httpx.stream(
                    "GET", f"{base}/v1/stream", params={"user_id": USER_A},
                    timeout=httpx.Timeout(10, read=30),
                ) as stream_resp:
                    for line_ in stream_resp.iter_lines():
                        frames.append(line_)
                        if stop.is_set():
                            break
            except Exception as e:
                frames.append(f"__consumer_error__ {e!r}")

        t = threading.Thread(target=consume, daemon=True)
        t.start()
        time.sleep(2.0)  # let the subscription land

        # Own private write — emitted through the REAL worker emission point
        # (the extension registry's memory_stored fan-out).
        from extensions import ExtensionRegistry

        registry = ExtensionRegistry()
        asyncio.run(registry.emit_event("memory_stored", {
            "user_id": USER_A, "memory_id": "evt-own", "content": "own private fact",
            "category": "decision", "visibility": "private", "source": "worker",
        }))
        # Another user's private write — must never reach USER_A's stream.
        asyncio.run(registry.emit_event("memory_stored", {
            "user_id": USER_B, "memory_id": "evt-foreign", "content": "bob's secret",
            "category": "decision", "visibility": "private", "source": "worker",
        }))
        # Shared event — must arrive.
        es.publish_event("insights_stored", {
            "user_id": USER_B, "visibility": "shared", "count": 1, "pool": "shared",
        })
        deadline = time.time() + 8
        while time.time() < deadline:
            joined = "\n".join(frames)
            if "evt-own" in joined and "insights_stored" in joined:
                break
            time.sleep(0.25)
        stop.set()
        joined = "\n".join(frames)
        check("own memory_stored event received", "evt-own" in joined, "")
        check("foreign private event NOT received", "evt-foreign" not in joined, "")
        check("shared event received", "insights_stored" in joined, "")
        # keep-alive/connected comments prove SSE framing
        check("SSE framing (connected comment)", ": connected" in joined, "")

        # ── [5] surprisal-targeted REM: planted anomaly leads the substrate ──
        print("\n[5] dream sweep — planted anomaly biases the reflection substrate")
        subsystems = [
            "the payments warehouse nightly batch pipeline",
            "the invoice PDF renderer fleet",
            "the customer-facing GraphQL gateway",
            "the fraud-scoring feature store",
            "the settlement reconciliation job",
            "the ledger snapshot compactor",
            "the notification fan-out service",
            "the KYC document OCR queue",
            "the FX rate ingestion daemon",
            "the chargeback evidence uploader",
            "the merchant onboarding webhook relay",
        ]
        for i, name in enumerate(subsystems):
            seed(service,
                 f"Ops fact {i}: {name} completed its scheduled run on time "
                 f"with all health checks green and no operator intervention "
                 f"required this week.",
                 USER_S, category="decision", visibility="private", confidence=0.8)
        anomaly_text = (
            "A wild anomaly: the espresso machine in the Reykjavik office "
            "started broadcasting whale-song MIDI over the guest wifi during "
            "the solstice party."
        )
        anomaly_id = seed(service, anomaly_text, USER_S, category="interaction",
                          visibility="private", confidence=0.8)

        from extensions.dreaming import consolidate as consolidate_mod
        from extensions.dreaming import reflect as reflect_mod
        from extensions.dreaming import sweep as sweep_mod
        from extensions.dreaming.config import DreamingSettings

        captured: list[list[dict]] = []
        real_render = reflect_mod.render_memories_block

        def spy_render(memories, *, include_strength=True):
            if not include_strength:  # the reflection substrate
                captured.append([dict(m) for m in memories])
            return real_render(memories, include_strength=include_strength)

        # Pin the DEEP consolidation pass to a no-op for THIS check: the LLM
        # may legitimately merge synthetic near-duplicate seeds, which would
        # consume the substrate before REM and make the A5 assertion flaky.
        # The REM pass under test (vector fetch → surprisal → bias → real
        # reflection LLM call) runs unmodified.
        real_decide = consolidate_mod.decide

        async def no_actions(batch, llm_call):
            return []

        reflect_mod.render_memories_block = spy_render
        consolidate_mod.decide = no_actions
        try:
            dsettings = DreamingSettings(
                enabled=True, reflection_enabled=True, surprisal_top_k=3,
                vault_pages_enabled=False, identity_card_enabled=False,
                bridges_enabled=False, dynamics_enabled=False,
                obsidian_vault_path="/tmp/obs-e2e-vault",
            )
            run = asyncio.run(sweep_mod.dream_all(
                service=service, settings=dsettings, dry_run=True,
                only_pool=f"user--{USER_S}", force=True,
            ))
        finally:
            reflect_mod.render_memories_block = real_render
            consolidate_mod.decide = real_decide

        pool_report = run.pools[0] if run.pools else None
        check("pool dreamt", pool_report is not None and pool_report.status == "dreamt",
              pool_report.status if pool_report else "no pool")
        check("reflection substrate captured", bool(captured), "")
        if captured:
            substrate = captured[0]
            surprisals = {m["memory_id"]: m.get("surprisal") for m in substrate}
            scored = {k: v for k, v in surprisals.items() if isinstance(v, float)}
            check("staged dicts expose per-memory surprisal",
                  len(scored) == len(substrate), f"{len(scored)}/{len(substrate)}")
            check("planted anomaly has max surprisal",
                  scored and max(scored, key=scored.get) == anomaly_id,
                  f"anomaly={scored.get(anomaly_id)} max={max(scored.values()) if scored else '-'}")
            check("anomaly leads the biased substrate (top-K first)",
                  substrate and substrate[0]["memory_id"] == anomaly_id,
                  substrate[0]["memory_id"] if substrate else "-")

    finally:
        # ── Cleanup: stop the server, drop the collection + Redis keys ──
        print("\n[cleanup]")
        try:
            server.should_exit = True  # noqa: F821 — set when the server booted
            server_thread.join(timeout=10)
            print("  server stopped")
        except Exception:
            pass
        try:
            client.delete_collection(settings.qdrant_collection)
            print(f"  dropped collection {settings.qdrant_collection}")
        except Exception as e:
            print(f"  collection drop failed: {e}")
        try:
            for user in (USER_A, USER_B, USER_S):
                for pattern in (f"ns:savings:{user}", f"ns:savings:totals:{user}"):
                    r.delete(pattern)
                for key in r.scan_iter(f"ns:savings:schema-charged:{user}:*"):
                    r.delete(key)
            for key in r.scan_iter(f"dreaming:*user--{USER_S}*"):
                r.delete(key)
            print("  redis keys cleaned")
        except Exception as e:
            print(f"  redis cleanup failed: {e}")

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'='*60}\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    for name, ok, detail in failed:
        print(f"  FAILED: {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
