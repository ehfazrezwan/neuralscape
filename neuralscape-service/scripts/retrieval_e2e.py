"""End-to-end exercise of retrieval economics (C1 index-first recall + batch
get, C2 timeline) against real Qdrant/Neo4j/Redis.

SAFE BY CONSTRUCTION: runs against a dedicated Qdrant collection
(``QDRANT_COLLECTION`` must be overridden to a throwaway name — the script
refuses to run against the default collection) and dedicated test users.
Cleans up after itself (drops the collection).

Usage (from neuralscape-service/, with Neo4j/Redis/Qdrant up):

    set -a; source ../.env; set +a
    QDRANT_COLLECTION=retrieval_e2e \
    uv run python scripts/retrieval_e2e.py

Exercises:

1. write path      → store_raw stamps title + token_estimate at write time
2. index recall    → search results render as compact index rows within the
                     ~100-token budget, titles/estimates from stored metadata
3. batch get       → get_memories_by_ids returns full payloads (v2 +
                     provenance fields) for readable ids; unknown ids and
                     another user's private ids land in `missing`
4. timeline        → a mid-anchor window returns exactly ±depth memories in
                     chronological order (real Qdrant order_by scroll with
                     the DATETIME payload index), excluding tombstoned rows
                     and other users' private rows, including shared rows;
                     query anchors resolve to the best vector hit
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

USER = "retrieval-e2e"
OTHER = "retrieval-e2e-other"
NOW = datetime.now(timezone.utc)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))


def backdate(service, mid: str, dt: datetime) -> None:
    """Patch a stored memory's created_at so the seed spans a time range."""
    from config import settings

    service._memory.vector_store.client.set_payload(
        collection_name=settings.qdrant_collection,
        payload={"created_at": dt.isoformat()},
        points=[mid],
    )


def seed(service, content: str, user_id: str, dt: datetime, **kw) -> str:
    [resp] = service.store_raw(
        content=content, user_id=user_id, add_to_graph=False, **kw
    )
    backdate(service, resp.id, dt)
    return resp.id


def main() -> int:
    from config import settings

    if settings.qdrant_collection == "neuralscape_memories":
        print("REFUSING to run against the default collection. Set QDRANT_COLLECTION.")
        return 2

    from index_format import distill_title, estimate_tokens, index_row
    from memory_service import MemoryService

    print(f"collection={settings.qdrant_collection} user={USER}")
    service = MemoryService()
    service._get_memory()
    client = service._memory.vector_store.client

    try:
        # ── Seed: 20 user memories spread hourly, oldest → newest ──
        print("\n[1] seeding 20 memories over a 20-hour spread")
        ids: list[str] = []
        for i in range(20):
            mid = seed(
                service,
                f"Deploy pipeline event number {i}: step {i} of the blue-green "
                f"rollout completed with detailed notes attached. The canary "
                f"cohort held at {i}% traffic for thirty minutes while error "
                f"budgets, saturation, and p99 latency were compared against "
                f"the baseline before the traffic switch was ratified by the "
                f"release captain and recorded in the deploy journal.",
                USER,
                NOW - timedelta(hours=20 - i),
                category="decision",
                observation_type="decision",
                visibility="private",
                confidence=0.9,
                tags=["e2e"],
            )
            ids.append(mid)

        # Interlopers inside the window: another user's private row (must
        # never surface), a shared row (must surface), a tombstoned row
        # (must be excluded from timeline).
        foreign_private = seed(
            service, "Other user's private deploy secret.", OTHER,
            NOW - timedelta(hours=10, minutes=30),
            category="decision", visibility="private",
        )
        shared_id = seed(
            service, "Team-shared deploy retro conclusion for everyone.", OTHER,
            NOW - timedelta(hours=9, minutes=30),
            category="decision", visibility="shared",
        )
        tombstoned = seed(
            service, "Consolidated-away deploy note (tombstoned).", USER,
            NOW - timedelta(hours=8, minutes=30),
            category="decision", visibility="private",
        )
        pts = client.retrieve(
            collection_name=settings.qdrant_collection, ids=[tombstoned],
            with_payload=True, with_vectors=False,
        )
        meta = dict((pts[0].payload or {}).get("metadata") or {})
        meta["dream_tombstoned"] = True
        client.set_payload(
            collection_name=settings.qdrant_collection,
            payload={"metadata": meta}, points=[tombstoned],
        )
        check("seeded 20 + 3 interlopers", len(ids) == 20)

        # ── Write-time stamping ──
        print("\n[2] write-time title + token_estimate")
        pts = client.retrieve(
            collection_name=settings.qdrant_collection, ids=[ids[0]],
            with_payload=True, with_vectors=False,
        )
        m0 = (pts[0].payload or {}).get("metadata") or {}
        expected_title = distill_title((pts[0].payload or {}).get("data", ""))
        check("title stamped at write time",
              m0.get("title") == expected_title and expected_title != "(untitled)",
              repr(m0.get("title")))
        check("token_estimate stamped", isinstance(m0.get("token_estimate"), int)
              and m0["token_estimate"] > 10, str(m0.get("token_estimate")))

        # ── Index-only recall ──
        print("\n[3] index-only recall (compact rows)")
        results = service.search(
            query="blue-green deploy rollout", user_id=USER, limit=10
        )
        vec = [r for r in results if r.source == "vector"]
        check("recall returns vector hits", len(vec) >= 5, f"{len(vec)} hits")
        rows = [index_row(r) for r in vec]
        check("rows carry stored titles",
              all(r["title"] and r["title"] != "(untitled)" for r in rows))
        check("rows carry token estimates", all(r.get("tokens", 0) > 10 for r in rows))
        rendered = [json.dumps(r, ensure_ascii=False) for r in rows]
        check("every rendered row under ~100 tokens",
              all(len(s) < 400 for s in rendered),
              f"max {max(len(s) for s in rendered)} chars")
        full_payload_len = sum(len(json.dumps(r.model_dump(exclude_none=True), default=str)) for r in vec)
        index_len = sum(len(s) for s in rendered)
        check("index is meaningfully cheaper than full payloads",
              index_len * 3 < full_payload_len,
              f"{index_len} vs {full_payload_len} chars")

        # ── Batch get ──
        print("\n[4] get_memories batch fetch")
        pick = [ids[3], ids[7], ids[11]]
        ghost = str(uuid.uuid4())
        out = service.get_memories_by_ids(pick + [ghost, foreign_private], USER)
        check("returns full payloads for own ids",
              [r.id for r in out["results"]] == pick)
        got = out["results"][0]
        check("full payload carries v2 + provenance fields",
              got.category == "decision" and got.confidence == 0.9
              and got.observation_type == "decision" and got.visibility == "private"
              and got.owner_user_id == USER and got.title is not None
              and got.token_estimate == estimate_tokens(got.memory))
        check("unknown id lands in missing", ghost in out["missing"])
        check("another user's PRIVATE id is unreadable (reported missing)",
              foreign_private in out["missing"])
        out2 = service.get_memories_by_ids([shared_id], USER)
        check("another user's SHARED id is readable",
              [r.id for r in out2["results"]] == [shared_id])

        # ── Timeline around a mid anchor ──
        print("\n[5] timeline around a mid anchor (depth=5)")
        anchor = ids[10]  # NOW-10h
        tl = service.timeline(anchor, user_id=USER, depth=5)
        check("timeline resolved", tl is not None and tl["anchor_id"] == anchor)
        mems = tl["memories"]
        times = [m.created_at for m in mems]
        check("strict chronological order", times == sorted(times))
        got_ids = [m.id for m in mems]
        check("anchor present", anchor in got_ids)
        # Anchor sits at −10h. Nearest 5 before: ids[5..9] (−15h..−11h) — the
        # foreign private row at −10.5h is in range but must be excluded.
        # Nearest 5 after: shared (−9.5h), ids[11] (−9h), ids[12..14] — the
        # tombstoned row at −8.5h is in range but must be excluded.
        before = got_ids[: got_ids.index(anchor)]
        after = got_ids[got_ids.index(anchor) + 1:]
        check("±depth bounds respected", len(before) == 5 and len(after) == 5,
              f"{len(before)} before / {len(after)} after")
        check("nearest-before are the right neighbors (foreign private displaced)",
              before == ids[5:10], str(before))
        check("nearest-after are the right neighbors (shared included, tombstone displaced)",
              after == [shared_id, ids[11], ids[12], ids[13], ids[14]], str(after))
        check("tombstoned row excluded (−8.5h was in range)",
              tombstoned not in got_ids)
        check("foreign private row excluded (−10.5h was in range)",
              foreign_private not in got_ids)

        # ── Timeline anchored by query ──
        print("\n[6] timeline anchored by a query")
        tlq = service.timeline("blue-green rollout step 15", user_id=USER, depth=2)
        check("query anchor resolves to a vector hit", tlq is not None)
        if tlq:
            check("query-anchored window is chronological",
                  [m.created_at for m in tlq["memories"]]
                  == sorted(m.created_at for m in tlq["memories"]))

        # ── Edge: anchor at the very start of history ──
        tl0 = service.timeline(ids[0], user_id=USER, depth=3)
        idx0 = [m.id for m in tl0["memories"]].index(ids[0])
        check("history-start anchor has empty before-side", idx0 == 0)

    finally:
        print("\n[cleanup] dropping collection")
        try:
            client.delete_collection(collection_name=settings.qdrant_collection)
        except Exception as e:
            print(f"  cleanup failed: {e}")
        service.close()

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'='*60}\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        for name, _, detail in failed:
            print(f"  FAILED: {name} {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
