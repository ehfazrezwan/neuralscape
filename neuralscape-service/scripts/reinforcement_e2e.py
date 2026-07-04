"""End-to-end exercise of reinforcement-aware dedup (A2, times_derived).

SAFE BY CONSTRUCTION: runs against a dedicated Qdrant collection
(``QDRANT_COLLECTION`` must be overridden to a throwaway name before
invoking — the script refuses to run against the default collection)
and a dedicated test user. Cleans up after itself.

Usage (from neuralscape-service/, with Neo4j/Redis/Qdrant up):

    set -a; source ../.env; set +a
    QDRANT_COLLECTION=reinforcement_e2e \
    uv run python scripts/reinforcement_e2e.py

Exercises the three counter paths plus the recall boost:

1. write-path      → storing the same fact 3× via store_raw dedups onto one
                     row whose times_derived reaches 3
2. dedup cron      → raw duplicate rows inserted behind the write path's
                     back (the batch-extract path has no write-time dedup)
                     are collapsed by dedup_memories, survivor absorbs the
                     dropped counters
3. recall ranking  → the reinforced survivor's search score equals
                     raw_cosine * (1 + K*log1p(times_derived-1)) and the
                     survivor outranks a one-off fact for a topical query
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

USER = "reinforce-e2e"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))


def fetch_metadata(service, mid: str) -> dict:
    from config import settings

    client = service._memory.vector_store.client
    pts = client.retrieve(
        collection_name=settings.qdrant_collection, ids=[mid],
        with_payload=True, with_vectors=False,
    )
    return dict((pts[0].payload or {}).get("metadata") or {}) if pts else {}


def insert_raw_duplicates(service, content: str, n: int) -> list[str]:
    """Insert N rows of the same content directly (no write-path dedup) —
    mimics the batch-extraction path, which inserts blindly."""
    from config import settings  # noqa: F401  (collection guard already ran)

    m = service._get_memory()
    ids = []
    for _ in range(n):
        mid = str(uuid.uuid4())
        embedding = m.embedding_model.embed(content, memory_action="add")
        m.vector_store.insert(
            vectors=[embedding], ids=[mid],
            payloads=[{
                "data": content,
                "hash": hashlib.md5(content.encode()).hexdigest(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "user_id": USER,
                "metadata": {"scope": "global", "category": "preference",
                             "owner_user_id": USER, "visibility": "private"},
            }],
        )
        ids.append(mid)
        time.sleep(0.05)  # distinct created_at ordering
    return ids


def main() -> int:
    from config import settings

    if settings.qdrant_collection == "neuralscape_memories":
        print("REFUSING to run against the default collection. Set QDRANT_COLLECTION.")
        return 2

    from memory_service import (
        REINFORCEMENT_BOOST_K,
        MemoryService,
    )

    print(f"collection={settings.qdrant_collection} user={USER}")
    service = MemoryService()
    service._get_memory()

    print("\n[1/4] write-path reinforcement: same fact stored 3x …")
    fact = "User prefers DuckDB for local analytics work because queries run in-process"
    first_id = None
    for i in range(3):
        res = service.store_raw(
            content=fact, user_id=USER, category="preference", scope="global",
            visibility="private", add_to_graph=False,
        )
        rid = res[0].id
        first_id = first_id or rid
        check(f"store #{i + 1} routed to one row", rid == first_id, rid)
        time.sleep(0.2)  # let Qdrant settle between read-merge-writes
    meta = fetch_metadata(service, first_id)
    check("write-path counter reached 3", meta.get("times_derived") == 3,
          f"times_derived={meta.get('times_derived')}")

    print("\n[2/4] dedup cron: raw duplicate rows collapse onto the survivor …")
    dup_fact = "User always runs the full test suite before merging any pull request"
    dup_ids = insert_raw_duplicates(service, dup_fact, 3)
    result = service.dedup_memories(USER, semantic=False)
    check("exact dedup removed the extra rows",
          result["exact_duplicates_removed"] >= 2, str(result))
    survivor_id = dup_ids[-1]  # newest survives
    surv_meta = fetch_metadata(service, survivor_id)
    td = surv_meta.get("times_derived") or 0
    check("survivor times_derived >= 2 (absorbed dropped counters)", td >= 2,
          f"times_derived={td}")

    print("\n[3/4] one-off control fact …")
    service.store_raw(
        content="User once mentioned trying SQLite for a side project",
        user_id=USER, category="preference", scope="global",
        visibility="private", add_to_graph=False,
    )

    print("\n[4/4] recall ranking boost …")
    query = "what does the user do before merging a pull request?"
    hits = service.search(query=query, user_id=USER, limit=10, include_shared=False)
    by_id = {h.id: h for h in hits}
    surv_hit = by_id.get(survivor_id)
    check("reinforced survivor recalled", surv_hit is not None,
          f"{len(hits)} hits")
    if surv_hit is not None:
        # mechanical boost verification: recompute the raw cosine and check
        # the returned score is exactly raw * (1 + K*log1p(td-1))
        m = service._get_memory()
        emb = m.embedding_model.embed(query, memory_action="search")
        raw_pts = m.vector_store.client.query_points(
            collection_name=settings.qdrant_collection, query=emb,
            limit=20, with_payload=False,
        ).points
        raw = next((p.score for p in raw_pts if str(p.id) == survivor_id), None)
        expected = (raw or 0.0) * (1.0 + REINFORCEMENT_BOOST_K * math.log1p(td - 1))
        check("boost formula applied to survivor score",
              raw is not None and abs(surv_hit.score - expected) < 1e-6,
              f"raw={raw} boosted={surv_hit.score} expected={expected}")
        one_offs = [h for h in hits if h.id not in (survivor_id, first_id)
                    and h.source == "vector"]
        check("reinforced survivor outranks every one-off",
              all((surv_hit.score or 0) > (o.score or 0) for o in one_offs),
              f"survivor={surv_hit.score:.4f} vs " + ", ".join(
                  f"{(o.score or 0):.4f}" for o in one_offs))

    # cleanup
    try:
        service._memory.vector_store.client.delete_collection(settings.qdrant_collection)
        print(f"\ncleaned up collection {settings.qdrant_collection}")
    except Exception as exc:
        print(f"\ncleanup warning: {exc}")

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'=' * 60}\nE2E: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
