#!/usr/bin/env python3
"""One-off backfill: stamp owner and tier on conversation-extracted rows.

Background
----------
Conversation extraction (``_batch_store_facts``) used to write rows with no
``metadata.owner_user_id`` and no ``metadata.visibility`` (the owner existed
only as the payload's top-level ``user_id``). New rows are now stamped
(``owner_user_id`` + ``visibility=private``); this stamps the rows already
stored so explicit ``metadata.visibility=private`` filters (list/retag by
visibility, private-only store filters) find them and old and new rows behave
the same. Readers already treat an unstamped row as private to its writer, so
the stamp changes no access decision.

What it does
------------
For every point whose ``metadata.source`` is in ``--source`` (default
``conversation``, the provenance of extracted rows) and which lacks
``metadata.owner_user_id`` and/or ``metadata.visibility``:

- ``owner_user_id`` <- the payload's top-level ``user_id`` (skipped, and
  reported, when the row has none);
- ``visibility``    <- ``private``.

Only missing fields are written (an existing ``shared``/``standard`` tier or a
legacy stringified value is never touched). Writes are batched nested-key
``set_payload(key="metadata")`` calls, so the rest of the metadata is
preserved. Idempotent: a second run reports 0 rows to stamp.

Usage
-----
Dry run (default, writes nothing), then apply, inside the API container:

    python scripts/backfill_extracted_owner_visibility.py
    python scripts/backfill_extracted_owner_visibility.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import (  # noqa: E402
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchAny,
    PayloadField,
)

PRIVATE = "private"


def _meta(payload: dict) -> dict:
    meta = payload.get("metadata") or {}
    if isinstance(meta.get("metadata"), dict):
        meta = meta["metadata"]
    return meta


def backfill(
    client: QdrantClient,
    collection: str,
    *,
    apply: bool = False,
    batch_size: int = 200,
    sources: tuple[str, ...] = ("conversation",),
) -> dict:
    """Stamp owner/tier on extracted rows that lack them. Returns a summary.

    ``rows_to_stamp`` counts rows needing at least one field; ``stamped`` is
    that count when ``apply`` is set (0 on a dry run).
    """
    flt = Filter(
        must=[FieldCondition(key="metadata.source", match=MatchAny(any=list(sources)))],
        should=[
            IsEmptyCondition(is_empty=PayloadField(key="metadata.owner_user_id")),
            IsEmptyCondition(is_empty=PayloadField(key="metadata.visibility")),
        ],
    )
    summary = {
        "scanned": 0, "rows_to_stamp": 0, "stamped": 0,
        "owner_to_set": 0, "visibility_to_set": 0, "skipped_no_user_id": 0,
        "dry_run": not apply,
    }
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        # Group points by the exact patch they need, one set_payload per group.
        groups: dict[tuple, list] = {}
        for pt in points:
            summary["scanned"] += 1
            payload = pt.payload or {}
            meta = _meta(payload)
            patch: dict = {}
            if not meta.get("owner_user_id"):
                if payload.get("user_id"):
                    patch["owner_user_id"] = payload["user_id"]
                else:
                    summary["skipped_no_user_id"] += 1
            if not meta.get("visibility"):
                patch["visibility"] = PRIVATE
            if not patch:
                continue
            summary["rows_to_stamp"] += 1
            summary["owner_to_set"] += "owner_user_id" in patch
            summary["visibility_to_set"] += "visibility" in patch
            # the owner differs per row, so only the visibility-only patch batches
            key = ("vis",) if set(patch) == {"visibility"} else ("row", str(pt.id))
            groups.setdefault(key, []).append((pt.id, patch))
        if apply:
            for key, members in groups.items():
                client.set_payload(
                    collection_name=collection,
                    payload=members[0][1],  # identical across a group by construction
                    points=[pid for pid, _ in members],
                    key="metadata",
                    wait=True,
                )
                summary["stamped"] += len(members)
        if next_offset is None or not points:
            break
        offset = next_offset
    return summary


def _qdrant_client() -> QdrantClient:
    from config import settings

    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(path=str(Path(settings.qdrant_path).expanduser()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write the stamps (default: dry run)")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--source", action="append", default=None,
                    help="metadata.source value to backfill (repeatable; default: conversation)")
    args = ap.parse_args(argv)

    from config import settings

    summary = backfill(
        _qdrant_client(), settings.qdrant_collection,
        apply=args.apply, batch_size=args.batch_size,
        sources=tuple(args.source or ("conversation",)),
    )
    for k, v in summary.items():
        print(f"{k}: {v}")
    if summary["dry_run"]:
        print("dry run: nothing written; re-run with --apply to stamp.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
