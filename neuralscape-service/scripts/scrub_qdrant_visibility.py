#!/usr/bin/env python3
"""Scrub legacy stringified-enum visibility values in Qdrant metadata.

Background
----------
Before the ``MemoryVisibility.__str__`` override landed (see PR #56),
the Python 3.11+ ``str(Enum)`` regression caused
``str(MemoryVisibility.SHARED)`` to return ``"MemoryVisibility.SHARED"``
instead of ``"shared"``. That broken string landed in every Qdrant
point's ``metadata.visibility`` field on affected deployments.

The PR fixed three things:
1. New writes — ``__str__`` override emits canonical ``"shared"``.
2. Event consumers — ``normalize_visibility`` tolerates both formats.
3. Vault backfill — ``backfill_vault_writes.py`` replays events for
   affected memories so the consumer's defensive parsing fills the
   missing vault entries.

It did NOT scrub the broken values already sitting in Qdrant. That
hadn't bitten anything yet — UNTIL the wiki synthesizer ran. The
synthesizer's Qdrant scroll filter
(``extensions/wiki_synthesizer/synthesizer.py:374``) matches
``metadata.visibility == "shared"`` strictly, so legacy-format points
become invisible to it. Result: synthesis processes 6 of 92 SHARED
memories on a Windows deployment that ran the old code.

What this script does
---------------------
Scrolls Qdrant for points where ``metadata.visibility`` is in the
legacy stringified-enum format, rewrites just the visibility field to
canonical lowercase, and uses ``client.set_payload(..., key="metadata")``
so the surrounding metadata dict is preserved untouched.

What it does NOT do
-------------------
- Does NOT touch Neo4j. Graphiti nodes don't store visibility directly;
  it's encoded in the ``group_id`` namespace which is unaffected.
- Does NOT touch points whose ``metadata.visibility`` is already
  canonical or NULL — those are no-ops.
- Does NOT migrate ``MemoryVisibility.XYZ`` for unknown XYZ — only the
  two known legacy values (SHARED, PRIVATE) are converted. Anything
  else is logged + skipped so a typo can't silently flip a memory's
  visibility.

Usage
-----
Inside the running ``neuralscape`` container:

    docker exec neuralscape-neuralscape-1 \\
        python scripts/scrub_qdrant_visibility.py [--dry-run]

Idempotent. Safe to re-run. Once everything is canonical the next run
reports ``0 points to update`` and exits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: E402

from config import settings  # noqa: E402

# Map of legacy stringified-enum value → canonical lowercase value.
# Add new entries here if MemoryVisibility ever grows new members.
_LEGACY_TO_CANONICAL: dict[str, str] = {
    "MemoryVisibility.SHARED": "shared",
    "MemoryVisibility.PRIVATE": "private",
}


def _qdrant_client() -> QdrantClient:
    """Build a QdrantClient using the same config the service uses.

    No API key is wired through — the deployed Qdrant runs on the
    internal docker network and config.py exposes no auth field. If
    Qdrant ever moves behind auth, add the key here.
    """
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    # Local on-disk fallback (matches MemoryService config).
    return QdrantClient(path=str(Path(settings.qdrant_path).expanduser()))


def _scrub(client: QdrantClient, collection: str, legacy: str, canonical: str,
           dry_run: bool, batch: int = 200) -> tuple[int, int]:
    """Convert all points where ``metadata.visibility == legacy``.

    Returns ``(scanned, updated)``. ``updated`` is 0 in dry-run mode.
    """
    flt = Filter(must=[
        FieldCondition(
            key="metadata.visibility",
            match=MatchValue(value=legacy),
        ),
    ])

    scanned = 0
    updated = 0
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=batch,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            break
        scanned += len(points)
        ids = [p.id for p in points]

        if not dry_run:
            # ``key="metadata"`` scopes the payload merge to the metadata
            # sub-object, preserving every other metadata.* field. Without
            # this the call would replace the entire metadata dict with
            # just {"visibility": canonical} and nuke category/scope/etc.
            client.set_payload(
                collection_name=collection,
                payload={"visibility": canonical},
                points=ids,
                key="metadata",
                wait=True,
            )
            updated += len(ids)

        if next_offset is None:
            break
        offset = next_offset

    return scanned, updated


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan + report, but do not write")
    parser.add_argument("--collection", default=None,
                        help="Override collection name (default: from config)")
    args = parser.parse_args()

    client = _qdrant_client()
    collection = args.collection or settings.qdrant_collection

    print(f"Collection: {collection}")
    if args.dry_run:
        print("(dry-run — no writes)")

    total_scanned = 0
    total_updated = 0
    for legacy, canonical in _LEGACY_TO_CANONICAL.items():
        scanned, updated = _scrub(client, collection, legacy, canonical,
                                  dry_run=args.dry_run)
        print(f"  {legacy!r} → {canonical!r}: scanned={scanned} updated={updated}")
        total_scanned += scanned
        total_updated += updated

    print()
    print(f"Summary: scanned={total_scanned} updated={total_updated}")
    if args.dry_run and total_scanned > 0:
        print("Re-run without --dry-run to apply the rewrites.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
