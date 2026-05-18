#!/usr/bin/env python
"""Bulk-promote a slice of memories from PRIVATE to SHARED visibility.

Used once-per-migration when a Neuralscape install moves from single-user
to multi-user. Memories written before the multi-user model have no
`metadata.visibility` set; the server treats them as private. This script
lets you promote a category (or a whole owner's set) to `shared` so they
become readable to all authenticated users in the instance.

Usage:
    # Dry-run: see what would change.
    python scripts/bulk_promote_visibility.py --owner ehfaz --category tech_stack

    # Apply.
    python scripts/bulk_promote_visibility.py --owner ehfaz --category tech_stack --apply

    # Multiple categories at once.
    python scripts/bulk_promote_visibility.py --owner ehfaz \\
        --category tech_stack --category convention --category architecture --apply

    # Promote ALL of an owner's memories (use with care).
    python scripts/bulk_promote_visibility.py --owner ehfaz --apply

Direction can also be flipped with `--to private` to demote.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SERVICE_DIR = _HERE.parent
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--owner",
        required=True,
        help="user_id of the memories' writer (matches the Qdrant payload's user_id field).",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Limit to memories of this category. Repeatable. Omit to match all categories.",
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help="Limit to memories with this project_id. Omit to match all projects.",
    )
    parser.add_argument(
        "--to",
        choices=["private", "shared"],
        default="shared",
        help="Target visibility (default: shared).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually update memories. Without this flag, only previews what would change.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
        help="Max memories to update in one run (default: 10000).",
    )
    args = parser.parse_args(argv)

    # Lazy imports so the script doesn't blow up if deps are missing at parse time.
    from config import settings  # noqa: E402
    from qdrant_client import QdrantClient  # noqa: E402
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue  # noqa: E402

    if not settings.qdrant_url:
        print(
            "error: QDRANT_URL is not set. This script needs a Qdrant server endpoint.",
            file=sys.stderr,
        )
        return 2

    client = QdrantClient(url=settings.qdrant_url)
    collection = settings.qdrant_collection

    # Historical mem0 writes used a double-wrapped payload shape
    # (`metadata.metadata.category` instead of `metadata.category`). The
    # read-side unwrap below handles both, but the server-side scroll
    # filter must too — otherwise `--category` / `--project-id` silently
    # skip legacy rows. Use `should` (OR) to match either key path.
    must: list = [FieldCondition(key="user_id", match=MatchValue(value=args.owner))]
    if args.category:
        must.append(
            Filter(
                should=[
                    FieldCondition(key="metadata.category", match=MatchAny(any=args.category)),
                    FieldCondition(key="metadata.metadata.category", match=MatchAny(any=args.category)),
                ]
            )
        )
    if args.project_id:
        must.append(
            Filter(
                should=[
                    FieldCondition(key="metadata.project_id", match=MatchValue(value=args.project_id)),
                    FieldCondition(key="metadata.metadata.project_id", match=MatchValue(value=args.project_id)),
                ]
            )
        )

    scroll_filter = Filter(must=must)
    candidates: list[tuple[str, dict]] = []
    offset = None
    while len(candidates) < args.limit:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=min(100, args.limit - len(candidates)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in points:
            candidates.append((str(pt.id), pt.payload or {}))
        if next_offset is None:
            break
        offset = next_offset

    target_vis = args.to
    print(f"Found {len(candidates)} candidate memor{'y' if len(candidates) == 1 else 'ies'} "
          f"for owner={args.owner!r}"
          + (f", category in {args.category!r}" if args.category else "")
          + (f", project_id={args.project_id!r}" if args.project_id else "")
          + f". Target visibility: {target_vis!r}.")

    to_update: list[tuple[str, dict]] = []
    skipped_already = 0
    for mid, payload in candidates:
        metadata = payload.get("metadata", {}) or {}
        if isinstance(metadata.get("metadata"), dict):
            metadata = metadata["metadata"]
        if metadata.get("visibility") == target_vis:
            skipped_already += 1
            continue
        to_update.append((mid, payload))

    print(f"  - already at target visibility (skip): {skipped_already}")
    print(f"  - will update: {len(to_update)}")

    if to_update:
        print("\nSample (up to 5):")
        for mid, payload in to_update[:5]:
            md = payload.get("metadata", {}) or {}
            if isinstance(md.get("metadata"), dict):
                md = md["metadata"]
            print(
                f"  {mid}: category={md.get('category')!r} "
                f"current_visibility={md.get('visibility')!r} -> {target_vis!r}"
            )

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to actually update.")
        return 0

    # Apply: read-modify-write so we don't clobber existing metadata keys.
    # Qdrant's set_payload merges only at the top level; nested dicts get
    # replaced wholesale. So we fetch each point, merge our two keys into
    # the existing metadata dict, and write the whole metadata back.
    print(f"\nApplying updates to {len(to_update)} memor{'y' if len(to_update) == 1 else 'ies'}...")
    updated = 0
    for mid, payload in to_update:
        try:
            metadata = dict(payload.get("metadata", {}) or {})
            # Handle the historical double-wrap
            if isinstance(metadata.get("metadata"), dict):
                metadata = dict(metadata["metadata"])
            metadata["visibility"] = target_vis
            metadata.setdefault("owner_user_id", args.owner)
            client.set_payload(
                collection_name=collection,
                payload={"metadata": metadata},
                points=[mid],
            )
            updated += 1
        except Exception as e:
            print(f"  failed for {mid}: {e}", file=sys.stderr)
    print(f"Done: {updated} updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
