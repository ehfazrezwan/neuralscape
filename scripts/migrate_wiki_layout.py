#!/usr/bin/env python3
"""One-shot migration: pivot the Wiki/ tree from the old by-topic layout
to the new by-scope layout.

OLD (pre-PR #N): ``Wiki/<TypeGroup>/<CategoryFolder>/<filename>.md``
                  e.g. ``Wiki/Project/Architecture/neuralscape.md``
                       ``Wiki/Episodic/Decisions/shared.md``

NEW (this PR):  ``Wiki/<scope>/<TypeGroup>/<CategoryLeaf>.md``
                  e.g. ``Wiki/neuralscape/General/Architecture.md``
                       ``Wiki/global/Episodic/Decisions.md``

Why use this script instead of re-running the synthesizer fresh?
Each wiki page carries state in its frontmatter (``synthesis_count``,
``last_synthesized``) and in its body (Gemini-polished prose accumulated
across many incremental merges). Re-running from scratch loses all of
that. ``mv``-style migration preserves it — the next synthesis sees
existing content at the new path and increments rather than replaces.

Usage:
    uv run python scripts/migrate_wiki_layout.py --vault <vault_path> [--dry-run]

Idempotent. Re-running after a successful migration is a no-op: files
already at the new layout are skipped silently.

Requires the new wiki_renderer (which exposes the unified path builder)
to be installed in the same Python environment — typically you run this
from the repo root with the same ``uv run`` invocation as the service.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the service importable when this script is run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "neuralscape-service"))

from extensions.wiki_synthesizer.wiki_renderer import wikilink_path  # noqa: E402
from schemas import CATEGORY_VAULT_PATHS  # noqa: E402

logger = logging.getLogger("migrate_wiki_layout")


# Reverse lookup: "Project/Architecture" → "architecture", etc.
# Built once at import time; the assertion inside wiki_renderer already
# guarantees every value is exactly two segments.
_FOLDER_TO_CATEGORY: dict[str, str] = {
    folder: category for category, folder in CATEGORY_VAULT_PATHS.items()
}

# Old layout's first segment is always one of these type-group names.
# Anything else means the file is already in (or unrelated to) the new
# layout and we leave it alone.
_OLD_TYPE_GROUPS: set[str] = {
    "Project",
    "Episodic",
    "Procedural",
    "Semantic",
    "Working",
    "Uncategorized",
}


def reverse_old_path(rel_path: Path) -> tuple[str, str] | None:
    """Parse an OLD-layout vault-relative path into ``(category, group_id)``.

    Returns ``None`` for anything that doesn't match the 4-segment old
    shape, or for files whose ``(TypeGroup, CategoryFolder)`` pair isn't
    in :data:`CATEGORY_VAULT_PATHS`. Caller treats ``None`` as "leave
    this file alone".
    """
    parts = rel_path.parts
    # Expect ('Wiki', '<TypeGroup>', '<CategoryFolder>', '<filename>.md')
    if len(parts) != 4 or parts[0] != "Wiki" or not parts[3].endswith(".md"):
        return None
    type_group, category_folder, filename = parts[1], parts[2], parts[3]
    if type_group not in _OLD_TYPE_GROUPS:
        # Either already migrated (Wiki/<scope>/...) or junk in vault root.
        return None
    folder_value = f"{type_group}/{category_folder}"
    category = _FOLDER_TO_CATEGORY.get(folder_value)
    if category is None:
        # E.g. user-created subfolders, or Uncategorized leaves from the
        # old fallback path. Skip with a warning so they don't silently
        # rot under the old layout.
        logger.warning(
            "skipping %s — (%r, %r) is not in CATEGORY_VAULT_PATHS",
            rel_path, type_group, category_folder,
        )
        return None
    stem = filename[:-3]  # strip ".md"
    if stem == "shared":
        group_id = "shared"
    else:
        # Old layout's per-project files were named ``<slugified_pid>.md``.
        # The slug IS the pid for migration purposes — we re-feed it
        # through the new wikilink_path which will re-slug (idempotent).
        group_id = f"shared--project--{stem}"
    return category, group_id


def plan_moves(wiki_root: Path) -> list[tuple[Path, Path]]:
    """Walk ``wiki_root`` and return ``[(old_abs, new_abs), ...]`` pairs.

    Pages already at new-layout paths, or that don't match the old shape,
    are silently skipped (not included in the result).
    """
    moves: list[tuple[Path, Path]] = []
    if not wiki_root.exists():
        logger.warning("Wiki root does not exist: %s", wiki_root)
        return moves
    for md_path in sorted(wiki_root.rglob("*.md")):
        rel = md_path.relative_to(wiki_root.parent)  # 'Wiki/...'
        parsed = reverse_old_path(rel)
        if parsed is None:
            continue
        category, group_id = parsed
        new_rel = wikilink_path(category, group_id)
        if new_rel is None:
            # Reserved/empty project_id — the original page was in a
            # collision zone we now refuse. Leave it; user can decide.
            logger.warning(
                "skipping %s — new layout rejects this (category=%r, "
                "group_id=%r); page left at old path for manual handling",
                rel, category, group_id,
            )
            continue
        new_abs = wiki_root.parent / new_rel
        if new_abs == md_path:
            # Already at new layout (shouldn't happen given reverse_old_path
            # filters by OLD typegroup names, but guard anyway).
            continue
        moves.append((md_path, new_abs))
    return moves


def apply_moves(moves: list[tuple[Path, Path]], dry_run: bool) -> int:
    """Execute the planned moves. Returns count of successful moves."""
    n_done = 0
    for old, new in moves:
        rel_old = old.relative_to(old.parents[len(old.parents) - 1])  # best-effort
        if new.exists():
            logger.warning(
                "destination already exists, skipping: %s → %s "
                "(merge the two manually if you want one canonical page)",
                old, new,
            )
            continue
        action = "WOULD MOVE" if dry_run else "MOVING"
        logger.info("%s  %s → %s", action, old, new)
        if not dry_run:
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
        n_done += 1
    return n_done


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate Wiki/ from old by-topic layout to new by-scope layout",
    )
    parser.add_argument(
        "--vault",
        type=Path,
        required=True,
        help="Obsidian vault root (the directory that contains Wiki/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without touching the filesystem",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show INFO-level logs (default: WARNING)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose or args.dry_run else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    vault: Path = args.vault.expanduser().resolve()
    wiki_root = vault / "Wiki"
    if not vault.exists():
        logger.error("vault path does not exist: %s", vault)
        return 1

    moves = plan_moves(wiki_root)
    if not moves:
        logger.info(
            "no pages to migrate — vault either has no old-layout files "
            "or migration already ran",
        )
        return 0

    logger.info(
        "%s%d page(s) to migrate under %s",
        "[DRY RUN] " if args.dry_run else "",
        len(moves),
        wiki_root,
    )
    n = apply_moves(moves, dry_run=args.dry_run)
    logger.info(
        "%s %d page(s)%s",
        "would move" if args.dry_run else "moved",
        n,
        "" if args.dry_run else " — next synthesis will increment "
        "synthesis_count instead of resetting to 1",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
