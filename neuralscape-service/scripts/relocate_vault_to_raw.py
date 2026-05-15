"""One-time migration: relocate conversation-compiler output under ``_raw/``.

The conversation-compiler used to write everything to the vault root:
``Daily/``, ``Sessions/``, ``Projects/``, ``Decisions/``, ``Research/``,
the per-category folders (``Semantic/`` etc.), and a few index files.
That collides with the new ``Wiki/`` tree produced by the wiki_synthesizer
extension, which wants the vault root reserved for synthesized topical
wikis. This script moves the existing rolled-up content into ``_raw/``
so the two trees can coexist.

Usage:

    # Show what would happen (default — no changes)
    python scripts/relocate_vault_to_raw.py --vault /data/vault

    # Apply the move
    python scripts/relocate_vault_to_raw.py --vault /data/vault --apply

    # Undo (move from _raw/ back to vault root)
    python scripts/relocate_vault_to_raw.py --vault /data/vault --reverse --apply

Both directions are idempotent: re-running them after they've finished is
a no-op. The script never overwrites: if a target already exists at the
destination, the source is left in place and a warning is logged.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("relocate_vault_to_raw")

# Folders the conversation-compiler is known to write to the vault root.
# The category folders (Semantic/Project/Episodic/Procedural/Working/
# Uncategorized) hold per-category entries.md files via append_category_entry.
TOP_LEVEL_FOLDERS = (
    "Daily",
    "Sessions",
    "Projects",
    "Decisions",
    "Research",
    "Semantic",
    "Project",
    "Episodic",
    "Procedural",
    "Working",
    "Uncategorized",
)

# Loose files the writer creates at the vault root.
TOP_LEVEL_FILES = (
    "index.md",
    "log.md",
    "category-index.md",
)

RAW_DIRNAME = "_raw"


def plan_moves(vault: Path, reverse: bool) -> list[tuple[Path, Path]]:
    """Compute the (source, destination) pairs that need to move.

    Returns only pairs whose source exists. Pairs whose destination
    already exists are dropped from the plan with a warning — we never
    overwrite existing content.
    """
    pairs: list[tuple[Path, Path]] = []
    raw_root = vault / RAW_DIRNAME

    for name in (*TOP_LEVEL_FOLDERS, *TOP_LEVEL_FILES):
        if reverse:
            src = raw_root / name
            dst = vault / name
        else:
            src = vault / name
            dst = raw_root / name

        if not src.exists():
            continue
        if dst.exists():
            logger.warning(
                "destination %s already exists — leaving source %s in place",
                dst,
                src,
            )
            continue
        pairs.append((src, dst))
    return pairs


def execute(pairs: list[tuple[Path, Path]], dry_run: bool) -> int:
    """Move every (src, dst) pair. Returns the count of moves performed."""
    moved = 0
    for src, dst in pairs:
        if dry_run:
            logger.info("DRY-RUN would move: %s -> %s", src, dst)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        logger.info("moved: %s -> %s", src, dst)
        moved += 1
    return moved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--vault",
        type=Path,
        required=True,
        help="Path to the Obsidian vault root (e.g. /data/vault).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Without this flag the script only logs what it would do.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Undo a previous migration: move contents from _raw/ back to vault root.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Lower log verbosity to WARNING.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(levelname)s %(name)s: %(message)s",
        level=logging.WARNING if args.quiet else logging.INFO,
    )

    if not args.vault.exists() or not args.vault.is_dir():
        logger.error("vault path does not exist or is not a directory: %s", args.vault)
        return 2

    pairs = plan_moves(args.vault, reverse=args.reverse)
    if not pairs:
        logger.info("nothing to do — vault is already in the target state")
        return 0

    direction = "REVERSE" if args.reverse else "FORWARD"
    logger.info(
        "%s migration: %d top-level item(s) to move under %s",
        direction,
        len(pairs),
        args.vault / (RAW_DIRNAME if not args.reverse else ""),
    )

    moved = execute(pairs, dry_run=not args.apply)
    if not args.apply:
        logger.info("dry-run complete — pass --apply to perform the move")
    else:
        logger.info("migration complete — moved %d item(s)", moved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
