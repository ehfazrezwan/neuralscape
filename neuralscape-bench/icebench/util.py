"""Small shared utilities for the ICEBench harness."""

from pathlib import Path


def dir_size_bytes(path: str | Path) -> int:
    """
    Compute the total on-disk size (in bytes) of a directory tree.

    Follows no symlinks (counts the link entry itself, not its target).

    Args:
        path: Directory (or file) path.

    Returns:
        Total size in bytes; 0 if the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size

    total = 0
    for child in p.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            # File may vanish mid-walk (live store); skip it.
            continue
    return total
