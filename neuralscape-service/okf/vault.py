"""OKF bundle metadata for the dreaming vault.

The librarian/card/diary renderers emit concept pages; this module adds
the *structural* OKF surface on top of them so the vault doubles as a
spec-conformant knowledge bundle:

- a per-folder ``index.md`` listing (progressive disclosure, §6) for each
  bundle directory (``Projects/``, ``Projects/<pid>/``, ``Knowledge/``,
  ``Me/``, ``Dreams/`` and the vault root);
- the bundle-root version marker (§11) in the root index.

Only the dreaming vault v2 tree is treated as the bundle; the
conversation-compiler's ``_raw/`` audit trail predates the OKF surface
and is excluded. All writes are byte-idempotent (unchanged indexes are
not rewritten), so repeated sweeps produce zero churn.
"""

from __future__ import annotations

import logging
from pathlib import Path

from okf import translate

logger = logging.getLogger(__name__)

#: Top-level vault directories that belong to the OKF bundle surface.
BUNDLE_DIRS = ("Projects", "Knowledge", "Me", "Dreams")

_DIR_DESCRIPTIONS = {
    "Projects": "Per-project knowledge hubs, dreamt from shared project pools.",
    "Knowledge": "Team-wide shared knowledge topics.",
    "Me": "The operator's private topics and identity card.",
    "Dreams": "Dream diaries — one consolidation history per memory pool.",
}


def _href(name: str) -> str:
    return name.replace(" ", "%20")


def _concept_entries(directory: Path, *, skip_stems: tuple[str, ...] = ()) -> list[str]:
    """§6 index bullets for the concept documents directly in ``directory``."""
    entries: list[str] = []
    for path in sorted(directory.glob("*.md")):
        if translate.is_reserved_filename(path.name) or path.stem in skip_stems:
            continue
        try:
            fm, _ = translate.parse_document(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        title = translate.concept_title(fm) or path.stem
        description = translate.concept_description(fm)
        entries.append(translate.index_entry(title, _href(path.name), description))
    return entries


def _write_if_changed(path: Path, content: str, atomic_write, out: dict) -> None:
    try:
        if path.exists() and path.read_text(encoding="utf-8") == content:
            out["indexes_unchanged"] += 1
            return
        atomic_write(path, content)
        out["indexes_written"] += 1
    except Exception:
        logger.warning("okf index write failed for %s (non-fatal)", path, exc_info=True)


def refresh_bundle_indexes(vault: Path) -> dict:
    """(Re)generate every bundle directory's index + the root version marker.

    Safe to call after any renderer pass — reads the vault from disk alone
    and rewrites only indexes whose content actually changed.
    """
    from extensions.conversation_compiler.obsidian_writer import _atomic_write

    out = {"indexes_written": 0, "indexes_unchanged": 0}
    vault = Path(vault)
    if not vault.exists():
        return out

    # ── Pool directories (Knowledge/, Me/, Dreams/, Projects/<pid>/) ──
    leaf_dirs: list[tuple[Path, str]] = []
    for name in ("Knowledge", "Me", "Dreams"):
        leaf_dirs.append((vault / name, name))
    projects = vault / "Projects"
    project_dirs: list[Path] = []
    if projects.exists():
        project_dirs = sorted(p for p in projects.iterdir() if p.is_dir())
        for pdir in project_dirs:
            leaf_dirs.append((pdir, pdir.name))

    for directory, label in leaf_dirs:
        if not directory.exists():
            continue
        entries = _concept_entries(directory)
        if not entries:
            continue
        content = translate.render_index([(label, entries)])
        _write_if_changed(directory / translate.INDEX_FILENAME, content, _atomic_write, out)

    # ── Projects/ index: one bullet per project hub ──
    if project_dirs:
        entries = []
        for pdir in project_dirs:
            hub = pdir / f"{pdir.name}.md"
            description = None
            if hub.exists():
                try:
                    fm, _ = translate.parse_document(hub.read_text(encoding="utf-8"))
                    description = translate.concept_description(fm)
                except OSError:
                    pass
            entries.append(
                translate.index_entry(
                    pdir.name,
                    f"{_href(pdir.name)}/{translate.INDEX_FILENAME}",
                    description or f"Knowledge hub for project {pdir.name}.",
                )
            )
        if entries:
            content = translate.render_index([("Projects", entries)])
            _write_if_changed(projects / translate.INDEX_FILENAME, content, _atomic_write, out)

    # ── Root index: concepts at the root + subdirectory listing + version marker ──
    sections: list[tuple[str, list[str]]] = []
    root_entries = _concept_entries(vault)
    if root_entries:
        sections.append(("Concepts", root_entries))
    subdir_entries = []
    for name in BUNDLE_DIRS:
        directory = vault / name
        if directory.exists() and any(directory.rglob("*.md")):
            subdir_entries.append(
                translate.index_entry(
                    name, f"{name}/{translate.INDEX_FILENAME}", _DIR_DESCRIPTIONS.get(name)
                )
            )
    if subdir_entries:
        sections.append(("Subdirectories", subdir_entries))
    if sections:
        content = translate.render_index(sections, is_bundle_root=True)
        _write_if_changed(vault / translate.INDEX_FILENAME, content, _atomic_write, out)
    return out
