"""OKF §9 conformance walker.

A bundle is conformant with OKF v0.1 when:

1. every non-reserved ``.md`` file contains a parseable YAML frontmatter
   block;
2. every frontmatter block contains a non-empty ``type`` field;
3. reserved files (``index.md``, ``log.md``) follow the §6/§7 structure
   when present — index files carry no frontmatter (except the bundle
   root, which may carry only the version marker) and list entries under
   headings; log files open with a title heading and group entries under
   ``## YYYY-MM-DD`` date headings.

Used by the unit suite (over tmp vaults / built bundles) and by the E2E
harness (over an exported zip re-expanded to disk). Returns violations as
strings instead of raising, so a test can assert the list is empty and
print every problem at once.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from okf import translate


def _check_concept(rel: str, text: str) -> list[str]:
    problems: list[str] = []
    fm, _body = translate.parse_document(text)
    if not fm:
        if text.lstrip().startswith("---"):
            problems.append(f"{rel}: frontmatter block is not parseable YAML")
        else:
            problems.append(f"{rel}: missing frontmatter block")
        return problems
    if not translate.concept_type(fm):
        problems.append(f"{rel}: frontmatter has no non-empty required type field")
    return problems


def _check_index(rel: str, text: str, *, is_root: bool) -> list[str]:
    problems: list[str] = []
    has_fm = text.lstrip().startswith("---")
    if has_fm:
        fm, body = translate.parse_document(text)
        if not is_root:
            problems.append(f"{rel}: index files must not contain frontmatter (§6)")
        elif not translate.has_version_marker(fm):
            problems.append(
                f"{rel}: bundle-root index frontmatter must carry the version marker (§11)"
            )
    else:
        body = text
    # §6 body: one or more `# heading` sections over `* [..](..)` bullets.
    headings = [l for l in body.splitlines() if l.startswith("# ")]
    bullets = [l for l in body.splitlines() if l.lstrip().startswith("* ")]
    if not headings:
        problems.append(f"{rel}: index has no section heading (§6)")
    if bullets and not all(re.match(r"^\s*\* \[[^\]]*\]\([^)]*\)", b) for b in bullets):
        problems.append(f"{rel}: index bullets must be markdown links (§6)")
    return problems


def _check_log(rel: str, text: str) -> list[str]:
    problems: list[str] = []
    if text.lstrip().startswith("---"):
        problems.append(f"{rel}: log files must not contain frontmatter (§7)")
        return problems
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines or not lines[0].startswith("# "):
        problems.append(f"{rel}: log must open with a title heading (§7)")
    date_headings = [l for l in lines if l.startswith("## ")]
    if not date_headings:
        problems.append(f"{rel}: log has no date headings (§7)")
    for heading in date_headings:
        if not translate.LOG_DATE_HEADING_RE.match(heading):
            problems.append(f"{rel}: log date heading {heading!r} is not ISO 8601 (§7)")
    dates = [
        translate.LOG_DATE_HEADING_RE.match(h).group(1)
        for h in date_headings
        if translate.LOG_DATE_HEADING_RE.match(h)
    ]
    if dates != sorted(dates, reverse=True):
        problems.append(f"{rel}: log dates must run newest-first (§7)")
    return problems


def check_files(files: Mapping[str, str]) -> list[str]:
    """§9 conformance over ``{relative_path: text}``. Empty list = conformant."""
    problems: list[str] = []
    for rel in sorted(files):
        if not rel.endswith(".md"):
            continue
        text = files[rel]
        name = rel.rsplit("/", 1)[-1]
        if name == translate.INDEX_FILENAME:
            problems.extend(_check_index(rel, text, is_root=(rel == translate.INDEX_FILENAME)))
        elif name == translate.LOG_FILENAME:
            problems.extend(_check_log(rel, text))
        else:
            problems.extend(_check_concept(rel, text))
    return problems


def check_bundle(root: Path, *, ignore: tuple[str, ...] = ()) -> list[str]:
    """§9 conformance over a bundle directory on disk.

    ``ignore`` names top-level directories excluded from the walk (e.g. a
    vault's ``_raw/`` audit-trail tree, which predates the OKF surface).
    """
    files: dict[str, str] = {}
    root = Path(root)
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        top = rel.split("/", 1)[0]
        if top in ignore or any(part.startswith(".") for part in rel.split("/")):
            continue
        try:
            files[rel] = path.read_text(encoding="utf-8")
        except OSError as e:
            files[rel] = ""
            _ = e
    return check_files(files)
