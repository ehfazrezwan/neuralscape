"""Render a synthesized wiki page (frontmatter + body).

Reuses the atomic-write primitives from the conversation_compiler so
both extensions share the same file-locking + temp-file-rename semantics.

Page identity is ``(category, group_id)`` — deterministic, one page per
bucket. The on-disk layout is pivoted by scope:

- ``group_id == "shared"`` → ``Wiki/global/<TypeGroup>/<Leaf>.md``
- ``group_id == "shared--project--<pid>"`` → ``Wiki/<pid>/<TypeGroup>/<Leaf>.md``

The "Project" type-group is renamed to "General" everywhere — inside
both per-project trees (where "Project" would be redundant) and the
global tree (consistent rename rather than asymmetric exceptions).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from extensions.conversation_compiler.obsidian_writer import (
    _atomic_write,
    _slugify,
)
from schemas import CATEGORY_VAULT_PATHS, MEMORY_CATEGORIES

logger = logging.getLogger(__name__)


_SHARED_PROJECT_PREFIX = "shared--project--"
_GLOBAL_SCOPE_DIR = "global"
_UNCATEGORIZED_TYPE_GROUP = "Uncategorized"
_TYPE_GROUP_RENAMES: dict[str, str] = {"Project": "General"}

# Reserve every directory name that has independent meaning in the layout.
# A project_id colliding with any of these would either overwrite the
# global tree, masquerade as a type-group folder, or hijack the
# Uncategorized fallback. Better to skip the bucket and log than silently
# shadow data — especially on case-insensitive filesystems (macOS APFS,
# Windows NTFS) where ``Wiki/Project`` and ``Wiki/project`` share storage.
# Entries are lowercased because ``_slugify`` always lowercases its input.
_RESERVED_PROJECT_IDS: set[str] = {
    _GLOBAL_SCOPE_DIR,
    "shared",
    "project",
    "episodic",
    "procedural",
    "semantic",
    "working",
    "general",
    _UNCATEGORIZED_TYPE_GROUP.lower(),
}

# Startup invariant: every CATEGORY_VAULT_PATHS value is exactly
# "TypeGroup/Leaf" — two segments. The new layout depends on splitting
# at this single slash; any future schema drift would silently corrupt
# the wiki tree.
assert all(p.count("/") == 1 for p in CATEGORY_VAULT_PATHS.values()), (
    "wiki_renderer assumes CATEGORY_VAULT_PATHS values are 'TypeGroup/Leaf'"
)


def _resolve_wiki_parts(
    category: str, group_id: str
) -> tuple[str, str, str] | None:
    """Resolve ``(scope_dir, type_group, leaf)`` for a wiki bucket.

    Returns ``None`` when the bucket should be skipped — caller logs
    once and moves on without writing or patching anything.

    Skip conditions:

    - ``group_id`` is neither ``"shared"`` nor a ``shared--project--<pid>``
      shape (the synthesizer should never reach this path, but guard
      against drift).
    - ``project_id`` is empty after stripping the prefix (e.g.
      ``"shared--project--"``).
    - ``project_id`` slug collides with a reserved layout name
      (``global``, ``shared``, type-group names, ``Uncategorized``,
      ``General``).
    """
    if not group_id or group_id == "shared":
        scope_dir = _GLOBAL_SCOPE_DIR
    elif group_id.startswith(_SHARED_PROJECT_PREFIX):
        pid_raw = group_id[len(_SHARED_PROJECT_PREFIX):]
        pid_slug = _slugify(pid_raw)
        if not pid_slug:
            logger.warning(
                "wiki_renderer: empty project_id after stripping prefix from "
                "group_id %r — skipping bucket",
                group_id,
            )
            return None
        if pid_slug in _RESERVED_PROJECT_IDS:
            logger.warning(
                "wiki_renderer: project_id %r collides with a reserved layout "
                "name; skipping bucket for group_id %r",
                pid_slug,
                group_id,
            )
            return None
        scope_dir = pid_slug
    else:
        logger.warning(
            "wiki_renderer: unexpected group_id %r (not a shared synthesis "
            "target) — skipping bucket",
            group_id,
        )
        return None

    folder = CATEGORY_VAULT_PATHS.get(category)
    if folder is None:
        leaf = _slugify(category) or "uncategorized"
        return (scope_dir, _UNCATEGORIZED_TYPE_GROUP, leaf)

    type_group_raw, leaf = folder.split("/", 1)
    type_group = _TYPE_GROUP_RENAMES.get(type_group_raw, type_group_raw)
    return (scope_dir, type_group, leaf)


def wiki_page_path(
    wiki_root: Path, category: str, group_id: str
) -> Path | None:
    """Absolute path of a wiki page under ``wiki_root``.

    Returns ``None`` when the bucket should be skipped (see
    :func:`_resolve_wiki_parts`).
    """
    parts = _resolve_wiki_parts(category, group_id)
    if parts is None:
        return None
    scope_dir, type_group, leaf = parts
    return wiki_root / scope_dir / type_group / f"{leaf}.md"


def wikilink_path(category: str, group_id: str) -> str | None:
    """Vault-root-relative path used in API responses and ``[[wikilinks]]``.

    Returns ``None`` when the bucket should be skipped. The synthesizer
    uses this value verbatim as the Neo4j ``wiki_path`` property, so the
    None case must short-circuit before any graph writeback.
    """
    parts = _resolve_wiki_parts(category, group_id)
    if parts is None:
        return None
    scope_dir, type_group, leaf = parts
    return f"Wiki/{scope_dir}/{type_group}/{leaf}.md"


def category_page_title(category: str, group_id: str) -> str:
    """Human-readable title for a ``(category, group_id)`` page.

    - ``("convention", "shared")`` → ``"Conventions"`` (team-wide)
    - ``("convention", "shared--project--neuralscape")`` →
      ``"Conventions — neuralscape"``
    """
    folder = CATEGORY_VAULT_PATHS.get(category, category)
    # Folder leaves are already nicely capitalized ("Conventions",
    # "Tech-Stack"); use the trailing segment.
    base = folder.rsplit("/", 1)[-1].replace("-", " ")
    if group_id == "shared" or not group_id:
        return base
    if group_id.startswith(_SHARED_PROJECT_PREFIX):
        pid = group_id[len(_SHARED_PROJECT_PREFIX):]
        return f"{base} — {pid}"
    return base


_FRONTMATTER_RE = re.compile(r"^---\n(?P<fm>.*?)\n---\n(?P<body>.*)$", re.DOTALL)


def split_existing_page(content: str) -> tuple[dict, str]:
    """Return ``(frontmatter_dict, body_text)``. Empty input → ``({}, "")``."""
    if not content:
        return {}, ""
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return {}, content
    fm_text = m.group("fm")
    body = m.group("body").lstrip("\n")
    fm: dict[str, str] = {}
    for line in fm_text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        fm[k.strip()] = v.strip()
    return fm, body


def render_page(
    *,
    title: str,
    category: str,
    group_id: str,
    visibility: str,
    body: str,
    source_memory_ids: list[str],
    synthesis_count: int,
    source_count: int,
    now: datetime | None = None,
) -> str:
    """Return the full page text (frontmatter + body), ready to atomic-write."""
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    description = MEMORY_CATEGORIES.get(category, "")
    fm_lines: list[str] = [
        "---",
        f"title: {title}",
        f"category: {category}",
        f"category_description: {description}",
        f"visibility: {visibility}",
        f"group_id: {group_id}",
        f"source_memory_ids: {_yaml_list(source_memory_ids)}",
        f"last_synthesized: {timestamp}",
        f"synthesis_count: {synthesis_count}",
        f"source_count: {source_count}",
        "---",
        "",
        f"# {title}",
        "",
        body.strip(),
        "",
    ]
    return "\n".join(fm_lines)


def write_page(path: Path, content: str) -> None:
    """Atomic-write a fully-rendered wiki page."""
    _atomic_write(path, content)


def _yaml_list(items: list[str]) -> str:
    """Render a YAML flow-sequence: ``[a, b, c]``."""
    if not items:
        return "[]"
    quoted = ", ".join(str(i) for i in items)
    return f"[{quoted}]"
