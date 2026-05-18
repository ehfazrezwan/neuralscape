"""Render a synthesized wiki page (frontmatter + body).

Reuses the atomic-write primitives from the conversation_compiler so
both extensions share the same file-locking + temp-file-rename semantics.

Page identity is ``(category, group_id)`` — deterministic, one page per
bucket. Filenames are derived from ``group_id``:

- ``"shared"`` → ``shared.md`` (team-wide pool)
- ``"shared--project--<pid>"`` → ``<pid>.md`` (per-project shared pool)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from extensions.conversation_compiler.obsidian_writer import (
    _atomic_write,
    _slugify,
)
from schemas import CATEGORY_VAULT_PATHS, MEMORY_CATEGORIES


_SHARED_PROJECT_PREFIX = "shared--project--"


def category_filename(group_id: str) -> str:
    """Stable filename for a ``(category, group_id)`` wiki page.

    The category is encoded in the folder path (see
    :func:`wiki_page_path`); the filename only needs to disambiguate the
    group within that folder.
    """
    if not group_id or group_id == "shared":
        return "shared.md"
    if group_id.startswith(_SHARED_PROJECT_PREFIX):
        pid = group_id[len(_SHARED_PROJECT_PREFIX):]
        return f"{_slugify(pid) or 'unknown'}.md"
    # Fallback — should not happen for shared synthesis but keeps the
    # function total.
    return f"{_slugify(group_id) or 'unknown'}.md"


def wiki_page_path(wiki_root: Path, category: str, filename: str) -> Path:
    """Absolute path of a wiki page under ``{wiki_root}/{category_folder}/``."""
    folder = CATEGORY_VAULT_PATHS.get(category, f"Uncategorized/{_slugify(category)}")
    return wiki_root / folder / filename


def wikilink_path(category: str, filename: str) -> str:
    """Vault-root-relative path used in API responses and ``[[wikilinks]]``."""
    folder = CATEGORY_VAULT_PATHS.get(category, f"Uncategorized/{_slugify(category)}")
    return f"Wiki/{folder}/{filename}"


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
