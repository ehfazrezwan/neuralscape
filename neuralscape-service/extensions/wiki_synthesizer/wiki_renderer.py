"""Render a synthesized wiki page (frontmatter + body).

Reuses the atomic-write primitives from the conversation_compiler so
both extensions share the same file-locking + temp-file-rename semantics.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from extensions.conversation_compiler.obsidian_writer import (
    _atomic_write,
    _slugify,
)

# Maps the 13 NeuralScape categories to vault folders. Reuses the same
# table the conversation_compiler keys off of so the wiki tree mirrors
# the _raw/ tree's directory structure.
from schemas import CATEGORY_VAULT_PATHS


def community_filename(community_id: str, community_name: str) -> str:
    """Stable filename for a (category, community) wiki page."""
    short_id = community_id.split("-", 1)[0] if community_id else "noid"
    slug = _slugify(community_name) or "untitled"
    return f"community-{short_id}-{slug}.md"


def wiki_page_path(wiki_root: Path, category: str, filename: str) -> Path:
    """Absolute path of a wiki page under ``{wiki_root}/{category_folder}/``."""
    folder = CATEGORY_VAULT_PATHS.get(category, f"Uncategorized/{_slugify(category)}")
    return wiki_root / folder / filename


def wikilink_path(category: str, filename: str) -> str:
    """Vault-root-relative path used in API responses and `[[wikilinks]]`."""
    folder = CATEGORY_VAULT_PATHS.get(category, f"Uncategorized/{_slugify(category)}")
    return f"Wiki/{folder}/{filename}"


_FRONTMATTER_RE = re.compile(r"^---\n(?P<fm>.*?)\n---\n(?P<body>.*)$", re.DOTALL)


def split_existing_page(content: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body_text). Empty input → ({}, "")."""
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
    community_id: str,
    community_name: str,
    group_id: str,
    visibility: str,
    body: str,
    source_memory_ids: list[str],
    graph_node_uuids: list[str],
    synthesis_count: int,
    source_count: int,
    now: datetime | None = None,
) -> str:
    """Return the full page text (frontmatter + body), ready to atomic-write."""
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    fm_lines: list[str] = [
        "---",
        f"title: {title}",
        f"category: {category}",
        f"community_id: {community_id}",
        f"community_name: {community_name}",
        f"visibility: {visibility}",
        f"group_id: {group_id}",
        f"source_memory_ids: {_yaml_list(source_memory_ids)}",
        f"graph_node_uuids: {_yaml_list(graph_node_uuids)}",
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
