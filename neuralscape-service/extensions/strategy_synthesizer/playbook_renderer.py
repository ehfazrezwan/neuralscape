"""Render a synthesized strategy playbook page (frontmatter + body).

Reuses the atomic-write + slugify primitives from the conversation_compiler so
all synthesizers share file-locking/temp-rename semantics.

Page identity is ``(owner, strategy_name)`` — one playbook per strategy per
owner. Layout: ``Playbooks/<owner-or-'shared'>/<strategy-slug>.md``. Keying by
owner keeps one user's strategies from bleeding into another's on a multi-user
instance; single-user deployments just get one owner folder.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from extensions.conversation_compiler.obsidian_writer import _atomic_write, _slugify


def playbook_page_path(playbook_root: Path, owner: str | None, strategy_name: str) -> Path | None:
    """Absolute path of a strategy playbook under ``playbook_root``.

    Returns None when the strategy name has no usable slug (skip the bucket).
    """
    strat_slug = _slugify(strategy_name)
    if not strat_slug:
        return None
    owner_dir = _slugify(owner) if owner else "shared"
    owner_dir = owner_dir or "shared"
    return playbook_root / owner_dir / f"{strat_slug}.md"


def playbook_rel_path(owner: str | None, strategy_name: str) -> str | None:
    """Vault-root-relative path (used as the Neo4j ``strategy_playbook_path``)."""
    strat_slug = _slugify(strategy_name)
    if not strat_slug:
        return None
    owner_dir = (_slugify(owner) if owner else "shared") or "shared"
    return f"Playbooks/{owner_dir}/{strat_slug}.md"


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
    strategy_name: str,
    owner: str | None,
    body: str,
    source_memory_ids: list[str],
    version_number: int,
    source_count: int,
    now: datetime | None = None,
) -> str:
    """Return the full playbook page text (frontmatter + body), ready to atomic-write."""
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    fm_lines: list[str] = [
        "---",
        f"title: {strategy_name}",
        "kind: strategy_playbook",
        f"strategy_name: {strategy_name}",
        f"owner: {owner or 'shared'}",
        f"source_memory_ids: {_yaml_list(source_memory_ids)}",
        f"last_synthesized: {timestamp}",
        f"version_number: {version_number}",
        f"source_count: {source_count}",
        "---",
        "",
        f"# {strategy_name} — Playbook",
        "",
        body.strip(),
        "",
    ]
    return "\n".join(fm_lines)


def write_page(path: Path, content: str) -> None:
    """Atomic-write a fully-rendered playbook page."""
    _atomic_write(path, content)


def _yaml_list(items: list[str]) -> str:
    if not items:
        return "[]"
    return "[" + ", ".join(str(i) for i in items) + "]"
