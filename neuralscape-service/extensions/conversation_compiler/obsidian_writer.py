"""Obsidian vault I/O for the conversation-compiler extension.

All file writes use a temp-file + atomic rename pattern to prevent
corruption from concurrent access. Frontmatter is managed with
standard YAML delimiters.
"""

import fcntl
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from .config import compiler_settings

logger = structlog.get_logger(__name__)


def _slugify(text: str) -> str:
    """Convert text to a kebab-case slug suitable for filenames."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")[:80]


def _ensure_dir(path: Path) -> None:
    """Create directory and parents if they don't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically using temp file + rename.

    Uses file locking to prevent concurrent write corruption.
    """
    _ensure_dir(path.parent)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_append(path: Path, content: str) -> None:
    """Append content to path with file locking."""
    _ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(content)
        f.flush()
        os.fsync(f.fileno())


def _read_file(path: Path) -> str:
    """Read file content, returning empty string if file doesn't exist."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _build_frontmatter(
    title: str,
    tags: list[str] | None = None,
    date: str | None = None,
    source_count: int | None = None,
    compiled: bool | None = None,
    extra: dict | None = None,
) -> str:
    """Build YAML frontmatter block."""
    lines = ["---"]
    lines.append(f"title: {title}")
    if date:
        lines.append(f"date: {date}")
    if tags:
        lines.append(f"tags: [{', '.join(tags)}]")
    if source_count is not None:
        lines.append(f"source_count: {source_count}")
    if compiled is not None:
        lines.append(f"compiled: {str(compiled).lower()}")
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _update_frontmatter_field(content: str, field: str, value: str) -> str:
    """Update a single field in existing frontmatter, or add it."""
    fm_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not fm_match:
        return content

    fm_body = fm_match.group(1)
    field_re = re.compile(rf"^{re.escape(field)}:.*$", re.MULTILINE)
    if field_re.search(fm_body):
        fm_body = field_re.sub(f"{field}: {value}", fm_body)
    else:
        fm_body += f"\n{field}: {value}"

    return f"---\n{fm_body}\n---\n" + content[fm_match.end() :]


class ObsidianWriter:
    """Handles all vault I/O for the conversation-compiler extension."""

    def __init__(self, vault_path: Path | None = None) -> None:
        self.vault = vault_path or compiler_settings.vault_path
        logger.info("ObsidianWriter initialized", vault=str(self.vault))

    # ── Daily logs ────────────────────────────────

    def append_daily_log(self, date: str, entries: list[dict]) -> str:
        """Append extracted entries to the daily log file.

        Args:
            date: ISO date string (YYYY-MM-DD).
            entries: List of dicts with 'category', 'content', 'time', 'session_id' keys.

        Returns:
            Path to the daily log file (relative to vault).
        """
        rel_path = f"Daily/{date}.md"
        path = self.vault / rel_path
        existing = _read_file(path)

        if not existing:
            # Create with frontmatter
            content = _build_frontmatter(
                title=f"Daily Log — {date}",
                tags=["daily-log", "auto-generated"],
                date=date,
                compiled=False,
            )
            content += f"# Daily Log — {date}\n\n"
        else:
            content = ""

        for entry in entries:
            time_str = entry.get("time", "")
            category = entry.get("category", "uncategorized")
            fact = entry.get("content", "")
            session = entry.get("session_id", "")
            content += f"- **[{time_str}]** `{category}` {fact}"
            if session:
                content += f" _(session: {session})_"
            content += "\n"

        if existing:
            _atomic_append(path, content)
        else:
            _atomic_write(path, content)

        logger.info("Daily log updated", date=date, entries=len(entries))
        return rel_path

    def is_daily_log_compiled(self, date: str) -> bool:
        """Check if a daily log has already been compiled."""
        path = self.vault / f"Daily/{date}.md"
        content = _read_file(path)
        if not content:
            return False
        return "compiled: true" in content.split("---")[1] if "---" in content else False

    def mark_daily_log_compiled(self, date: str) -> None:
        """Mark a daily log as compiled by updating its frontmatter."""
        path = self.vault / f"Daily/{date}.md"
        content = _read_file(path)
        if not content:
            return
        updated = _update_frontmatter_field(content, "compiled", "true")
        updated = _update_frontmatter_field(
            updated, "compiled_at", datetime.now().isoformat()
        )
        _atomic_write(path, updated)

    def get_daily_log_entries(self, date: str) -> list[dict]:
        """Parse entries from a daily log file.

        Returns:
            List of dicts with 'time', 'category', 'content', 'session_id'.
        """
        path = self.vault / f"Daily/{date}.md"
        content = _read_file(path)
        if not content:
            return []

        entries = []
        # Match lines like: - **[HH:MM]** `category` content _(session: xxx)_
        pattern = re.compile(
            r"^- \*\*\[([^\]]*)\]\*\* `(\w+)` (.+?)(?:\s*_\(session: ([^)]+)\)_)?$",
            re.MULTILINE,
        )
        for m in pattern.finditer(content):
            entries.append(
                {
                    "time": m.group(1),
                    "category": m.group(2),
                    "content": m.group(3).strip(),
                    "session_id": m.group(4) or "",
                }
            )
        return entries

    def list_daily_logs(self) -> list[str]:
        """List all daily log dates (YYYY-MM-DD) in the vault."""
        daily_dir = self.vault / "Daily"
        if not daily_dir.exists():
            return []
        dates = []
        for f in sorted(daily_dir.glob("*.md")):
            # Extract date from filename
            date_str = f.stem
            if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
                dates.append(date_str)
        return dates

    def list_uncompiled_dates(self) -> list[str]:
        """List daily log dates that haven't been compiled yet."""
        return [d for d in self.list_daily_logs() if not self.is_daily_log_compiled(d)]

    # ── Session summaries ─────────────────────────

    def write_session_summary(self, date: str, summary: str) -> str:
        """Write or update a session summary for a given date.

        Returns:
            Relative path within the vault.
        """
        rel_path = f"Sessions/{date}.md"
        path = self.vault / rel_path
        content = _build_frontmatter(
            title=f"Session Summary — {date}",
            tags=["session-summary", "auto-generated"],
            date=date,
        )
        content += f"# Session Summary — {date}\n\n{summary}\n"
        _atomic_write(path, content)
        logger.info("Session summary written", date=date)
        return rel_path

    # ── Project pages ─────────────────────────────

    def update_project_page(self, project: str, content: str) -> str:
        """Update a project's README page.

        Returns:
            Relative path within the vault.
        """
        slug = _slugify(project)
        rel_path = f"Projects/{slug}/README.md"
        path = self.vault / rel_path
        existing = _read_file(path)

        if existing:
            # Preserve frontmatter, replace body
            fm_match = re.match(r"^(---\n.*?\n---\n)", existing, re.DOTALL)
            if fm_match:
                fm = _update_frontmatter_field(
                    fm_match.group(0) + "\n",
                    "updated",
                    datetime.now().strftime("%Y-%m-%d"),
                )
                full_content = fm + content + "\n"
            else:
                full_content = content + "\n"
        else:
            full_content = _build_frontmatter(
                title=project,
                tags=["project", slug, "auto-generated"],
                date=datetime.now().strftime("%Y-%m-%d"),
            )
            full_content += f"# {project}\n\n{content}\n"

        _atomic_write(path, full_content)
        logger.info("Project page updated", project=project)
        return rel_path

    # ── Decisions ──────────────────────────────────

    def write_decision(self, slug: str, content: str) -> str:
        """Create or update a decision record.

        Returns:
            Relative path within the vault.
        """
        safe_slug = _slugify(slug)
        rel_path = f"Decisions/{safe_slug}.md"
        path = self.vault / rel_path
        existing = _read_file(path)

        if existing:
            updated = _update_frontmatter_field(
                existing, "updated", datetime.now().strftime("%Y-%m-%d")
            )
            # Replace body after frontmatter
            fm_match = re.match(r"^(---\n.*?\n---\n)", updated, re.DOTALL)
            if fm_match:
                full_content = fm_match.group(0) + "\n" + content + "\n"
            else:
                full_content = content + "\n"
        else:
            full_content = _build_frontmatter(
                title=slug.replace("-", " ").title(),
                tags=["decision", "auto-generated"],
                date=datetime.now().strftime("%Y-%m-%d"),
            )
            full_content += content + "\n"

        _atomic_write(path, full_content)
        logger.info("Decision written", slug=safe_slug)
        return rel_path

    # ── Research ──────────────────────────────────

    def write_research(self, topic: str, content: str) -> str:
        """Create or update a research article.

        Returns:
            Relative path within the vault.
        """
        safe_slug = _slugify(topic)
        rel_path = f"Research/{safe_slug}.md"
        path = self.vault / rel_path
        existing = _read_file(path)

        if existing:
            updated = _update_frontmatter_field(
                existing, "updated", datetime.now().strftime("%Y-%m-%d")
            )
            fm_match = re.match(r"^(---\n.*?\n---\n)", updated, re.DOTALL)
            if fm_match:
                full_content = fm_match.group(0) + "\n" + content + "\n"
            else:
                full_content = content + "\n"
        else:
            full_content = _build_frontmatter(
                title=topic.replace("-", " ").title(),
                tags=["research", "auto-generated"],
                date=datetime.now().strftime("%Y-%m-%d"),
            )
            full_content += content + "\n"

        _atomic_write(path, full_content)
        logger.info("Research article written", topic=safe_slug)
        return rel_path

    # ── Index ─────────────────────────────────────

    def update_index(self, entries: list[dict]) -> str:
        """Update the vault index with new or changed pages.

        Args:
            entries: List of dicts with 'path', 'title', 'type' keys.

        Returns:
            Relative path to the index file.
        """
        rel_path = "index.md"
        path = self.vault / rel_path
        existing = _read_file(path)

        # Parse existing index entries into a dict keyed by path
        existing_entries: dict[str, str] = {}
        if existing:
            for line in existing.split("\n"):
                m = re.match(r"^- \[\[([^\]]+)\]\]\s*—\s*(.+)$", line)
                if m:
                    existing_entries[m.group(1)] = m.group(2).strip()

        # Merge new entries
        for entry in entries:
            entry_path = entry.get("path", "")
            title = entry.get("title", "")
            etype = entry.get("type", "")
            existing_entries[entry_path] = f"{title} ({etype})"

        # Rebuild index
        content = _build_frontmatter(
            title="Vault Index",
            tags=["index", "auto-generated"],
            date=datetime.now().strftime("%Y-%m-%d"),
        )
        content += "# Vault Index\n\n"

        # Group by type
        groups: dict[str, list[tuple[str, str]]] = {}
        for p, desc in sorted(existing_entries.items()):
            # Extract type from description
            type_match = re.search(r"\((\w+)\)$", desc)
            group = type_match.group(1) if type_match else "other"
            groups.setdefault(group, []).append((p, desc))

        for group_name in sorted(groups.keys()):
            content += f"## {group_name.title()}\n\n"
            for p, desc in groups[group_name]:
                content += f"- [[{p}]] — {desc}\n"
            content += "\n"

        _atomic_write(path, content)
        logger.info("Index updated", entries=len(existing_entries))
        return rel_path

    # ── Chronological log ─────────────────────────

    def append_log(self, entry: str) -> str:
        """Append a timestamped entry to the chronological log.

        Returns:
            Relative path to the log file.
        """
        rel_path = "log.md"
        path = self.vault / rel_path

        if not path.exists():
            header = _build_frontmatter(
                title="Chronological Log",
                tags=["log", "auto-generated"],
            )
            header += "# Chronological Log\n\n"
            _atomic_write(path, header)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _atomic_append(path, f"- **{timestamp}** — {entry}\n")
        return rel_path

    # ── Category entries ─────────────────────────

    def append_category_entry(
        self,
        category: str,
        content: str,
        project_id: str | None,
        session_id: str,
        timestamp: str,
    ) -> str:
        """Append a fact to the category-specific file in the vault.

        For project-scoped categories (tech_stack, convention, architecture,
        dependency), the file is named after the project. For global/flexible
        categories, the file is entries.md.

        Args:
            category: NeuralScape category (e.g. 'preference', 'tech_stack').
            content: The extracted fact text.
            project_id: Optional project slug for project-scoped categories.
            session_id: Session identifier.
            timestamp: ISO timestamp string.

        Returns:
            Relative path within the vault.
        """
        from schemas import CATEGORY_VAULT_PATHS, PROJECT_CATEGORIES

        vault_folder = CATEGORY_VAULT_PATHS.get(category, f"Uncategorized/{_slugify(category)}")

        if category in PROJECT_CATEGORIES and project_id:
            filename = f"{_slugify(project_id)}.md"
        else:
            filename = "entries.md"

        rel_path = f"{vault_folder}/{filename}"
        path = self.vault / rel_path
        existing = _read_file(path)

        if not existing:
            # Determine a human-readable title
            folder_name = vault_folder.split("/")[-1]
            title = folder_name.replace("-", " ")
            if category in PROJECT_CATEGORIES and project_id:
                title = f"{title} — {project_id}"

            tags = [category, "auto-generated"]
            if category in PROJECT_CATEGORIES:
                tags.append("project")

            header = _build_frontmatter(
                title=title,
                date=timestamp[:10],
                tags=tags,
            )
            _atomic_write(path, header)

        time_str = timestamp[11:16] if len(timestamp) > 16 else datetime.now().strftime("%H:%M")
        _atomic_append(path, f"- **[{time_str}]** {content} _(session: {session_id})_\n")
        return rel_path

    def update_category_index(self) -> str:
        """Rebuild category-index.md from vault category folders.

        Scans all type folders and builds an index page listing each category,
        its files, and entry counts.

        Returns:
            Relative path to category-index.md.
        """
        from schemas import CATEGORY_VAULT_PATHS

        # Group paths by type (first path component)
        types_order = ["Semantic", "Project", "Episodic", "Procedural", "Working"]
        type_scope = {
            "Semantic": "Global",
            "Project": "Project-Scoped",
            "Episodic": "Flexible",
            "Procedural": "Flexible",
            "Working": "Flexible",
        }

        # Build type → category → files mapping from what exists on disk
        type_categories: dict[str, dict[str, list[tuple[str, int]]]] = {}
        for category, vault_folder in CATEGORY_VAULT_PATHS.items():
            parts = vault_folder.split("/")
            type_name = parts[0]
            category_label = parts[1] if len(parts) > 1 else category

            folder_path = self.vault / vault_folder
            if not folder_path.exists():
                continue

            files_with_counts = []
            for md_file in sorted(folder_path.glob("*.md")):
                if md_file.name.startswith("."):
                    continue
                content = _read_file(md_file)
                entry_count = content.count("\n- **[")
                rel = str(md_file.relative_to(self.vault))
                files_with_counts.append((rel, entry_count))

            if files_with_counts:
                type_categories.setdefault(type_name, {})[category_label] = files_with_counts

        # Build the index content
        lines = [
            _build_frontmatter(
                title="Category Index",
                date=datetime.now().strftime("%Y-%m-%d"),
                tags=["index", "auto-generated", "categories"],
            ),
            "# Category Index\n",
        ]

        for type_name in types_order:
            categories = type_categories.get(type_name)
            if not categories:
                continue
            scope = type_scope.get(type_name, "")
            lines.append(f"## {type_name} ({scope})\n")
            for category_label, files in sorted(categories.items()):
                lines.append(f"### {category_label.replace('-', ' ')}\n")
                for rel_path, count in files:
                    label = "entry" if count == 1 else "entries"
                    lines.append(f"- [[{rel_path}]] — {count} {label}")
                lines.append("")

        rel_path = "category-index.md"
        _atomic_write(self.vault / rel_path, "\n".join(lines) + "\n")
        return rel_path

    # ── Utility methods ──────────────────────────

    def _safe_resolve(self, rel_path: str) -> Path:
        """Resolve a relative path within the vault, preventing path traversal."""
        resolved = (self.vault / rel_path).resolve()
        if not resolved.is_relative_to(self.vault.resolve()):
            raise ValueError(f"Path traversal detected: {rel_path}")
        return resolved

    def list_all_files(self) -> list[str]:
        """List all markdown files in the vault (relative paths)."""
        if not self.vault.exists():
            return []
        return [
            str(f.relative_to(self.vault))
            for f in self.vault.rglob("*.md")
            if not f.name.startswith(".")
        ]

    def read_file(self, rel_path: str) -> str:
        """Read a file from the vault by relative path."""
        return _read_file(self._safe_resolve(rel_path))

    def file_exists(self, rel_path: str) -> bool:
        """Check if a file exists in the vault."""
        return self._safe_resolve(rel_path).exists()

    def find_wikilinks(self, content: str) -> list[str]:
        """Extract all [[wikilink]] targets from content."""
        return re.findall(r"\[\[([^\]]+)\]\]", content)

    def find_mentions(self, content: str, known_pages: list[str]) -> list[str]:
        """Find page titles mentioned in content but not linked."""
        mentions = []
        content_lower = content.lower()
        for page in known_pages:
            # Get the page title from filename
            title = Path(page).stem.replace("-", " ")
            if title.lower() in content_lower and f"[[{page}]]" not in content:
                mentions.append(page)
        return mentions
