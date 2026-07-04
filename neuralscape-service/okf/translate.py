"""THE Open Knowledge Format translation module — every OKF name lives here.

OKF (`GoogleCloudPlatform/knowledge-catalog`, v0.1 draft) specifies
knowledge bundles as markdown + YAML frontmatter with a tiny set of
structural conventions: a required ``type`` frontmatter field, reserved
``index.md`` / ``log.md`` filenames, and an ``okf_version`` marker in the
bundle-root index. The draft spec may still rename fields (e.g. their
issue #154 proposes ``type`` → ``kind``), so **all** OKF key names,
reserved filenames, version strings, and the category↔type mapping tables
are confined to this module. Nothing outside it may hardcode an OKF name —
renderers and walkers call the helpers below instead (enforced by a
tokenizer-based isolation test in ``tests/test_okf.py``).

Two mapping directions:

- **Export**: :func:`type_for_category` / :func:`type_for_page_kind` turn a
  Neuralscape category or vault page kind into an OKF ``type`` value.
- **Ingest**: :func:`category_for_type` inverts the table exactly for our
  own types and heuristically (keyword aliases) for foreign producers'
  types; unknown types return ``None`` so the caller can fall back to an
  LLM pass or a default category.

The rich Neuralscape envelope (confidence, epistemic_level, derived_from,
valid_at/invalid_at, source_ref, times_derived, …) rides along as
spec-permitted extension keys (§4.1 "Producers MAY include any additional
keys"); :func:`envelope_extensions` / :func:`extensions_to_envelope`
translate both ways.
"""

from __future__ import annotations

import re
from datetime import date, datetime

import yaml

# ── Version + reserved names (spec §3.1, §11) ───────────────────────

OKF_VERSION = "0.1"

_KEY_VERSION = "okf_version"

INDEX_FILENAME = "index.md"
LOG_FILENAME = "log.md"
RESERVED_FILENAMES = (INDEX_FILENAME, LOG_FILENAME)


def is_reserved_filename(name: str) -> bool:
    """True for ``index.md`` / ``log.md`` (reserved at any level, §3.1)."""
    return name.rsplit("/", 1)[-1] in RESERVED_FILENAMES


# ── Frontmatter keys (spec §4.1) ────────────────────────────────────

_KEY_TYPE = "type"
_KEY_TITLE = "title"
_KEY_DESCRIPTION = "description"
_KEY_RESOURCE = "resource"
_KEY_TAGS = "tags"
_KEY_TIMESTAMP = "timestamp"

#: The OKF-defined field set, in the spec's priority order. Everything
#: else in a frontmatter block is a producer extension key.
OKF_FIELD_ORDER = (
    _KEY_TYPE,
    _KEY_TITLE,
    _KEY_DESCRIPTION,
    _KEY_RESOURCE,
    _KEY_TAGS,
    _KEY_TIMESTAMP,
)

# ── type ↔ category mapping tables ──────────────────────────────────

#: Neuralscape memory category → OKF ``type`` value. Exact both-ways
#: mapping for round-tripping our own bundles; descriptive, self-
#: explanatory values per §4.1's guidance ("Playbook" is the spec's own
#: example for procedural knowledge).
CATEGORY_TYPES: dict[str, str] = {
    "preference": "Preference",
    "personal_fact": "Personal Fact",
    "technical_skill": "Skill",
    "domain_knowledge": "Domain Knowledge",
    "tech_stack": "Tech Stack",
    "convention": "Convention",
    "architecture": "Architecture Note",
    "dependency": "Dependency",
    "decision": "Decision",
    "interaction": "Interaction",
    "workflow": "Workflow",
    "procedure": "Playbook",
    "task_context": "Task Context",
}

_TYPE_CATEGORIES: dict[str, str] = {t.casefold(): c for c, t in CATEGORY_TYPES.items()}

#: Vault page kind → OKF ``type``. Page kinds are Neuralscape's own
#: vocabulary; only the resulting type values are OKF-facing.
PAGE_KIND_TYPES: dict[str, str] = {
    "topic": "Topic",
    "hub": "Project Hub",
    "home": "Home",
    "card": "Identity Card",
    "diary": "Dream Diary",
}

#: Keyword heuristics for foreign producers' types (Dataplex / Unity
#: Catalog exports, LLM-wiki repos, …). First hit wins; scanned in order.
_TYPE_ALIAS_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("playbook", "procedure"),
    ("runbook", "procedure"),
    ("how-to", "procedure"),
    ("howto", "procedure"),
    ("guide", "procedure"),
    ("tutorial", "procedure"),
    ("workflow", "workflow"),
    ("pipeline", "workflow"),
    ("process", "workflow"),
    ("decision", "decision"),
    ("adr", "decision"),
    ("preference", "preference"),
    ("convention", "convention"),
    ("policy", "convention"),
    ("standard", "convention"),
    ("style", "convention"),
    ("architecture", "architecture"),
    ("design", "architecture"),
    ("dependency", "dependency"),
    ("library", "dependency"),
    ("vendor", "dependency"),
    ("stack", "tech_stack"),
    ("tool", "tech_stack"),
    ("platform", "tech_stack"),
    ("meeting", "interaction"),
    ("interaction", "interaction"),
    ("conversation", "interaction"),
    ("skill", "technical_skill"),
    ("task", "task_context"),
    # Catalog-shaped types (tables, datasets, metrics, APIs, references)
    # are subject-matter knowledge about data assets.
    ("table", "domain_knowledge"),
    ("dataset", "domain_knowledge"),
    ("schema", "domain_knowledge"),
    ("metric", "domain_knowledge"),
    ("api", "domain_knowledge"),
    ("endpoint", "domain_knowledge"),
    ("reference", "domain_knowledge"),
    ("report", "domain_knowledge"),
    ("view", "domain_knowledge"),
    ("glossary", "domain_knowledge"),
    ("concept", "domain_knowledge"),
    ("note", "domain_knowledge"),
    ("knowledge", "domain_knowledge"),
)


def type_for_category(category: str | None) -> str:
    """OKF ``type`` value for a memory category (adapter categories →
    title-cased words, e.g. ``visual_exemplar`` → ``Visual Exemplar``)."""
    if not category:
        return CATEGORY_TYPES["domain_knowledge"]
    known = CATEGORY_TYPES.get(category)
    if known:
        return known
    return " ".join(w.capitalize() for w in re.split(r"[_\s]+", category.strip()) if w) or (
        CATEGORY_TYPES["domain_knowledge"]
    )


def type_for_page_kind(kind: str) -> str:
    """OKF ``type`` value for a vault page kind (topic/hub/home/card/diary)."""
    return PAGE_KIND_TYPES.get(kind, PAGE_KIND_TYPES["topic"])


def category_for_type(type_value: str | None) -> str | None:
    """Invert ``type`` → category: exact table first, keyword aliases second.

    Returns ``None`` for types no heuristic covers — the caller decides
    (LLM fallback, then a default category). Consumers MUST tolerate
    unknown types (§4.1), so this never raises.
    """
    if not type_value or not str(type_value).strip():
        return None
    key = str(type_value).strip().casefold()
    exact = _TYPE_CATEGORIES.get(key)
    if exact:
        return exact
    # Adapter-registered categories round-trip through their title-cased
    # type (type_for_category("visual_exemplar") == "Visual Exemplar"):
    # reverse the transform and accept any currently-registered category.
    snake = re.sub(r"\s+", "_", key)
    try:
        from schemas import MEMORY_CATEGORIES

        if snake in MEMORY_CATEGORIES:
            return snake
    except Exception:  # pragma: no cover — schemas is always importable in-service
        pass
    # Also match our page-kind types when a Neuralscape vault is re-ingested.
    for kind, t in PAGE_KIND_TYPES.items():
        if key == t.casefold():
            return "domain_knowledge" if kind != "card" else "personal_fact"
    for keyword, category in _TYPE_ALIAS_KEYWORDS:
        if keyword in key:
            return category
    return None


# ── Envelope extension keys (spec §4.1 "Extensions") ────────────────

#: Neuralscape envelope field → frontmatter extension key. Plain names,
#: intentionally aligned with what the OKF community is proposing for
#: confidence/provenance metadata (their issues #148/#151/#140).
ENVELOPE_EXTENSION_KEYS: dict[str, str] = {
    "memory_id": "memory_id",
    "category": "category",
    "scope": "scope",
    "project_id": "project_id",
    "visibility": "visibility",
    "memory_kind": "memory_kind",
    "confidence": "confidence",
    "epistemic_level": "epistemic_level",
    "derived_from": "derived_from",
    "valid_at": "valid_at",
    "invalid_at": "invalid_at",
    "source_ref": "source_ref",
    "times_derived": "times_derived",
    "salience": "salience",
}


def envelope_extensions(memory: dict) -> dict:
    """Extension-key dict for a memory's envelope fields (skips empties)."""
    out: dict = {}
    for field, key in ENVELOPE_EXTENSION_KEYS.items():
        value = memory.get(field)
        if value is None or value == "" or value == []:
            continue
        out[key] = value
    return out


def extensions_to_envelope(frontmatter: dict) -> dict:
    """Recover envelope fields from a concept's extension keys (ingest side)."""
    out: dict = {}
    for field, key in ENVELOPE_EXTENSION_KEYS.items():
        if key in frontmatter and frontmatter[key] not in (None, "", []):
            out[field] = frontmatter[key]
    return out


# ── Frontmatter rendering / parsing ─────────────────────────────────

_FM_RE = re.compile(r"^---\r?\n(?P<fm>.*?)\r?\n---\r?\n?(?P<body>.*)$", re.DOTALL)


def _scalar(value) -> str:
    """One YAML-safe flow-style scalar/collection on a single line."""
    if isinstance(value, str):
        value = re.sub(r"\s+", " ", value).strip()
    dumped = yaml.safe_dump(
        value, default_flow_style=True, allow_unicode=True, width=1_000_000, sort_keys=False
    ).strip()
    if dumped.endswith("\n..."):
        dumped = dumped[: -len("\n...")].strip()
    return dumped


def render_frontmatter(fields: dict) -> str:
    """A ``---``-delimited YAML frontmatter block from an ordered dict.

    Values are rendered one per line in flow style, so the block stays
    both greppable (line-oriented) and YAML-parseable. ``None``/empty
    values are omitted. String values already shaped like a YAML flow
    list (``[a, b]``) are emitted verbatim (they parse as lists).
    """
    lines = ["---"]
    for key, value in fields.items():
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {_scalar(value)}")
    lines.append("---")
    return "\n".join(lines)


def concept_frontmatter(
    *,
    type_value: str | None = None,
    page_kind: str | None = None,
    category: str | None = None,
    title: str = "",
    description: str | None = None,
    resource: str | None = None,
    tags: list[str] | None = None,
    timestamp: str | None = None,
    extensions: dict | None = None,
) -> str:
    """The full frontmatter block for one concept document.

    ``type`` resolves from (in order): an explicit ``type_value``, the
    vault ``page_kind``, or the memory ``category`` — at least one must
    yield a non-empty type, since ``type`` is the one REQUIRED field (§4.1).
    Extension keys render after the OKF field set, order preserved.
    """
    resolved_type = type_value or (
        type_for_page_kind(page_kind) if page_kind else type_for_category(category)
    )
    fields: dict = {
        _KEY_TYPE: resolved_type,
        _KEY_TITLE: title,
        _KEY_DESCRIPTION: description,
        _KEY_RESOURCE: resource,
        _KEY_TAGS: [str(t) for t in tags] if tags else None,
        _KEY_TIMESTAMP: timestamp,
    }
    for key, value in (extensions or {}).items():
        fields.setdefault(key, value)
    return render_frontmatter(fields)


def parse_document(text: str) -> tuple[dict, str]:
    """Split a concept document into (frontmatter dict, body).

    Tolerant by design (§9's permissive consumption model): missing or
    unparseable frontmatter yields ``({}, text)`` rather than raising.
    """
    if not text:
        return {}, ""
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group("fm"))
    except yaml.YAMLError:
        return {}, text
    if not isinstance(fm, dict):
        return {}, text
    return fm, m.group("body").lstrip("\n")


def frontmatter_span(text: str) -> tuple[int, int] | None:
    """(start, end) char offsets of the frontmatter block, or None.

    Lets a chunker skip the metadata block while keeping spans accurate
    against the original document text.
    """
    if not text:
        return None
    m = _FM_RE.match(text)
    if not m:
        return None
    return 0, m.start("body")


# ── Field accessors (consumers never touch key names) ───────────────


def concept_type(frontmatter: dict) -> str | None:
    """The concept's ``type`` value, or None when absent/empty."""
    value = frontmatter.get(_KEY_TYPE)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def concept_title(frontmatter: dict) -> str | None:
    value = frontmatter.get(_KEY_TITLE)
    return str(value).strip() if value else None


def concept_description(frontmatter: dict) -> str | None:
    value = frontmatter.get(_KEY_DESCRIPTION)
    return str(value).strip() if value else None


def concept_tags(frontmatter: dict) -> list[str]:
    value = frontmatter.get(_KEY_TAGS)
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    if isinstance(value, str) and value.strip():
        return [t.strip() for t in value.strip("[]").split(",") if t.strip()]
    return []


def concept_timestamp(frontmatter: dict) -> str | None:
    value = frontmatter.get(_KEY_TIMESTAMP)
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value).strip() or None


def concept_resource(frontmatter: dict) -> str | None:
    value = frontmatter.get(_KEY_RESOURCE)
    return str(value).strip() if value else None


# ── Bundle-root version marker (spec §11) ───────────────────────────


def root_index_frontmatter() -> str:
    """The bundle-root ``index.md`` frontmatter carrying the version marker
    (the only place an index file is permitted frontmatter, §11)."""
    return render_frontmatter({_KEY_VERSION: OKF_VERSION})


def has_version_marker(frontmatter: dict) -> bool:
    return bool(str(frontmatter.get(_KEY_VERSION) or "").strip())


def declared_version(frontmatter: dict) -> str | None:
    value = frontmatter.get(_KEY_VERSION)
    return str(value).strip() if value not in (None, "") else None


# ── Index files (spec §6) ───────────────────────────────────────────


def index_entry(title: str, href: str, description: str | None = None) -> str:
    """One §6 index bullet: ``* [Title](href) - description``."""
    line = f"* [{title}]({href})"
    if description:
        line += " - " + re.sub(r"\s+", " ", description).strip()
    return line


def render_index(
    sections: list[tuple[str, list[str]]],
    *,
    is_bundle_root: bool = False,
) -> str:
    """A §6 index document: ``# Section`` headings over entry bullets.

    Index files contain no frontmatter — except the bundle root, which
    carries the ``okf_version`` marker block (§11).
    """
    parts: list[str] = []
    if is_bundle_root:
        parts.append(root_index_frontmatter())
        parts.append("")
    for heading, entries in sections:
        if not entries:
            continue
        parts.append(f"# {heading}")
        parts.append("")
        parts.extend(entries)
        parts.append("")
    return "\n".join(parts).rstrip("\n") + "\n"


# ── Log files (spec §7) ─────────────────────────────────────────────

LOG_DATE_HEADING_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})\s*$")


def log_entry(kind: str, text: str) -> str:
    """One §7 log bullet: ``* **Kind**: text`` (leading bold word convention)."""
    return "* **" + kind + "**: " + re.sub(r"\s+", " ", text).strip()


def log_date_heading(day: str) -> str:
    """A §7 ``## YYYY-MM-DD`` date heading."""
    return f"## {day}"


def render_log(title: str, dated_entries: list[tuple[str, list[str]]]) -> str:
    """A §7 log document: title heading, then date-grouped entries newest first.

    ``dated_entries`` is ``[(YYYY-MM-DD, [entry-line, ...]), ...]`` already
    sorted newest-first by the caller. Log files carry no frontmatter.
    """
    lines = [f"# {title}", ""]
    for day, entries in dated_entries:
        if not entries:
            continue
        lines.append(log_date_heading(day))
        lines.extend(entries)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def parse_log(text: str) -> list[tuple[str, list[str]]]:
    """Recover ``[(date, [entry-line, ...])]`` from an existing §7 log file."""
    out: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    for line in (text or "").splitlines():
        m = LOG_DATE_HEADING_RE.match(line.strip())
        if m:
            current = []
            out.append((m.group(1), current))
        elif current is not None and line.strip().startswith("* "):
            current.append(line.rstrip())
    return out


# ── Cross-links (spec §5) ───────────────────────────────────────────

_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def extract_concept_links(body: str, concept_id: str) -> list[str]:
    """Concept IDs this body links to (bundle-relative + relative forms).

    External URLs and anchors are skipped; ``.md`` suffixes are stripped
    to concept IDs (§2). Broken links are legal (§5.3) — the caller
    decides whether a target exists in the bundle.
    """
    links: list[str] = []
    base_dir = concept_id.rsplit("/", 1)[0] if "/" in concept_id else ""
    for _, target in _MD_LINK_RE.findall(body or ""):
        target = target.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith(("mailto:", "#")):
            continue
        if target.startswith("/"):
            resolved = target.lstrip("/")
        else:
            resolved = f"{base_dir}/{target}" if base_dir else target
        # normalize ./ and ../ segments
        parts: list[str] = []
        for seg in resolved.split("/"):
            if seg in ("", "."):
                continue
            if seg == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(seg)
        resolved = "/".join(parts)
        if resolved.endswith("/"):
            continue
        if resolved.endswith(".md"):
            resolved = resolved[: -len(".md")]
        elif "." in resolved.rsplit("/", 1)[-1]:
            continue  # links to non-markdown assets aren't concept links
        if resolved and not is_reserved_filename(f"{resolved}.md"):
            if resolved != concept_id and resolved not in links:
                links.append(resolved)
    return links
