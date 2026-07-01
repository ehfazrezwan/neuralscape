"""Format memory context for hook injection into Claude Code sessions.

Produces concise markdown organized by category, with progressive disclosure:
- Layer 1 (injected): Category headers + one-line summaries
- Layer 2 (on-demand): Full details via MCP recall_memories tool
"""

from schemas import MemoryResponse

# Category display order (most actionable first)
CATEGORY_ORDER = [
    "preference",
    "convention",
    "architecture",
    "tech_stack",
    "dependency",
    "workflow",
    "procedure",
    "decision",
    "personal_fact",
    "technical_skill",
    "domain_knowledge",
    "interaction",
    "task_context",
]

CATEGORY_LABELS = {
    "preference": "Preferences",
    "personal_fact": "Personal",
    "technical_skill": "Skills",
    "domain_knowledge": "Domain Knowledge",
    "tech_stack": "Tech Stack",
    "convention": "Conventions",
    "architecture": "Architecture",
    "dependency": "Dependencies",
    "decision": "Decisions",
    "interaction": "Interactions",
    "workflow": "Workflows",
    "procedure": "Procedures",
    "task_context": "Recent Context",
}

# Default max characters (~2000 tokens at ~4 chars/token)
DEFAULT_MAX_CHARS = 8000

# Guaranteed budget for authoritative standards, on top of DEFAULT_MAX_CHARS.
# Standards are binding org rules and must never be truncated away by a large
# recalled-context payload, so they get their own reserved allowance.
STANDARDS_MAX_CHARS = 3000


def format_standards_block(
    standards: list[MemoryResponse],
    max_chars: int = STANDARDS_MAX_CHARS,
) -> str:
    """Format authoritative standards as a binding-directive markdown block.

    Rendered ABOVE the ordinary memory context and framed as binding: on
    conflict these override personal preferences and project conventions.
    Empty string when there are no standards.
    """
    if not standards:
        return ""
    header = (
        "# ⚖️ Neuralscape AUTHORITATIVE Standards (binding)\n\n"
        "These are organization standards set by a Neuralscape dictator. They "
        "are BINDING directives, not preferences. On any conflict they OVERRIDE "
        "personal preferences and project conventions. Follow them unless the "
        "user explicitly overrides them in this session."
    )
    # Standards are BINDING and always injected in full — they are exempt from
    # the ordinary context char budget (see format_context_for_injection). We do
    # NOT truncate: dropping an authoritative directive (or, worse, emitting a
    # header with no rules when the first one is large) would silently weaken the
    # contract. The set size is bounded by governance, not by this formatter.
    lines: list[str] = [header, ""]
    for mem in standards:
        lines.append(f"- {mem.memory}")
    return "\n".join(lines)


def format_context_for_injection(
    categories: dict[str, list[MemoryResponse]],
    max_chars: int = DEFAULT_MAX_CHARS,
    standards: list[MemoryResponse] | None = None,
) -> str:
    """Format memories as concise markdown for hook injection.

    Args:
        categories: Memory responses organized by category name.
        max_chars: Maximum character budget for the category context.
        standards: Authoritative standards, prepended as a binding block that
            is exempt from ``max_chars`` (its own reserved budget).

    Returns:
        Formatted markdown string, or empty string if no memories.
    """
    sections: list[str] = []
    total_chars = 0

    for cat in CATEGORY_ORDER:
        memories = categories.get(cat)
        if not memories:
            continue

        label = CATEGORY_LABELS.get(cat, cat)
        lines: list[str] = [f"## {label}"]

        for mem in memories:
            line = f"- {mem.memory}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

        if len(lines) > 1:
            sections.append("\n".join(lines))

        if total_chars >= max_chars:
            break

    standards_block = format_standards_block(standards or [])

    if not sections:
        # Standards alone are still worth injecting even with no other context.
        return standards_block

    body = "# Neuralscape Memory Context\n\n" + "\n\n".join(sections)
    return f"{standards_block}\n\n---\n\n{body}" if standards_block else body
