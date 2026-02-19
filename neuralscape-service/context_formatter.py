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


def format_context_for_injection(
    categories: dict[str, list[MemoryResponse]],
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Format memories as concise markdown for hook injection.

    Args:
        categories: Memory responses organized by category name.
        max_chars: Maximum character budget for the output.

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

    if not sections:
        return ""

    return f"# Neuralscape Memory Context\n\n" + "\n\n".join(sections)
