"""Junk/tool-log detection and conversation cleaning helpers.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

import re

# Patterns matching raw tool/event log lines that should not be stored as memories
_JUNK_PATTERNS = [
    r"^Ran command:",
    r"^Edited file[:\s]",
    r"^Wrote file[:\s]",
    r"^Read file[:\s]",
    r"^Created file[:\s]",
    r"^Deleted file[:\s]",
    r"^Launched \w+ task:",
    r"^Tool result:",
    r"^Command output:",
]
_JUNK_RE = re.compile("|".join(_JUNK_PATTERNS), re.IGNORECASE | re.MULTILINE)


def _is_junk_fact(content: str) -> bool:
    """Return True if an extracted fact is a raw event log rather than contextual knowledge."""
    stripped = content.strip()
    if len(stripped) < 10:
        return True
    return bool(_JUNK_RE.search(stripped))


def _deleted_msg(noun: str, deleted: int, skipped_shared: int, skipped_standard: int) -> str:
    """Human-readable summary for a filtered delete, naming each preserved tier.

    Standards and shared writes are preserved for different reasons (dictator-only
    vs. team-owned), so report them separately rather than lumping standards under
    "shared".
    """
    msg = f"Deleted {deleted} {noun}"
    preserved = []
    if skipped_shared:
        preserved.append(f"{skipped_shared} shared")
    if skipped_standard:
        preserved.append(f"{skipped_standard} standard")
    if preserved:
        msg += f" (preserved {', '.join(preserved)})"
    return msg


def _clean_conversation_for_graph(messages: list[dict]) -> list[dict]:
    """Filter junk lines from conversation messages before graph ingestion.

    Removes lines matching _JUNK_RE from each message's content.
    Messages that become empty after filtering are dropped entirely.
    """
    cleaned = []
    for msg in messages:
        content = msg.get("content", "")
        if not content:
            cleaned.append(msg)
            continue
        clean_lines = [
            line for line in content.splitlines()
            if not _JUNK_RE.match(line.strip())
        ]
        clean_content = "\n".join(clean_lines).strip()
        if clean_content:
            cleaned.append({**msg, "content": clean_content})
    return cleaned
