"""Seed marker and victim selection for the cascade real-store test (pure, no I/O).

The real extractor paraphrases: it keeps a distinctive invented proper noun but
drops digits/hex suffixes appended to it. So the marker is a fixed, letters-only
invented word, matched case-insensitively as a whole word. Per-run uniqueness
comes from the test's unique user id, not from the marker.
"""

from __future__ import annotations

import re

VICTIM_MARKER = "Quillfeather"
_VICTIM_RE = re.compile(rf"\b{VICTIM_MARKER}\b", re.IGNORECASE)


def conversation() -> list[dict]:
    """A small 3-fact conversation; only the dog's name is the victim marker."""
    return [
        {"role": "user", "content": (
            f"Quick facts about me for my profile. My dog is named {VICTIM_MARKER}. "
            "I live in Reykjavik since last March. My favourite tea is smoked "
            "lapsang souchong."
        )},
        {"role": "assistant", "content": "Noted, thanks for sharing."},
    ]


def mentions_victim(text: str) -> bool:
    return bool(_VICTIM_RE.search(text or ""))


def select_victims(rows: list[tuple[str, str]]) -> list[str]:
    """Ids of the ``(id, text)`` rows whose text carries the victim marker."""
    return [rid for rid, text in rows if mentions_victim(text)]
