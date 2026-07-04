"""Index-first retrieval economics (roadmap C1): titles, token estimates, index rows.

Implements claude-mem's "the map, not the path" principle: recall can return a
compact *index* — what exists and what it costs to read — instead of full
payloads, and the agent chooses what to batch-get. Every helper here is a
cheap, deterministic heuristic; nothing in this module ever calls an LLM, so
title distillation is safe on the hot write path.

The three-layer contract this powers:

1. ``recall_memories(index_only=true)`` → compact rows (~50-100 tokens/hit)
2. agent filters the index by title/category/age/cost
3. ``get_memories(ids=[...])`` → full payloads for the chosen few
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

# ~10-word titles; hard char cap guards against pathological "words".
TITLE_MAX_WORDS = 10
TITLE_MAX_CHARS = 80

# Fallback when content has no titleable text at all (whitespace, pure
# punctuation/symbol noise). Deterministic — callers can test for it.
UNTITLED = "(untitled)"

# Leading markdown / list / quote noise stripped from the first line.
_MD_NOISE = re.compile(r"^[\s#>*+\-`~|:\d.)\]\[]+")
# First-sentence boundary: ., !, ? followed by whitespace. Deliberately naive
# (abbreviations like "e.g." split early) — a slightly short title is fine,
# the id→full-payload path always exists.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"\s+")

# observation_type → single-glyph marker for index rows. Unknown/absent types
# render as the neutral dot. Kept in sync with OBSERVATION_TYPE_VOCAB
# (schemas.py) but tolerant of drift — adapters may register new types.
OBSERVATION_GLYPHS: dict[str, str] = {
    "bugfix": "🐛",
    "feature": "✨",
    "refactor": "♻",
    "decision": "⚖",
    "discovery": "🔍",
    "gotcha": "⚠",
    "pattern": "◇",
    "trade_off": "⇄",
    "research_note": "📝",
    "meeting_outcome": "🤝",
    "task_plan": "🗺",
    "fact": "•",
    "reflection": "💭",
}
DEFAULT_GLYPH = "·"


def glyph_for(observation_type: str | None) -> str:
    """Single-character marker for an observation_type (index-row shorthand)."""
    if not observation_type:
        return DEFAULT_GLYPH
    return OBSERVATION_GLYPHS.get(observation_type, DEFAULT_GLYPH)


def estimate_tokens(content: str | None) -> int:
    """Cheap token-cost estimate: ceil(len/4), floor 1.

    Intentionally rough — the index only needs to communicate relative read
    cost ("this hit is ~40 tokens, that one ~900"), not billing-grade counts.
    """
    if not content:
        return 1
    return max(1, math.ceil(len(content) / 4))


def _clean_first_line(content: str) -> str:
    """First non-empty line, stripped of markdown noise and wrapping quotes."""
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _MD_NOISE.sub("", line).strip()
        line = line.strip("\"'“”‘’ ").strip()
        if line:
            return line
    return ""


def _has_signal(text: str) -> bool:
    """A title is 'garbage' unless it carries at least 3 alphanumeric chars."""
    return sum(1 for c in text if c.isalnum()) >= 3


def _clip_words(text: str, max_words: int, max_chars: int) -> str:
    words = text.split()
    clipped = " ".join(words[:max_words])
    truncated = len(words) > max_words
    if len(clipped) > max_chars:
        clipped = clipped[: max_chars - 1].rstrip()
        truncated = True
    return clipped + " …" if truncated else clipped


def distill_title(
    content: str | None,
    max_words: int = TITLE_MAX_WORDS,
    max_chars: int = TITLE_MAX_CHARS,
) -> str:
    """Distill a ~10-word title from memory content — heuristic only, no LLM.

    Strategy: first sentence of the first non-empty line, markdown noise and
    wrapping quotes stripped, clipped to ``max_words`` words / ``max_chars``
    chars (with an ellipsis marker when clipped). If the first sentence is
    garbage (fewer than 3 alphanumeric chars — e.g. a divider line or emoji
    run), fall back to clipping the whole flattened content; if that is still
    garbage, return :data:`UNTITLED`.
    """
    if not content or not content.strip():
        return UNTITLED

    first_line = _clean_first_line(content)
    sentence = _SENTENCE_SPLIT.split(first_line, maxsplit=1)[0] if first_line else ""
    sentence = sentence.rstrip(".!?").strip()

    if _has_signal(sentence):
        return _clip_words(sentence, max_words, max_chars)

    # Fallback: flatten the whole content and clip.
    flattened = _WS.sub(" ", content).strip()
    flattened = _MD_NOISE.sub("", flattened).strip("\"'“”‘’ ").rstrip(".").strip()
    if _has_signal(flattened):
        return _clip_words(flattened, max_words, max_chars)
    return UNTITLED


def humanize_age(created_at: str | None, now: datetime | None = None) -> str:
    """Compact humanized age of an ISO-8601 timestamp: 'now', '5m', '3h', '2d', '3w', '4mo', '2y'.

    Unparseable / missing timestamps render as '?' rather than raising —
    an index row must never fail to render.
    """
    if not created_at:
        return "?"
    try:
        dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except ValueError:
        return "?"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    seconds = (now - dt).total_seconds()
    if seconds < 0:
        seconds = 0
    if seconds < 60:
        return "now"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)}m"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)}h"
    days = hours / 24
    if days < 7:
        return f"{int(days)}d"
    if days < 30:
        return f"{int(days // 7)}w"
    if days < 365:
        return f"{int(days // 30)}mo"
    return f"{int(days // 365)}y"


def index_row(mem, *, anchor: bool = False, now: datetime | None = None) -> dict:
    """Render a MemoryResponse as a compact index row (~50-100 tokens).

    Shape: ``{id, title, category, glyph, age, tokens, score?, anchor?}``.
    Legacy memories without a stored title/token_estimate get both computed
    on the fly from content — no migration needed. ``None`` values are
    dropped so rows stay minimal.
    """
    row = {
        "id": mem.id,
        "title": getattr(mem, "title", None) or distill_title(mem.memory),
        "category": mem.category,
        "glyph": glyph_for(getattr(mem, "observation_type", None)),
        "age": humanize_age(getattr(mem, "created_at", None), now=now),
        "tokens": getattr(mem, "token_estimate", None) or estimate_tokens(mem.memory),
    }
    if getattr(mem, "score", None) is not None:
        row["score"] = round(mem.score, 4)
    if anchor:
        row["anchor"] = True
    return {k: v for k, v in row.items() if v is not None}
