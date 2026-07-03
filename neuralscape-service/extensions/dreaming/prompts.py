"""LLM prompts for the dreaming extension.

Two calls per pool per sweep:

1. **Consolidation decision pass** (deep phase) — memories in, a strict
   JSON action list out. The prompt encodes the action vocabulary from
   docs/DREAMING_MODE_SPEC.md §3.3 and the quality bar from the vendored
   memory-dream protocol (atomic, third person, temporal anchors).
2. **Reflection pass** (REM) — post-consolidation memories in, a strict
   JSON list of *new higher-order insights* out (pattern lens + failure
   lens), each cited back to its source memory ids.
"""

from __future__ import annotations

import json

CONSOLIDATION_PROMPT = """\
You are the memory consolidation process ("dreaming") of an agentic memory system.
You are given a batch of stored memories from ONE memory pool. Decide, per problem
you find, ONE action. Emit STRICT JSON only — no prose, no markdown fences.

Current date: {today}

Actions (use exactly these types):
- "merge": two or more memories state the same fact. Pick the most complete as
  survivor; fold every unique detail from the others into "content" (max ~60
  words, self-contained). List ALL involved ids in "memory_ids"; put the survivor
  in "survivor_id".
- "invalidate": a memory is contradicted by a NEWER memory in the batch. The
  older one goes in "memory_ids" (the superseding one in "superseded_by_id").
- "prune": credentials/secrets/API keys ("contains_secret": true), pure noise
  (raw tool output, bare timestamps, "ok"/acknowledgments), or stale content the
  low retention_strength supports pruning.
- "rewrite": keep the memory but improve it: first person → third person, add a
  temporal anchor ("As of YYYY-MM-DD") for time-sensitive facts, tighten vague
  wording, fix an obviously wrong category. New text in "content".
- "temporal_reframe": a future-dated/event-anchored memory whose date has now
  passed. Rewrite in past perspective (e.g. "planning X in July" → "did X in
  July {year}"). New text in "content".

Rules:
- Only reference memory ids that appear in the input.
- Never merge/invalidate across different visibility values.
- Do NOT touch memories marked source_type="dream" for merge (they may be
  invalidated if contradicted).
- Confidence: your 0.0-1.0 certainty the action is correct. Be conservative —
  destructive actions below the operator's threshold are only reported.
- No action for a memory is a valid outcome. Prefer fewer, higher-confidence
  actions over many speculative ones.

Output schema:
{{"actions": [{{"type": "merge|invalidate|prune|rewrite|temporal_reframe",
  "memory_ids": ["..."], "survivor_id": "...", "superseded_by_id": "...",
  "content": "...", "contains_secret": false, "confidence": 0.0,
  "reason": "one short sentence"}}]}}

MEMORIES (id | created | category | vis | strength | content):
{memories_block}
"""


REFLECTION_PROMPT = """\
You are the REM phase of an agentic memory system: you reflect on a pool of
memories and produce NEW higher-order insights that are not literally stated in
any single memory. Emit STRICT JSON only — no prose, no markdown fences.

Current date: {today}

Two lenses:
- "pattern": recurring behaviors, implied preferences, cross-memory conclusions
  ("across N memories, the user consistently ...").
- "failure": error → retry → correction sequences; emit the transferable lesson
  ("X fails when Y; do Z instead").

Rules:
- Each insight must be supported by >= 2 source memories; cite their ids.
- Each insight is ONE self-contained sentence (max ~50 words), third person,
  concrete enough to act on. No platitudes.
- category: one of {categories}.
- At most {max_insights} insights; fewer is fine; an empty list is fine.
- Do not restate any single memory — an insight must ADD something.

Output schema:
{{"insights": [{{"content": "...", "lens": "pattern|failure",
  "category": "...", "source_memory_ids": ["..."], "confidence": 0.0}}]}}

MEMORIES (id | created | category | content):
{memories_block}
"""


def render_memories_block(memories: list[dict], *, include_strength: bool = True) -> str:
    """Render staged memories for either prompt (compact single-line rows)."""
    lines: list[str] = []
    for mem in memories:
        content = (mem.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        parts = [
            mem.get("memory_id", "?"),
            str(mem.get("created_at") or "?")[:10],
            mem.get("category") or "?",
        ]
        if include_strength:
            parts.append(mem.get("visibility") or "?")
            strength = mem.get("retention_strength")
            parts.append(f"{strength:.2f}" if isinstance(strength, float) else "?")
        lines.append(" | ".join(parts) + " | " + content)
    return "\n".join(lines) if lines else "(no memories)"


def parse_json_response(raw: str, *, key: str) -> list[dict]:
    """Parse an LLM JSON response defensively; returns [] on garbage.

    Tolerates markdown fences and leading/trailing prose around the JSON
    object — models drift, the sweep must not crash.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    items = obj.get(key)
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []
