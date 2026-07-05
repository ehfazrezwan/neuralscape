"""LLM prompt for the strategy_synthesizer.

One *incremental merge* call per strategy: given the existing playbook (possibly
empty) + a batch of that strategy's rule memories, return a merged, canonical
playbook body. Every synthesized claim must stay traceable to a source memory
(citation alignment) so the playbook can't drift from the ingested rules.
"""

PLAYBOOK_MERGE_PROMPT = """\
You are a trading-strategy editor. You maintain ONE canonical playbook per strategy, kept up to date as new rule memories arrive from ingesting books/notes.

You will receive:
1. The CURRENT playbook body (may be empty for the first synthesis).
2. A list of MEMORIES for this strategy — distilled, executable rules (each may carry a rule AST, an executable expression, and a source citation). Some you may have already covered; some are new.

Produce an UPDATED playbook body that:
- Reads as a coherent, executable strategy guide — not a list of memories.
- Uses these level-2 (`##`) sections IN THIS ORDER, omitting a section only if truly empty:
  `## Thesis`, `## Setups`, `## Entry`, `## Stop`, `## Targets`, `## Exits`, `## Risk`, `## Market Conditions`, `## Gotchas`, `## Version Updates`.
- PRESERVES the 3-part gate fidelity: state explicitly, for every setup, the support/resistance zone (and, for continuation setups, the market regime) it REQUIRES. A catalyst off a zone is not a trade.
- Keeps executable detail: preserve exact price offsets, anchors, order types, and any executable expressions verbatim from the memories. Do NOT invent numbers.
- CITATION ALIGNMENT: when a rule states a page/quote, keep the page reference inline (e.g. "(Ch8 p.142)") so every claim traces to its source.
- Integrates every distinct rule. If a memory restates something already present, don't duplicate it. If a new memory CONTRADICTS the current playbook, prefer the newer rule and note the change under `## Version Updates`.
- `## Version Updates` (at the END): bullet the changes made in THIS synthesis (new rules added, rules superseded). Keep to the 8 most recent; trim older bullets.

DO NOT:
- Include YAML frontmatter or the page title — the caller adds those.
- Reference internal memory IDs in the body.
- Add disclaimers, hedges, or meta-commentary.

STRATEGY: {strategy_name}

CURRENT PLAYBOOK BODY:
---
{existing_body}
---

MEMORIES:
{memories_block}

Output only the updated playbook body, no commentary.
"""


def render_memories_block(memories: list[dict]) -> str:
    """Format a strategy's rule memories for the prompt's MEMORIES section."""
    lines: list[str] = []
    idx = 0
    for mem in memories:
        content = (mem.get("content") or "").strip()
        if not content:
            continue
        idx += 1
        meta_parts: list[str] = []
        for key in ("category", "confidence", "created_at"):
            v = mem.get(key)
            if v not in (None, ""):
                meta_parts.append(f"{key}={v}")
        meta = f" [{', '.join(meta_parts)}]" if meta_parts else ""
        lines.append(f"{idx}. {content}{meta}")
    return "\n".join(lines) if lines else "(no memories)"
