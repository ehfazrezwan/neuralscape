"""LLM prompts for the wiki_synthesizer.

The synthesizer's single LLM call is an *incremental merge*: given an
existing wiki page (possibly empty) plus a batch of new or updated
memories, return an updated wiki page that integrates the new facts
without losing existing context. Style stays consistent across runs by
being explicit about structure in the prompt.
"""

INCREMENTAL_MERGE_PROMPT = """\
You are a wiki editor. Your job is to keep one topical wiki page up to date as new memories arrive.

You will receive:
1. The CURRENT wiki page body (may be empty for the first synthesis).
2. A list of MEMORIES that belong to this topic — some you may have already covered, some are new.

Produce an UPDATED wiki body that:
- Reads as a coherent, well-organized wiki page on this topic. Not a list of memories.
- Integrates every distinct fact from the memories. If a memory restates something already in the page, do not duplicate it.
- Preserves the structure and prior context where possible. Reorganize only when clarity demands it.
- Uses level-2 headings (`##`) for sections. Keep paragraphs short and direct.
- Includes a short `## Recent updates` section at the END listing any new facts you just merged in, in bullet form. Keep this section to the 5 most recent additions; trim older bullets when adding new ones.

DO NOT:
- Include YAML frontmatter — the caller adds that.
- Include the wiki page title — the caller adds that.
- Reference memory IDs in the body.
- Add disclaimers, hedges, or meta-commentary.

TOPIC: {topic_title}
CATEGORY: {category}

CURRENT PAGE BODY:
---
{existing_body}
---

MEMORIES:
{memories_block}

Output only the updated wiki body, no commentary.
"""


def render_memories_block(memories: list[dict]) -> str:
    """Format a batch of memories into the prompt's MEMORIES section.

    Each entry is shown with its content plus the most-useful v2 metadata
    so the LLM can weight by confidence and observation_type.
    """
    lines: list[str] = []
    idx = 0
    for mem in memories:
        content = mem.get("content", "").strip()
        if not content:
            continue
        idx += 1
        meta_parts: list[str] = []
        for key in ("observation_type", "domain", "confidence", "created_at"):
            v = mem.get(key)
            if v not in (None, ""):
                meta_parts.append(f"{key}={v}")
        meta = f" [{', '.join(meta_parts)}]" if meta_parts else ""
        lines.append(f"{idx}. {content}{meta}")
    return "\n".join(lines) if lines else "(no memories)"
