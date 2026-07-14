"""E3 — the token-budgeted ``get_context`` assembler.

``assemble_context`` builds a prompt-ready bundle for one session under a
hard token budget:

- **recent messages** — the tail of the session's Redis buffer, ~60% of
  the working budget;
- **rolling summary** — the session's summary slot (the ``long`` slot when
  it fits the summary share, else ``short``, hard-truncated as a last
  resort), ~40% of the working budget;
- **identity card** — the dreaming identity card (user pool, plus the
  project pool when a ``project_id`` is given), served whole under a small
  reserved share;
- **relevant index rows** — compact C1 index rows for ``query`` hits,
  under their own small reserved share.

Budget discipline (Honcho's 60/40 pattern, adapted): the card + index
sections are carved out first (each capped at a fixed fraction of the
total), then the REMAINDER is split 60/40 between recent messages and the
summary. When the summary uses less than its share, the leftover rolls
into the message window (more raw recency), so the bundle converges toward
the budget without ever exceeding it. All counting goes through
``session_summarizer.text_tokens`` (tiktoken when available, len/4
heuristic otherwise) — the same arithmetic the summarizer enforces its
slot caps with.

Provider formatters (``format``):

- ``plain``      — one markdown string (``bundle`` is ``{"text": ...}``);
- ``anthropic``  — ``{"system": <grounding>, "messages": [{role, content}]}``
  ready for the Messages API (roles normalized to user/assistant);
- ``openai``     — ``{"messages": [{"role": "system", ...}, ...]}`` ready
  for Chat Completions.

Every response is ledgered through the savings meter (op
``context_assemble``): baseline = the FULL session transcript plus the
full content of every query hit (what a memoryless client would have had
to inject), served = the bundle actually returned. Signed, never clamped.

Read path per NS convention: synchronous, no writes (the summarizer that
*produces* the slots is the async half).
"""

from __future__ import annotations

import logging

import session_summarizer as ss
from index_format import index_row

logger = logging.getLogger(__name__)

FORMATS = ("plain", "anthropic", "openai")

# Fraction of the total budget reserved for the identity card(s) and the
# relevant-index section, respectively. Small on purpose: cards are ≤40
# grammar lines and index rows ~50-100 tokens each — the bulk of the budget
# belongs to the 60/40 messages/summary split over the remainder.
CARD_BUDGET_SHARE = 0.15
INDEX_BUDGET_SHARE = 0.15

# The messages share of the post-card/index remainder (the summary gets
# the complement).
MESSAGES_SHARE = 0.6

MIN_BUDGET_TOKENS = 100
MAX_INDEX_ROWS = 10

# Fixed allowance for the bundle's own framing (section headers like
# "## Session summary", the "## Recent messages" divider). Carved out of
# the budget BEFORE the 60/40 split so the rendered plain-text bundle stays
# within ``budget_tokens`` even after headers — the budget bounds the whole
# served payload, not just the raw section contents.
FRAMING_OVERHEAD_TOKENS = 48


def _clip_lines_to_tokens(lines: list[str], budget: int) -> tuple[list[str], int]:
    """Greedy prefix of ``lines`` whose summed token cost fits ``budget``."""
    kept: list[str] = []
    used = 0
    for line in lines:
        cost = ss.text_tokens(line) + 1  # +1 ≈ the joining newline
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    return kept, used


def _load_cards(user_id: str, project_id: str | None, redis) -> list[str]:
    """Identity-card grammar lines: the caller's own user card, plus the
    project card when a project is in play. Best-effort — a missing card
    or unreachable Redis yields an empty section, never an error."""
    from extensions.dreaming.card import load_card, resolve_card_pool

    lines: list[str] = []
    pools = [resolve_card_pool(user_id=user_id)]
    if project_id:
        pools.append(resolve_card_pool(project_id=project_id))
    for pool in pools:
        if not pool:
            continue
        try:
            data = load_card(redis, pool)
        except Exception:  # noqa: BLE001
            logger.warning("card load failed for %s (non-fatal)", pool, exc_info=True)
            data = None
        for line in (data or {}).get("lines") or []:
            if isinstance(line, str) and line not in lines:
                lines.append(line)
    return lines


def _index_lines(service, query: str, user_id: str, project_id: str | None):
    """(rendered index-row lines, the full hits behind them). The hits ride
    back so the savings baseline can count their full content."""
    try:
        hits = service.search(
            query=query, user_id=user_id, project_id=project_id, limit=MAX_INDEX_ROWS
        )
    except Exception:  # noqa: BLE001 — assembly degrades, never 500s on search
        logger.warning("assemble query search failed (non-fatal)", exc_info=True)
        return [], []
    lines = []
    for hit in hits or []:
        row = index_row(hit)
        lines.append(
            f"- [{row['id']}] {row.get('glyph', '·')} {row.get('title', '')} "
            f"({row.get('category', '?')}, {row.get('age', '?')}, ~{row.get('tokens', 0)} tok)"
        )
    return lines, list(hits or [])


def _pick_summary(user_id: str, session_id: str, budget: int, redis) -> tuple[str, int, str | None]:
    """(summary text, tokens, slot used). Prefers the ``long`` slot when it
    fits the summary share; falls back to ``short``; hard-truncates as a
    last resort so the share is never exceeded."""
    for slot in ("long", "short"):
        record = ss.load_slot(user_id, session_id, slot, redis)
        if not record:
            continue
        text = str(record.get("text") or "")
        tokens = int(record.get("tokens") or ss.text_tokens(text))
        if tokens <= budget:
            return text, tokens, slot
        if slot == "short":  # even short overflows the share — truncate
            clipped = ss.truncate_to_tokens(text, budget)
            return clipped, ss.text_tokens(clipped), slot
    return "", 0, None


def _recent_messages(user_id: str, session_id: str, budget: int, redis):
    """Newest-backwards fill of the message window, returned oldest-first."""
    all_msgs = ss.get_recent_messages(user_id, session_id, redis=redis)
    kept: list[dict] = []
    used = 0
    for msg in reversed(all_msgs):
        cost = ss.text_tokens(f"{msg.get('role', 'user')}: {msg.get('content', '')}") + 1
        if used + cost > budget:
            break
        kept.append(msg)
        used += cost
    kept.reverse()
    return kept, used, all_msgs


def _grounding_text(
    card_lines: list[str], summary: str, summary_slot: str | None, index_lines: list[str]
) -> str:
    parts: list[str] = []
    if card_lines:
        parts.append("## Identity card\n" + "\n".join(card_lines))
    if summary:
        label = f" ({summary_slot} slot)" if summary_slot else ""
        parts.append(f"## Session summary{label}\n" + summary)
    if index_lines:
        parts.append(
            "## Relevant memories (index — fetch full payloads via "
            "POST /v1/memories/batch-get)\n" + "\n".join(index_lines)
        )
    return "\n\n".join(parts)


def _norm_role(role: str) -> str:
    return "assistant" if str(role).lower() == "assistant" else "user"


def _format_bundle(fmt: str, grounding: str, messages: list[dict]):
    msg_list = [
        {"role": _norm_role(m.get("role", "user")), "content": str(m.get("content", ""))}
        for m in messages
    ]
    if fmt == "anthropic":
        return {"system": grounding, "messages": msg_list}
    if fmt == "openai":
        out = []
        if grounding:
            out.append({"role": "system", "content": grounding})
        return {"messages": out + msg_list}
    # plain: one prompt-ready markdown string
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in msg_list)
    text = grounding
    if transcript:
        text = (text + "\n\n" if text else "") + "## Recent messages\n" + transcript
    return {"text": text}


def _served_tokens(fmt: str, bundle: dict) -> int:
    """Token cost of the bundle AS SERVED (headers/framing included).

    - plain: the whole rendered text;
    - anthropic/openai: the grounding/system text plus each message's
      ``role: content`` cost (+1 joining separator each) — the same
      per-message arithmetic the fill loop budgets with.
    """
    if fmt == "plain":
        return ss.text_tokens(bundle.get("text") or "")
    if fmt == "anthropic":
        system = bundle.get("system") or ""
        msgs = bundle.get("messages") or []
    else:  # openai — system rides as messages[0]
        msgs = bundle.get("messages") or []
        system = ""
    total = ss.text_tokens(system)
    for m in msgs:
        total += ss.text_tokens(f"{m.get('role', '')}: {m.get('content', '')}") + 1
    return total


def assemble_context(
    service,
    *,
    user_id: str,
    budget_tokens: int,
    session_id: str | None = None,
    project_id: str | None = None,
    query: str | None = None,
    fmt: str = "plain",
    redis=None,
) -> dict:
    """Build the token-budgeted context bundle (sync — call in a thread).

    Returns the ``AssembleContextResponse`` field set as a plain dict,
    including a measured savings event already ledgered for ``user_id``.
    Raises ``ValueError`` on an unknown ``format``.
    """
    if fmt not in FORMATS:
        raise ValueError(f"Invalid format {fmt!r}. Must be one of: {list(FORMATS)}")
    budget = max(MIN_BUDGET_TOKENS, int(budget_tokens))
    r = redis if redis is not None else ss._get_redis()

    # ── Reserved sections: identity card, then relevant index ──
    card_all = _load_cards(user_id, project_id, r)
    card_lines, card_used = _clip_lines_to_tokens(
        card_all, int(budget * CARD_BUDGET_SHARE)
    )

    index_lines_all: list[str] = []
    query_hits: list = []
    if query and str(query).strip():
        index_lines_all, query_hits = _index_lines(service, str(query), user_id, project_id)
    index_lines, index_used = _clip_lines_to_tokens(
        index_lines_all, int(budget * INDEX_BUDGET_SHARE)
    )

    # ── 60/40 split of the remainder: messages / rolling summary ──
    remaining = max(0, budget - card_used - index_used - FRAMING_OVERHEAD_TOKENS)
    messages_budget = int(remaining * MESSAGES_SHARE)
    summary_budget = remaining - messages_budget

    summary_text, summary_used, summary_slot = ("", 0, None)
    if session_id:
        summary_text, summary_used, summary_slot = _pick_summary(
            user_id, session_id, summary_budget, r
        )
    # Unspent summary share rolls into the message window (more recency).
    messages_budget += max(0, summary_budget - summary_used)

    messages: list[dict] = []
    messages_used = 0
    all_msgs: list[dict] = []
    if session_id:
        messages, messages_used, all_msgs = _recent_messages(
            user_id, session_id, messages_budget, r
        )

    grounding = _grounding_text(card_lines, summary_text, summary_slot, index_lines)
    bundle = _format_bundle(fmt, grounding, messages)
    # ``used_tokens`` measures the bundle AS SERVED — framing/headers
    # included, not just the section contents — so it is the honest
    # ``served`` side of the meter and stays consistent with the rendered
    # payload. Fits the budget by construction: the section fills came out
    # of ``budget - FRAMING_OVERHEAD_TOKENS`` and the real framing is
    # smaller than that allowance.
    used_tokens = _served_tokens(fmt, bundle)

    # ── E2 ledger: baseline vs served, measured (see module docstring) ──
    import savings_meter as sm

    savings = None
    savings_detail = None
    if sm._meter_enabled():
        baseline = sum(
            ss.text_tokens(f"{m.get('role', 'user')}: {m.get('content', '')}") + 1
            for m in all_msgs
        )
        baseline += sum(sm.hit_tokens(h) for h in query_hits)
        # M4: roll this assemble up to the session it served.
        event = sm.measure_assemble(baseline, used_tokens, corr_id=session_id)
        if event is not None:
            sm.record_event(user_id, event)
            savings = sm.format_savings_line(event)
            savings_detail = event.detail()

    return {
        "status": "ok",
        "user_id": user_id,
        "session_id": session_id,
        "format": fmt,
        "budget_tokens": budget,
        "used_tokens": used_tokens,
        "sections": {
            "card_tokens": card_used,
            "card_lines": len(card_lines),
            "index_tokens": index_used,
            "index_rows": len(index_lines),
            "summary_tokens": summary_used,
            "summary_slot": summary_slot,
            "messages_tokens": messages_used,
            "messages_included": len(messages),
            "messages_buffered": len(all_msgs),
        },
        "bundle": bundle,
        "savings": savings,
        "savings_detail": savings_detail,
    }
