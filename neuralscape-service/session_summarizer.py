"""E3 — session summarizer slots: recursive short/long rolling summaries.

Every conversation write that carries a session identifier (``run_id`` on
the extraction path, ``session_id`` on the conversation-compiler flush
path — NS treats them as the same concept and this module calls it a
*session id*) appends its messages to a per-session Redis buffer. When the
buffered message count crosses a slot's refresh interval, the worker
refreshes that slot:

- ``short`` — ≤ ``SESSION_SUMMARY_SHORT_MAX_TOKENS`` (default 1,000),
  refreshed every ~``SESSION_SUMMARY_SHORT_EVERY`` (default 20) messages;
- ``long`` — ≤ ``SESSION_SUMMARY_LONG_MAX_TOKENS`` (default 4,000),
  refreshed every ~``SESSION_SUMMARY_LONG_EVERY`` (default 60) messages.

Refresh is **recursive compression** (Honcho's pattern): the new summary is
produced from the PRIOR summary plus only the messages that arrived since
that slot's last refresh — never the whole transcript again. Each slot is a
single value per (user, session, slot) that is REPLACED on refresh, never
accumulated.

Storage decision (documented per the roadmap): slots live in **Redis only**
(TTL ``SESSION_TTL_DAYS``), NOT as memory rows. Rationale mirrors the
identity card (dreaming/card.py): a rolling summary is a *mutable,
session-scoped artifact* that gets replaced dozens of times per session —
as a memory row it would churn the vector store, pollute search results
with stale intermediate states, and fight the dedup/contradiction
machinery. Durable session knowledge already reaches the store through the
normal extraction path; the slot is a serving artifact for the context
assembler, not a fact.

Token budgets are enforced in code, not in faith: the LLM is instructed to
stay under budget AND the stored text is hard-truncated to the slot's
token cap (tiktoken when available, len/4 heuristic otherwise).

All Redis keys are namespaced per (user, session):

- ``ns:session:{user}:{session}:msgs``   — LIST of JSON messages (LTRIM'd)
- ``ns:session:{user}:{session}:meta``   — HASH {count, short_through, long_through}
- ``ns:session:{user}:{session}:summary:{slot}`` — JSON slot record
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone

from index_format import estimate_tokens

logger = logging.getLogger(__name__)

SLOTS = ("short", "long")

_MSGS_KEY = "ns:session:{user_id}:{session_id}:msgs"
_META_KEY = "ns:session:{user_id}:{session_id}:meta"
_SLOT_KEY = "ns:session:{user_id}:{session_id}:summary:{slot}"

# Session ids come from request fields validated at most for length (100);
# collapse anything that could mangle a Redis key. Deterministic so the
# same external id always maps to the same buffer.
_KEY_UNSAFE = re.compile(r"[\s{}\[\]()*?]")

_redis = None
_redis_lock = threading.Lock()


def _get_redis():
    """Lazy shared sync Redis client (short timeouts — never hangs a worker)."""
    global _redis
    if _redis is None:
        with _redis_lock:
            if _redis is None:
                import redis as redis_lib

                from config import settings

                _redis = redis_lib.Redis.from_url(
                    settings.redis_url, socket_timeout=3, socket_connect_timeout=3
                )
    return _redis


def _safe(part: str) -> str:
    return _KEY_UNSAFE.sub("-", str(part or ""))[:100]


def msgs_key(user_id: str, session_id: str) -> str:
    return _MSGS_KEY.format(user_id=_safe(user_id), session_id=_safe(session_id))


def meta_key(user_id: str, session_id: str) -> str:
    return _META_KEY.format(user_id=_safe(user_id), session_id=_safe(session_id))


def slot_key(user_id: str, session_id: str, slot: str) -> str:
    return _SLOT_KEY.format(
        user_id=_safe(user_id), session_id=_safe(session_id), slot=slot
    )


def _ttl_seconds() -> int:
    from config import settings

    return max(1, int(settings.session_ttl_days)) * 86400


def slot_interval(slot: str) -> int:
    from config import settings

    if slot == "short":
        return max(1, int(settings.session_summary_short_every))
    return max(1, int(settings.session_summary_long_every))


def slot_max_tokens(slot: str) -> int:
    from config import settings

    if slot == "short":
        return max(1, int(settings.session_summary_short_max_tokens))
    return max(1, int(settings.session_summary_long_max_tokens))


# ── Token accounting (shared with the assembler) ────────────────────


def text_tokens(text: str | None) -> int:
    """Real token count when the tiktoken encoder is available, len/4
    heuristic otherwise. Reuses the savings meter's cached encoder so the
    summarizer/assembler and the honest meter can never disagree on
    counting. Unlike the meter's hot-path guard this does NOT gate on
    ``savings_meter_enabled`` — budgets must hold with the meter off."""
    if not text:
        return 0
    from savings_meter import _get_encoder

    enc = _get_encoder()
    if enc is None:
        return estimate_tokens(text)
    return len(enc.encode(text))


def truncate_to_tokens(text: str, budget_tokens: int) -> str:
    """Hard cap ``text`` at ``budget_tokens`` (token-exact with tiktoken,
    ~4 chars/token heuristic otherwise). The in-faith instruction to the
    LLM is backed by this in-code enforcement."""
    if budget_tokens <= 0 or not text:
        return ""
    from savings_meter import _get_encoder

    enc = _get_encoder()
    if enc is None:
        max_chars = budget_tokens * 4
        return text if len(text) <= max_chars else text[:max_chars].rstrip()
    ids = enc.encode(text)
    if len(ids) <= budget_tokens:
        return text
    return enc.decode(ids[:budget_tokens]).rstrip()


# ── Message buffer ──────────────────────────────────────────────────


def record_messages(
    user_id: str, session_id: str, messages: list[dict], redis=None
) -> tuple[int, list[str]]:
    """Append conversation messages to the session buffer.

    Returns ``(total_count, due_slots)`` where ``due_slots`` lists the
    summary slots whose refresh interval the new count has crossed
    (``count - <slot>_through >= interval``). Best-effort: a down Redis
    returns ``(0, [])`` and never fails the write path.
    """
    from config import settings

    if not settings.session_summary_enabled or not user_id or not session_id:
        return 0, []
    cleaned = [
        {
            "role": str(m.get("role", "user"))[:32],
            "content": str(m.get("content", "")),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        for m in messages or []
        if str(m.get("content", "")).strip()
    ]
    if not cleaned:
        return 0, []
    try:
        r = redis if redis is not None else _get_redis()
        ttl = _ttl_seconds()
        mkey = msgs_key(user_id, session_id)
        hkey = meta_key(user_id, session_id)
        pipe = r.pipeline()
        pipe.rpush(mkey, *[json.dumps(m, ensure_ascii=False) for m in cleaned])
        pipe.ltrim(mkey, -int(settings.session_buffer_max_messages), -1)
        pipe.expire(mkey, ttl)
        pipe.hincrby(hkey, "count", len(cleaned))
        pipe.expire(hkey, ttl)
        results = pipe.execute()
        count = int(results[3])
        meta = r.hgetall(hkey) or {}
        due = [
            slot
            for slot in SLOTS
            if count - _meta_int(meta, f"{slot}_through") >= slot_interval(slot)
        ]
        return count, due
    except Exception:
        logger.warning("session buffer append failed (non-fatal)", exc_info=True)
        return 0, []


def _meta_int(meta: dict, field: str) -> int:
    value = meta.get(field)
    if value is None:
        value = meta.get(field.encode() if isinstance(field, str) else field)
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def get_recent_messages(
    user_id: str, session_id: str, limit: int | None = None, redis=None
) -> list[dict]:
    """The buffered session messages, oldest → newest (last ``limit``)."""
    try:
        r = redis if redis is not None else _get_redis()
        start = -int(limit) if limit else 0
        raw = r.lrange(msgs_key(user_id, session_id), start, -1) or []
    except Exception:
        logger.warning("session buffer read failed (non-fatal)", exc_info=True)
        return []
    out = []
    for item in raw:
        try:
            msg = json.loads(item)
            if isinstance(msg, dict) and msg.get("content"):
                out.append(msg)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return out


def load_slot(user_id: str, session_id: str, slot: str, redis=None) -> dict | None:
    """The stored slot record ``{text, tokens, through_count, updated_at}`` or None."""
    try:
        r = redis if redis is not None else _get_redis()
        raw = r.get(slot_key(user_id, session_id, slot))
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) and data.get("text") else None
    except Exception:
        logger.warning("summary slot read failed (non-fatal)", exc_info=True)
        return None


# ── The refresh pass (recursive compression) ────────────────────────

SUMMARY_PROMPT = """\
You maintain the rolling {slot} summary of one conversation session. Update
the PRIOR SUMMARY using only the NEW MESSAGES below — this is recursive
compression: the prior summary already covers everything older, so carry
its still-relevant content forward (condensing where needed) and fold the
new messages in.

Rules:
- Keep what matters for continuing the session: goals, decisions and their
  why, current state, open questions, blockers, named entities.
- Drop pleasantries, tool noise, and superseded intermediate states.
- Third person, plain prose or terse bullet lines. No preamble, no headers,
  no meta-commentary about summarizing.
- HARD LIMIT: at most {max_tokens} tokens. Shorter is fine.

PRIOR SUMMARY:
{prior}

NEW MESSAGES:
{messages}

Output ONLY the updated summary text.
"""


def render_messages_block(messages: list[dict], clip_chars: int = 2000) -> str:
    lines = []
    for msg in messages:
        content = str(msg.get("content", "")).strip()
        if len(content) > clip_chars:
            content = content[:clip_chars] + " …"
        lines.append(f"{msg.get('role', 'user')}: {content}")
    return "\n".join(lines) or "(none)"


async def refresh_slot(
    user_id: str, session_id: str, slot: str, llm_call, redis=None
) -> dict:
    """Refresh one summary slot: prior summary + messages-since → new summary.

    Idempotent per threshold crossing (the worker enqueues with a
    deterministic job id); a refresh that finds nothing new is a no-op.
    Returns a small status dict for the ARQ job result.
    """
    import asyncio

    from config import settings

    if slot not in SLOTS:
        return {"status": "skipped", "reason": f"unknown slot {slot!r}"}
    if not settings.session_summary_enabled:
        return {"status": "skipped", "reason": "SESSION_SUMMARY_ENABLED=false"}

    r = redis if redis is not None else _get_redis()
    meta = await asyncio.to_thread(lambda: r.hgetall(meta_key(user_id, session_id)) or {})
    count = _meta_int(meta, "count")
    through = _meta_int(meta, f"{slot}_through")
    since = count - through
    if since <= 0:
        return {"status": "skipped", "reason": "no new messages"}

    new_messages = await asyncio.to_thread(
        get_recent_messages, user_id, session_id, since, r
    )
    if not new_messages:
        # ``since > 0`` says messages arrived, yet the buffer read came back
        # empty — a transient Redis failure or a lost/expired buffer.
        # Bail WITHOUT advancing ``{slot}_through``: advancing here would
        # permanently skip summarizing messages this pass never saw. The
        # next threshold crossing (or a retry) picks them up.
        logger.warning(
            "summary refresh for %s slot found no buffered messages despite "
            "count-through=%d — not advancing through_count", slot, since,
        )
        return {"status": "failed", "reason": "buffer read empty"}
    prior = await asyncio.to_thread(load_slot, user_id, session_id, slot, r)
    prior_text = (prior or {}).get("text") or "(no prior summary)"
    max_tokens = slot_max_tokens(slot)

    prompt = SUMMARY_PROMPT.format(
        slot=slot,
        max_tokens=max_tokens,
        prior=prior_text,
        messages=render_messages_block(new_messages),
    )
    try:
        raw = await llm_call(prompt)
    except Exception as e:  # noqa: BLE001 — a failed refresh keeps the prior slot
        logger.warning("summary LLM call failed for %s slot (non-fatal): %s", slot, e)
        return {"status": "failed", "reason": "llm unavailable"}

    text = truncate_to_tokens((raw or "").strip(), max_tokens)
    if not text:
        return {"status": "skipped", "reason": "empty summary"}

    record = {
        "text": text,
        "tokens": text_tokens(text),
        "through_count": count,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "slot": slot,
    }

    def _store() -> None:
        ttl = _ttl_seconds()
        pipe = r.pipeline()
        # Single slot per (user, session, slot): SET replaces, never appends.
        pipe.set(slot_key(user_id, session_id, slot), json.dumps(record, ensure_ascii=False), ex=ttl)
        pipe.hset(meta_key(user_id, session_id), f"{slot}_through", count)
        pipe.expire(meta_key(user_id, session_id), ttl)
        pipe.execute()

    await asyncio.to_thread(_store)

    # M1 — compaction lifecycle. Refresh is RECURSIVE compression: prior
    # summary + delta messages → a new summary that REPLACES the prior in
    # served context. So the honest incremental counterfactual is
    # baseline = prior_summary + folded_messages (served = the new summary),
    # which telescopes over a session to Σ(messages) − S_final — the true
    # compaction saving. (Omitting the prior summary would under-state by every
    # intermediate summary and book spurious negatives once the running summary
    # exceeds a message batch.) Best-effort, off the request path (graph
    # worker); a meter failure never fails the refresh.
    try:
        import savings_meter as sm

        if sm._meter_enabled() and user_id:
            baseline_tok = int((prior or {}).get("tokens") or 0) + text_tokens(
                render_messages_block(new_messages)
            )
            event = sm.measure_compaction(
                baseline_tok, record["tokens"], item_id=slot, corr_id=session_id
            )
            if event is not None:
                await asyncio.to_thread(sm.record_event, user_id, event)
    except Exception:
        logger.debug("compaction savings metering failed (non-fatal)", exc_info=True)

    return {
        "status": "refreshed",
        "slot": slot,
        "tokens": record["tokens"],
        "through_count": count,
        "messages_compressed": len(new_messages),
    }


def _reset_for_tests() -> None:
    """Drop the cached Redis singleton (test isolation only)."""
    global _redis
    _redis = None
