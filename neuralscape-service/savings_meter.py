"""E2 — token-economics telemetry: the honest meter.

Measured, not modeled. Per lifecycle op the meter computes both sides of the
counterfactual:

- ``baseline_tokens`` — real token cost of the FULL content a memoryless
  client would have had to read to get the same answer,
- ``served_tokens`` — token cost of what NS actually returned in full,
- ``overhead_tokens`` — NS's own injected material: the rendered index rows
  + the savings line on this response, plus the per-release MCP tool-schema
  constant charged once per user per UTC day as its own ledger entry
  (see savings_constants.py),
- ``net_tokens_saved = (baseline − served) − overhead`` — **signed, never
  clamped**: a recall over ten tiny memories whose index rows cost more
  than their content honestly goes negative.

``rederivation_savings_estimate`` (tokens an agent would burn re-deriving
each fact from sources) is a clearly-labeled heuristic ESTIMATE kept in a
separate field and NEVER blended into the measured headline.

Lifecycle model (M1). Every event carries a ``lifecycle_stage`` — one of
``retrieval | ingest | context_assembly | code_nav | compaction`` — so the
meter can slice savings per stage, and an optional ``item_id`` /
``corr_id`` (task/session correlation, M4). Per-event rows land in the
append-only stream alongside the O(1) cumulative totals, which are now also
bucketed per-lifecycle and per-op.

Bounce accounting (M2). An index-only / locate op that is followed by a
full-fetch of the SAME item within a window saved nothing — the client
re-read the content anyway, so its overhead was pure cost. We surface BOTH
the raw ``net_tokens_saved`` and the ``adjusted_net_tokens_saved`` that
deducts those bounced credits.

Code-nav baseline (M3). For code locate/neighbors/query the baseline is the
avoided FILE-READ footprint (a disclosed per-distinct-file estimate), not
the memory-content size — the read-avoidance number.

Hot-path economics: baselines come from the ``token_estimate`` stamped on
each memory at write time, so a recall tokenizes only the small rendered
index payload — never the full contents again. Kill-switch:
``SAVINGS_METER_ENABLED=false`` short-circuits every entry point before any
tokenizer call and write-time stamping falls back to the heuristic.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from index_format import estimate_tokens
from savings_constants import (
    LIFECYCLE_STAGES,
    MCP_TOOL_SCHEMA_OVERHEAD_TOKENS,
    SAVINGS_LINE_OVERHEAD_TOKENS,
)

logger = logging.getLogger(__name__)

LEDGER_KEY = "ns:savings:{user_id}"
TOTALS_KEY = "ns:savings:totals:{user_id}"
INSTANCE_TOTALS_KEY = "ns:savings:totals:__instance__"
_SCHEMA_MARK_KEY = "ns:savings:schema-charged:{user_id}:{day}"
# Per-lifecycle / per-op / per-task rollup keys (M1, M4). ``scope`` is either
# a user_id or the ``__instance__`` sentinel.
LIFECYCLE_TOTALS_KEY = "ns:savings:lc:{scope}:{lifecycle}"
OP_TOTALS_KEY = "ns:savings:op:{scope}:{op}"
OPS_SET_KEY = "ns:savings:ops:{scope}"
TASK_TOTALS_KEY = "ns:savings:task:{user_id}:{corr_id}"
# M2 — short-lived "this item was served as an index row" marker.
BOUNCE_KEY = "ns:savings:bounce:{user_id}:{item_id}"

_INSTANCE_SCOPE = "__instance__"

# ``net`` is the RAW measured net; ``adjusted`` is the bounce-corrected net.
# Both are cumulative and both are surfaced; for a normal event they are
# equal, for a bounce deduction net==0 and adjusted<0.
_TOTAL_FIELDS = (
    "events",
    "baseline",
    "served",
    "overhead",
    "net",
    "adjusted",
    "rederiv_est",
)

# ── Tokenizer (lazy, cached, failure-tolerant) ──────────────────────

_encoder = None
_encoder_failed = False
_encoder_lock = threading.Lock()
_alt_encoders: dict[str, object] = {}
_redis = None


def _meter_enabled() -> bool:
    from config import settings

    return bool(settings.savings_meter_enabled)


def _get_encoder():
    """Lazy tiktoken encoding. One load attempt; on failure the meter falls
    back to the heuristic (logged once) rather than breaking recalls."""
    global _encoder, _encoder_failed
    if _encoder is not None or _encoder_failed:
        return _encoder
    with _encoder_lock:
        if _encoder is not None or _encoder_failed:
            return _encoder
        from config import settings

        try:
            import tiktoken

            _encoder = tiktoken.get_encoding(settings.savings_tokenizer)
        except Exception:
            _encoder_failed = True
            logger.warning(
                "tiktoken encoding %r unavailable — savings meter degrades to "
                "the len/4 heuristic", settings.savings_tokenizer, exc_info=True,
            )
    return _encoder


def _get_alt_encoder(encoding: str):
    """Lazy tiktoken encoding for an explicitly-requested basis (M5 optional
    per-model tokenizer). Cached; falls back to None (heuristic) on failure."""
    if encoding in _alt_encoders:
        return _alt_encoders[encoding]
    with _encoder_lock:
        if encoding in _alt_encoders:
            return _alt_encoders[encoding]
        try:
            import tiktoken

            enc = tiktoken.get_encoding(encoding)
        except Exception:
            enc = None
            logger.warning(
                "tiktoken encoding %r unavailable — using the len/4 heuristic",
                encoding, exc_info=True,
            )
        _alt_encoders[encoding] = enc
    return enc


def count_tokens(text: str | None, *, encoding: str | None = None) -> int:
    """Real token count of ``text``. Only call when the meter is enabled.

    ``encoding`` selects an alternate tiktoken basis (M5) — e.g. ``cl100k_base``
    for a client billed on that tokenizer; omit for the configured default."""
    if not text:
        return 0
    enc = _get_alt_encoder(encoding) if encoding else _get_encoder()
    if enc is None:  # tokenizer unavailable — degrade, don't break recall
        return estimate_tokens(text)
    return len(enc.encode(text))


def stamp_tokens(content: str | None) -> int:
    """Write-time ``token_estimate``: real count when the meter is enabled,
    the cheap len/4 heuristic when it's off (zero tokenizer calls)."""
    if not _meter_enabled():
        return estimate_tokens(content)
    return max(1, count_tokens(content)) if content else 1


def hit_tokens(mem) -> int:
    """Token cost of one hit's full content — the stored write-time count
    when present (no tokenizer call), else counted now.

    Any non-None stored value counts as present — including 0 — so a
    malformed/zero legacy stamp never silently re-tokenizes content on the
    hot path (the "baselines ride stored counts" guarantee)."""
    stored = getattr(mem, "token_estimate", None)
    if stored is not None:
        try:
            return max(0, int(stored))
        except (TypeError, ValueError):
            pass  # not coercible — fall back to counting
    return count_tokens(getattr(mem, "memory", None) or "")


# ── Measurement ─────────────────────────────────────────────────────


@dataclass(slots=True)
class SavingsEvent:
    op: str
    baseline_tokens: int
    served_tokens: int
    overhead_tokens: int
    net_tokens_saved: int  # SIGNED raw net — never clamped to zero
    rederivation_savings_estimate: int  # heuristic; never in net
    # M1/M2/M4 metadata (defaulted so the historical 6-positional
    # construction stays valid). ``adjusted_net_tokens_saved`` defaults to
    # the raw net; a bounce deduction sets net=0 and adjusted<0.
    adjusted_net_tokens_saved: int | None = None
    lifecycle_stage: str = "retrieval"
    item_id: str | None = None
    corr_id: str | None = None

    def __post_init__(self):
        if self.adjusted_net_tokens_saved is None:
            self.adjusted_net_tokens_saved = self.net_tokens_saved

    def detail(self) -> dict:
        """The compact per-response detail surface. Intentionally limited to
        the core measured fields so the rendered savings line stays within
        ``SAVINGS_LINE_OVERHEAD_TOKENS`` (the lifecycle/correlation metadata
        rides the ledger + metrics payload, not this line)."""
        return {
            "baseline_tokens": self.baseline_tokens,
            "served_tokens": self.served_tokens,
            "overhead_tokens": self.overhead_tokens,
            "net_tokens_saved": self.net_tokens_saved,
            "rederivation_savings_estimate": self.rederivation_savings_estimate,
        }


def _make_event(
    op: str,
    baseline: int,
    served: int,
    overhead: int,
    *,
    lifecycle_stage: str = "retrieval",
    item_id: str | None = None,
    corr_id: str | None = None,
) -> SavingsEvent:
    from config import settings

    saved = baseline - served
    return SavingsEvent(
        op=op,
        baseline_tokens=baseline,
        served_tokens=served,
        overhead_tokens=overhead,
        net_tokens_saved=saved - overhead,  # signed; may go negative
        rederivation_savings_estimate=int(
            baseline * max(0.0, settings.savings_rederivation_multiplier)
        ),
        lifecycle_stage=lifecycle_stage,
        item_id=item_id,
        corr_id=corr_id,
    )


def measure_recall(
    op: str,
    hits,
    *,
    served_full: bool = False,
    served_text: str | None = None,
    index_payload: str | None = None,
    include_line_overhead: bool = False,
    lifecycle_stage: str = "retrieval",
    corr_id: str | None = None,
) -> SavingsEvent | None:
    """Measure one recall op. Returns None when the meter is disabled
    (guaranteeing zero tokenizer calls on the hot path).

    - ``served_full`` — the response carries every hit's full content
      (plain search / batch-get): served == baseline, saved == 0.
    - ``served_text`` — the response carries this synthesized text instead
      of raw content (ask's answer).
    - ``index_payload`` — the rendered index rows actually returned;
      counted as NS-injected overhead (the rows are the map NS adds, not
      recalled content).
    - ``include_line_overhead`` — charge the savings line's own cost
      (constant; the meter pays for its own output).
    """
    if not _meter_enabled():
        return None
    baseline = sum(hit_tokens(h) for h in hits)
    if served_full:
        served = baseline
    elif served_text is not None:
        served = count_tokens(served_text)
    else:
        served = 0
    overhead = 0
    if index_payload is not None:
        overhead += count_tokens(index_payload)
    if include_line_overhead:
        overhead += SAVINGS_LINE_OVERHEAD_TOKENS
    return _make_event(
        op, baseline, served, overhead,
        lifecycle_stage=lifecycle_stage, corr_id=corr_id,
    )


def measure_ask(
    baseline_tokens: int,
    answer_text: str | None,
    *,
    corr_id: str | None = None,
) -> SavingsEvent | None:
    """Measure one ask op: baseline = full content of the retrieved evidence
    (precomputed from stored counts), served = the synthesized answer."""
    if not _meter_enabled():
        return None
    return _make_event(
        "ask", max(0, int(baseline_tokens)), count_tokens(answer_text or ""), 0,
        lifecycle_stage="retrieval", corr_id=corr_id,
    )


def measure_assemble(
    baseline_tokens: int,
    served_tokens: int,
    *,
    corr_id: str | None = None,
) -> SavingsEvent | None:
    """Measure one context-assemble op (E3): baseline = the full session
    transcript + full content of every relevant hit (what a memoryless
    client would inject), served = the token-budgeted bundle actually
    returned. Both sides arrive precomputed (the assembler already counts
    every section it serves), so this is arithmetic, not tokenization."""
    if not _meter_enabled():
        return None
    return _make_event(
        "context_assemble",
        max(0, int(baseline_tokens)),
        max(0, int(served_tokens)),
        0,
        lifecycle_stage="context_assembly",
        corr_id=corr_id,
    )


def measure_ingest(
    baseline_tokens: int,
    served_tokens: int,
    *,
    item_id: str | None = None,
    corr_id: str | None = None,
) -> SavingsEvent | None:
    """Measure one ingest op (M1): baseline = the full source document a
    client would otherwise re-read to extract its facts, served = the total
    token cost of the distilled facts + passages actually stored. The
    positive net is the one-time distillation compression — counted, never
    executed (the LLM extraction cost is provider-side, not injected)."""
    if not _meter_enabled():
        return None
    return _make_event(
        "ingest",
        max(0, int(baseline_tokens)),
        max(0, int(served_tokens)),
        0,
        lifecycle_stage="ingest",
        item_id=item_id,
        corr_id=corr_id,
    )


def measure_code_nav(
    op: str,
    *,
    served_text: str | None = None,
    served_tokens: int | None = None,
    files=None,
    file_count: int | None = None,
    avoided_read_tokens: int | None = None,
    item_id: str | None = None,
    corr_id: str | None = None,
) -> SavingsEvent | None:
    """Measure one code-nav op (M3): locate / neighbors / query.

    The baseline is the AVOIDED FILE-READ footprint — the token cost of the
    file(s) the model would otherwise have opened to find the answer — NOT
    the memory-content size. We never read files on the hot path, so we book
    a DISCLOSED per-distinct-file estimate (``avoided_read_tokens``, default
    from settings) times the number of distinct files the answer touched.
    Clearly an estimate; labeled as such in the metrics payload.

    Pass ``files`` (a list — distinct paths are counted) or an explicit
    ``file_count``; ``served_text``/``served_tokens`` is the compact answer
    actually returned."""
    if not _meter_enabled():
        return None
    from config import settings

    per_file = (
        avoided_read_tokens
        if avoided_read_tokens is not None
        else settings.savings_code_nav_avoided_read_tokens_per_file
    )
    if file_count is None:
        file_count = len({f for f in (files or []) if f})
    baseline = max(0, int(file_count)) * max(0, int(per_file))
    if served_tokens is not None:
        served = max(0, int(served_tokens))
    else:
        served = count_tokens(served_text or "")
    return _make_event(
        op, baseline, served, 0,
        lifecycle_stage="code_nav", item_id=item_id, corr_id=corr_id,
    )


def measure_compaction(
    baseline_tokens: int,
    served_tokens: int,
    *,
    item_id: str | None = None,
    corr_id: str | None = None,
) -> SavingsEvent | None:
    """Measure one compaction op (M1): baseline = the transcript messages
    folded into the summary, served = the resulting rolling summary. The net
    is the recursive-compression saving of the session summarizer."""
    if not _meter_enabled():
        return None
    return _make_event(
        "compaction",
        max(0, int(baseline_tokens)),
        max(0, int(served_tokens)),
        0,
        lifecycle_stage="compaction",
        item_id=item_id,
        corr_id=corr_id,
    )


def format_savings_line(event: SavingsEvent) -> str:
    """The compact honest headline: 'saved ~N tokens (X%), net of overhead'.

    N is the SIGNED net — a negative recall reads 'saved ~-42 tokens'.
    """
    if event.baseline_tokens > 0:
        pct = round(100 * event.net_tokens_saved / event.baseline_tokens)
    else:
        pct = 0
    return f"saved ~{event.net_tokens_saved} tokens ({pct}%), net of overhead"


# ── Ledger (append-only Redis stream + O(1) totals) ─────────────────


def _get_redis():
    global _redis
    if _redis is None:
        import redis as redis_lib

        from config import settings

        _redis = redis_lib.Redis.from_url(
            settings.redis_url, socket_timeout=2, socket_connect_timeout=2
        )
    return _redis


def _increments(event: SavingsEvent) -> dict:
    adjusted = (
        event.adjusted_net_tokens_saved
        if event.adjusted_net_tokens_saved is not None
        else event.net_tokens_saved
    )
    return {
        "events": 1,
        "baseline": event.baseline_tokens,
        "served": event.served_tokens,
        "overhead": event.overhead_tokens,
        "net": event.net_tokens_saved,
        "adjusted": adjusted,
        "rederiv_est": event.rederivation_savings_estimate,
    }


def _bump(pipe, key: str, incr: dict) -> None:
    for field_name, amount in incr.items():
        pipe.hincrby(key, field_name, amount)


def _append(r, user_id: str, event: SavingsEvent) -> None:
    from config import settings

    incr = _increments(event)
    lifecycle = event.lifecycle_stage or "retrieval"
    fields = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "op": event.op,
        "lifecycle": lifecycle,
        "item_id": event.item_id or "",
        "corr_id": event.corr_id or "",
        "baseline": event.baseline_tokens,
        "served": event.served_tokens,
        "overhead": event.overhead_tokens,
        "net": event.net_tokens_saved,
        "adjusted": incr["adjusted"],
        "rederiv_est": event.rederivation_savings_estimate,
    }
    pipe = r.pipeline()
    pipe.xadd(
        LEDGER_KEY.format(user_id=user_id),
        fields,
        maxlen=settings.savings_ledger_maxlen,
        approximate=True,
    )
    for scope in (user_id, _INSTANCE_SCOPE):
        _bump(pipe, TOTALS_KEY.format(user_id=scope) if scope != _INSTANCE_SCOPE
              else INSTANCE_TOTALS_KEY, incr)
        _bump(pipe, LIFECYCLE_TOTALS_KEY.format(scope=scope, lifecycle=lifecycle), incr)
        _bump(pipe, OP_TOTALS_KEY.format(scope=scope, op=event.op), incr)
        pipe.sadd(OPS_SET_KEY.format(scope=scope), event.op)
    if event.corr_id:
        task_key = TASK_TOTALS_KEY.format(user_id=user_id, corr_id=event.corr_id)
        _bump(pipe, task_key, incr)
        pipe.expire(task_key, max(60, int(settings.savings_task_rollup_ttl_seconds)))
    pipe.execute()


def record_event(user_id: str, event: SavingsEvent | None, redis=None) -> bool:
    """Append one measured event to the caller's ledger (best-effort).

    Also charges the per-release tool-schema overhead ONCE per user per UTC
    day (its own ``tool_schema`` entry, net = −constant) the first time that
    user meters an op that day — the honest side of the ledger without
    modeling session counts. Never raises; a down Redis skips silently.
    """
    if event is None or not _meter_enabled() or not user_id:
        return False
    try:
        r = redis if redis is not None else _get_redis()
        day = datetime.now(timezone.utc).date().isoformat()
        mark = _SCHEMA_MARK_KEY.format(user_id=user_id, day=day)
        if r.set(mark, "1", nx=True, ex=2 * 86400):
            _append(
                r,
                user_id,
                SavingsEvent(
                    op="tool_schema",
                    baseline_tokens=0,
                    served_tokens=0,
                    overhead_tokens=MCP_TOOL_SCHEMA_OVERHEAD_TOKENS,
                    net_tokens_saved=-MCP_TOOL_SCHEMA_OVERHEAD_TOKENS,
                    rederivation_savings_estimate=0,
                    lifecycle_stage="context_assembly",
                ),
            )
        _append(r, user_id, event)
        return True
    except Exception:
        logger.debug("savings ledger append failed (non-fatal)", exc_info=True)
        return False


# ── Bounce accounting (M2) ──────────────────────────────────────────


def arm_bounce(user_id: str, hits, redis=None) -> bool:
    """Mark each hit served as an index row (served content == 0) so a
    subsequent full-fetch of the same item within the window can be detected
    as a bounce. Best-effort; never raises."""
    if not _meter_enabled() or not user_id or not hits:
        return False
    try:
        from config import settings

        r = redis if redis is not None else _get_redis()
        window = max(1, int(settings.savings_bounce_window_seconds))
        pipe = r.pipeline()
        armed = 0
        for h in hits:
            hid = getattr(h, "id", None)
            if not hid:
                continue
            pipe.set(
                BOUNCE_KEY.format(user_id=user_id, item_id=hid),
                hit_tokens(h),
                ex=window,
            )
            armed += 1
        if armed:
            pipe.execute()
        return armed > 0
    except Exception:
        logger.debug("savings bounce arm failed (non-fatal)", exc_info=True)
        return False


def check_and_deduct_bounce(user_id: str, hits, redis=None) -> int:
    """A full-fetch of items previously served as index rows: those index
    ops saved nothing (the client re-read the content), so deduct each
    credited baseline from the bounce-adjusted total via a ``bounce`` event
    (net=0, adjusted<0). Returns how many bounces were deducted.
    Best-effort; never raises."""
    if not _meter_enabled() or not user_id or not hits:
        return 0
    deducted = 0
    try:
        r = redis if redis is not None else _get_redis()
        for h in hits:
            hid = getattr(h, "id", None)
            if not hid:
                continue
            key = BOUNCE_KEY.format(user_id=user_id, item_id=hid)
            val = r.get(key)
            if val is None:
                continue
            r.delete(key)
            try:
                baseline_share = max(0, int(val))
            except (TypeError, ValueError):
                continue
            _append(
                r,
                user_id,
                SavingsEvent(
                    op="bounce",
                    baseline_tokens=0,
                    served_tokens=0,
                    overhead_tokens=0,
                    net_tokens_saved=0,  # raw net untouched
                    rederivation_savings_estimate=0,
                    adjusted_net_tokens_saved=-baseline_share,  # only adjusted moves
                    lifecycle_stage="retrieval",
                    item_id=hid,
                ),
            )
            deducted += 1
    except Exception:
        logger.debug("savings bounce check failed (non-fatal)", exc_info=True)
    return deducted


# ── Metrics snapshot ────────────────────────────────────────────────


def _read_totals(r, key: str) -> dict:
    raw = r.hgetall(key) or {}
    out = {}
    for field_name in _TOTAL_FIELDS:
        value = raw.get(field_name) if field_name in raw else raw.get(field_name.encode())
        try:
            out[field_name] = int(value) if value is not None else 0
        except (TypeError, ValueError):
            out[field_name] = 0
    return {
        "events": out["events"],
        "baseline_tokens": out["baseline"],
        "served_tokens": out["served"],
        "overhead_tokens": out["overhead"],
        "net_tokens_saved": out["net"],
        "adjusted_net_tokens_saved": out["adjusted"],
        "rederivation_savings_estimate": out["rederiv_est"],
    }


def _decode(v) -> str:
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


def _read_breakdown(r, scope: str) -> dict:
    """Per-lifecycle (fixed enum) + per-op (from the ops set) totals for one
    scope. Only non-empty buckets are returned so the payload stays lean."""
    by_lifecycle: dict = {}
    for stage in LIFECYCLE_STAGES:
        totals = _read_totals(r, LIFECYCLE_TOTALS_KEY.format(scope=scope, lifecycle=stage))
        if totals["events"]:
            by_lifecycle[stage] = totals
    by_op: dict = {}
    try:
        members = r.smembers(OPS_SET_KEY.format(scope=scope)) or []
    except Exception:
        members = []
    for op in sorted(_decode(m) for m in members):
        totals = _read_totals(r, OP_TOTALS_KEY.format(scope=scope, op=op))
        if totals["events"]:
            by_op[op] = totals
    return {"by_lifecycle": by_lifecycle, "by_op": by_op}


def metrics_snapshot(user_id: str, redis=None, task_id: str | None = None) -> dict:
    """Cumulative savings totals for GET /v1/metrics (per-user + instance),
    now with per-lifecycle + per-op breakdowns and raw vs bounce-adjusted
    totals (M6), plus the honesty labels (M5)."""
    from config import settings

    body: dict = {
        "enabled": bool(settings.savings_meter_enabled),
        "tokenizer": settings.savings_tokenizer,
        "tokenizer_basis_note": (
            f"token counts measured with the '{settings.savings_tokenizer}' "
            "tiktoken encoding; a provider's billing units may differ."
        ),
        "tool_schema_overhead_tokens": MCP_TOOL_SCHEMA_OVERHEAD_TOKENS,
        "lifecycle_stages": list(LIFECYCLE_STAGES),
        "rederivation_multiplier": float(settings.savings_rederivation_multiplier),
        "rederivation_note": (
            "rederivation_savings_estimate = rederivation_multiplier × baseline "
            "tokens; a clearly-labeled heuristic ESTIMATE, never included in "
            "net_tokens_saved or adjusted_net_tokens_saved."
        ),
        "note": (
            "net_tokens_saved is measured and signed (may be negative); "
            "adjusted_net_tokens_saved additionally deducts bounces "
            "(index/locate ops whose item was re-fetched in full within the "
            "bounce window); rederivation_savings_estimate is a heuristic "
            "estimate and is never included in either net."
        ),
        "code_nav_avoided_read_tokens_per_file": int(
            settings.savings_code_nav_avoided_read_tokens_per_file
        ),
        "code_nav_baseline_note": (
            "code_nav baselines are an ESTIMATE of the avoided file-read "
            "footprint (distinct files × code_nav_avoided_read_tokens_per_file), "
            "not memory-content size."
        ),
    }
    if not settings.savings_meter_enabled:
        body["user"] = None
        body["instance"] = None
        if task_id is not None:
            body["task"] = None
        return body
    try:
        r = redis if redis is not None else _get_redis()
        body["user"] = {
            "user_id": user_id,
            **_read_totals(r, TOTALS_KEY.format(user_id=user_id)),
            **_read_breakdown(r, user_id),
        }
        body["instance"] = {
            **_read_totals(r, INSTANCE_TOTALS_KEY),
            **_read_breakdown(r, _INSTANCE_SCOPE),
        }
        if task_id is not None:
            body["task"] = {
                "corr_id": task_id,
                **_read_totals(
                    r, TASK_TOTALS_KEY.format(user_id=user_id, corr_id=task_id)
                ),
            }
    except Exception:
        logger.warning("savings totals read failed", exc_info=True)
        body["user"] = None
        body["instance"] = None
        if task_id is not None:
            body["task"] = None
        body["error"] = "ledger unavailable"
    return body


def _reset_for_tests() -> None:
    """Drop cached encoder/redis singletons (test isolation only)."""
    global _encoder, _encoder_failed, _redis, _alt_encoders
    _encoder = None
    _encoder_failed = False
    _redis = None
    _alt_encoders = {}
