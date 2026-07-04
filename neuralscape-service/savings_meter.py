"""E2 — token-economics telemetry: the honest meter.

Measured, not modeled. Per recall op (search / index_only / get_memories /
timeline / ask) the meter computes both sides of the counterfactual:

- ``baseline_tokens`` — real token cost of the FULL content of every hit
  (what a memoryless client would have had to read),
- ``served_tokens`` — token cost of the memory content actually returned in
  full (index-only ops serve zero content; full-payload ops serve exactly
  the baseline; ask serves the synthesized answer),
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

Hot-path economics: baselines come from the ``token_estimate`` stamped on
each memory at write time (a real tiktoken count once this module is live;
the len/4 heuristic on legacy rows), so a recall tokenizes only the small
rendered index payload — never the full contents again. Kill-switch:
``SAVINGS_METER_ENABLED=false`` short-circuits every entry point before any
tokenizer call and write-time stamping falls back to the heuristic.

Ledger: append-only per-user Redis stream ``ns:savings:{user_id}`` with
entries ``{ts, op, baseline, served, overhead, net, rederiv_est}``
(maxlen ~100k, approximate trim) plus O(1) cumulative totals in Redis
hashes (per-user and instance-wide) surfaced by ``GET /v1/metrics``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from index_format import estimate_tokens
from savings_constants import (
    MCP_TOOL_SCHEMA_OVERHEAD_TOKENS,
    SAVINGS_LINE_OVERHEAD_TOKENS,
)

logger = logging.getLogger(__name__)

LEDGER_KEY = "ns:savings:{user_id}"
TOTALS_KEY = "ns:savings:totals:{user_id}"
INSTANCE_TOTALS_KEY = "ns:savings:totals:__instance__"
_SCHEMA_MARK_KEY = "ns:savings:schema-charged:{user_id}:{day}"

_TOTAL_FIELDS = ("events", "baseline", "served", "overhead", "net", "rederiv_est")

# ── Tokenizer (lazy, cached, failure-tolerant) ──────────────────────

_encoder = None
_encoder_failed = False
_encoder_lock = threading.Lock()
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


def count_tokens(text: str | None) -> int:
    """Real token count of ``text``. Only call when the meter is enabled."""
    if not text:
        return 0
    enc = _get_encoder()
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
    net_tokens_saved: int  # SIGNED — never clamped to zero
    rederivation_savings_estimate: int  # heuristic; never in net

    def detail(self) -> dict:
        d = asdict(self)
        d.pop("op", None)
        return d


def _make_event(op: str, baseline: int, served: int, overhead: int) -> SavingsEvent:
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
    )


def measure_recall(
    op: str,
    hits,
    *,
    served_full: bool = False,
    served_text: str | None = None,
    index_payload: str | None = None,
    include_line_overhead: bool = False,
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
    return _make_event(op, baseline, served, overhead)


def measure_ask(baseline_tokens: int, answer_text: str | None) -> SavingsEvent | None:
    """Measure one ask op: baseline = full content of the retrieved evidence
    (precomputed from stored counts), served = the synthesized answer."""
    if not _meter_enabled():
        return None
    return _make_event(
        "ask", max(0, int(baseline_tokens)), count_tokens(answer_text or ""), 0
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


def _append(r, user_id: str, event: SavingsEvent) -> None:
    from config import settings

    fields = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "op": event.op,
        "baseline": event.baseline_tokens,
        "served": event.served_tokens,
        "overhead": event.overhead_tokens,
        "net": event.net_tokens_saved,
        "rederiv_est": event.rederivation_savings_estimate,
    }
    pipe = r.pipeline()
    pipe.xadd(
        LEDGER_KEY.format(user_id=user_id),
        fields,
        maxlen=settings.savings_ledger_maxlen,
        approximate=True,
    )
    for key in (TOTALS_KEY.format(user_id=user_id), INSTANCE_TOTALS_KEY):
        pipe.hincrby(key, "events", 1)
        pipe.hincrby(key, "baseline", event.baseline_tokens)
        pipe.hincrby(key, "served", event.served_tokens)
        pipe.hincrby(key, "overhead", event.overhead_tokens)
        pipe.hincrby(key, "net", event.net_tokens_saved)
        pipe.hincrby(key, "rederiv_est", event.rederivation_savings_estimate)
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
                ),
            )
        _append(r, user_id, event)
        return True
    except Exception:
        logger.debug("savings ledger append failed (non-fatal)", exc_info=True)
        return False


def _read_totals(r, key: str) -> dict:
    raw = r.hgetall(key) or {}
    out = {}
    for field in _TOTAL_FIELDS:
        value = raw.get(field) if field in raw else raw.get(field.encode())
        try:
            out[field] = int(value) if value is not None else 0
        except (TypeError, ValueError):
            out[field] = 0
    return {
        "events": out["events"],
        "baseline_tokens": out["baseline"],
        "served_tokens": out["served"],
        "overhead_tokens": out["overhead"],
        "net_tokens_saved": out["net"],
        "rederivation_savings_estimate": out["rederiv_est"],
    }


def metrics_snapshot(user_id: str, redis=None) -> dict:
    """Cumulative savings totals for GET /v1/metrics (per-user + instance)."""
    from config import settings

    body: dict = {
        "enabled": bool(settings.savings_meter_enabled),
        "tokenizer": settings.savings_tokenizer,
        "tool_schema_overhead_tokens": MCP_TOOL_SCHEMA_OVERHEAD_TOKENS,
        "note": (
            "net_tokens_saved is measured and signed (may be negative); "
            "rederivation_savings_estimate is a heuristic estimate and is "
            "never included in net."
        ),
    }
    if not settings.savings_meter_enabled:
        body["user"] = None
        body["instance"] = None
        return body
    try:
        r = redis if redis is not None else _get_redis()
        body["user"] = {
            "user_id": user_id,
            **_read_totals(r, TOTALS_KEY.format(user_id=user_id)),
        }
        body["instance"] = _read_totals(r, INSTANCE_TOTALS_KEY)
    except Exception:
        logger.warning("savings totals read failed", exc_info=True)
        body["user"] = None
        body["instance"] = None
        body["error"] = "ledger unavailable"
    return body


def _reset_for_tests() -> None:
    """Drop cached encoder/redis singletons (test isolation only)."""
    global _encoder, _encoder_failed, _redis
    _encoder = None
    _encoder_failed = False
    _redis = None
