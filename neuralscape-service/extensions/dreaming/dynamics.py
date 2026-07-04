"""Salience dynamics — pure functions, no I/O (roadmap §A4).

The MemPalace-inspired triplet, computed on top of the recall traces:

- **strength** (Hebbian): +δ per recall, an extra +δ when the memory was
  co-recalled with at least one sibling by the same query. Increments
  saturate against a hard cap, and repeat-query recalls are damped by
  ``REPEAT_QUERY_FACTOR`` (the query-diversity term) — together these are
  guardrail 3: no rich-get-richer runaway, and single-query hammering
  cannot compound. There is no retrieval-induced inhibition anywhere in
  this module: a memory's numbers change only when *it* is recalled or
  from *its own* disuse, never because a sibling was recalled.
- **stability** (spacing effect): grows only on reinforcements spaced
  ≥ ``SPACING_SECONDS`` apart; each unit of stability stretches the decay
  half-life — decay resistance earned by spaced repetition, exactly the
  reinforcement pattern massed repetition can't fake.
- **decay** (Ebbinghaus): exponential in time since last activation at a
  rate modulated by stability, floored at ``DECAY_FLOOR`` (0.05) — a
  memory dims from disuse but NEVER to zero ("dim, don't delete").

Recall-safety contract (§A4, enforced across the extension):

1. Salience never gates retrieval. The only recall-path influence is the
   bounded logarithmic tie-breaker :func:`recall_boost`, OFF by default
   (``DREAMING_SALIENCE_RECALL_K=0.0``); relevance always dominates.
2. Low salience only *nominates* for PRUNE (scoring.py feeds it into
   ``retention_strength``); the consolidation LLM, the confidence gate,
   and the reversible tombstone still stand between nomination and action.
3. See strength/stability notes above — every increment saturates.

Pure math lives here; Redis persistence lives in traces.py; consumption
happens at dream time in scoring.py / consolidate.stage_pool and (opt-in)
at recall time in memory_service._salience_tiebreak.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

# Defaults are mirrored by the DREAMING_DYNAMICS_* config fields
# (extensions/dreaming/config.py); the pure functions take them as
# keyword arguments so tests and callers can override without env vars.
INITIAL_STRENGTH = 1.0
STRENGTH_DELTA = 0.05
STRENGTH_CAP = 5.0
DECAY_FLOOR = 0.05
SPACING_SECONDS = 3600.0        # reinforcements closer than 1h don't grow stability
STABILITY_GROWTH = 1.0
STABILITY_CAP = 10.0
REPEAT_QUERY_FACTOR = 0.2       # damping for recalls by an already-seen query

# Salience normalization: a fresh dynamics state (strength=1.0) maps to
# 0.7 — the same neutral seed the legacy single-half-life path uses for
# unset confidence — so a memory's first recorded recall can never
# *reclassify* it toward pruning just by switching scoring paths.
#   1 - exp(-1.0 / k) = 0.7  ⇒  k = 1 / ln(10/3)
_STRENGTH_NORM_K = 1.0 / math.log(10.0 / 3.0)


@dataclass(frozen=True, slots=True)
class DynamicsState:
    """Per-memory salience state (persisted as JSON in Redis by traces.py)."""

    strength: float = INITIAL_STRENGTH
    stability: float = 1.0
    last_activated_at: float = 0.0   # unix ts of the last recall; 0 = never
    recall_count: int = 0
    co_recall_count: int = 0


def reinforce(
    state: DynamicsState,
    *,
    now: float,
    co_recalled: bool = False,
    novel_query: bool = True,
    delta: float = STRENGTH_DELTA,
    cap: float = STRENGTH_CAP,
    spacing_seconds: float = SPACING_SECONDS,
    repeat_query_factor: float = REPEAT_QUERY_FACTOR,
    stability_growth: float = STABILITY_GROWTH,
    stability_cap: float = STABILITY_CAP,
) -> DynamicsState:
    """Fold one recall into the state. Returns a new state (frozen dataclass).

    - ``co_recalled``: this recall returned the memory together with at
      least one sibling for the same query (Hebbian co-activation) — the
      increment doubles (+δ recall, +δ co-recall).
    - ``novel_query``: False when the query hash was already in the
      memory's per-query HLL; the whole increment is then damped by
      ``repeat_query_factor`` so hammering one query cannot compound
      (guardrail 3), co-recall bonus included.
    - Stability grows by ``stability_growth`` only when the previous
      activation was ≥ ``spacing_seconds`` ago (spacing effect); the first
      recorded recall never grows it (nothing to space against).
    """
    inc = delta * (2.0 if co_recalled else 1.0)
    if not novel_query:
        inc *= repeat_query_factor
    strength = min(cap, state.strength + inc)

    stability = state.stability
    if state.last_activated_at > 0 and (now - state.last_activated_at) >= spacing_seconds:
        stability = min(stability_cap, state.stability + stability_growth)

    return DynamicsState(
        strength=strength,
        stability=stability,
        last_activated_at=now,
        recall_count=state.recall_count + 1,
        co_recall_count=state.co_recall_count + (1 if co_recalled else 0),
    )


def salience(
    state: DynamicsState,
    *,
    now: float,
    base_half_life_days: float = 45.0,
    floor: float = DECAY_FLOOR,
) -> float:
    """Current salience in ``[floor, 1]``: normalized strength × decay.

    ``decay = 0.5 ** (days_since_activation / (base_half_life * stability))``
    — stability stretches the half-life (decay resistance), the floor
    guarantees a faded memory keeps a pulse forever ("dim, don't delete").
    A state with no recorded activation decays nothing (parallel to the
    legacy path's "no temporal anchor — never decay to zero silently").
    """
    norm = 1.0 - math.exp(-max(0.0, state.strength) / _STRENGTH_NORM_K)
    if state.last_activated_at <= 0:
        return max(floor, min(1.0, norm))
    age_days = max(0.0, (now - state.last_activated_at) / 86400.0)
    half_life_days = max(1e-9, base_half_life_days) * max(1.0, state.stability)
    return max(floor, min(1.0, norm * 0.5 ** (age_days / half_life_days)))


def strength_signal(state: DynamicsState) -> float:
    """Non-negative reinforcement signal for the recall tie-breaker.

    Strength *above the baseline* — a never-reinforced state signals 0.0,
    so the k>0 boost is a no-op for it and legacy rows.
    """
    return max(0.0, state.strength - INITIAL_STRENGTH)


def recall_boost(score: float | None, signal: float, k: float) -> float | None:
    """Guardrail-1 bounded logarithmic tie-breaker.

    ``score * (1 + k * log1p(signal))``; ``k <= 0`` (the config default)
    or ``signal <= 0`` returns the score *unchanged* — byte-identical to
    the no-dynamics world. With the conservative recommended k (0.05) a
    maxed-out memory (signal = cap - 1 = 4) gains ~8%: enough to win a
    near-tie, never enough for a hot-but-mediocre match to beat a
    faded-but-relevant one.
    """
    if score is None:
        return None
    if k <= 0.0 or signal <= 0.0:
        return score
    return score * (1.0 + k * math.log1p(signal))


# ── (De)serialization helpers (pure; Redis I/O stays in traces.py) ──


def to_dict(state: DynamicsState) -> dict:
    """Compact JSON-ready dict (short keys — this is a hot Redis hash)."""
    return {
        "s": round(state.strength, 6),
        "st": round(state.stability, 6),
        "la": state.last_activated_at,
        "n": state.recall_count,
        "co": state.co_recall_count,
    }


def from_dict(raw) -> DynamicsState:
    """Parse a persisted state (bytes/str JSON or dict). Broken → fresh state."""
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode()
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            return DynamicsState()
        return DynamicsState(
            strength=float(raw.get("s", INITIAL_STRENGTH)),
            stability=float(raw.get("st", 1.0)),
            last_activated_at=float(raw.get("la", 0.0)),
            recall_count=int(raw.get("n", 0)),
            co_recall_count=int(raw.get("co", 0)),
        )
    except (ValueError, TypeError, json.JSONDecodeError):
        return DynamicsState()
