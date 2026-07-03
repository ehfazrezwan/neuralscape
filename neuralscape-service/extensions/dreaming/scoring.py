"""Dream scoring — pure functions, no I/O.

Two derived quantities per memory:

- **Promotion score** — the weighted blend from the vendored reference
  (relevance .30, frequency .24, query-diversity .15, recency .15,
  consolidation .10, conceptual richness .06). Orders deep-phase work and
  gates reinforcement-based promotion.
- **Retention strength** (MemoryBank / Ebbinghaus, arXiv 2305.10250) —
  confidence-seeded strength that decays on a forgetting curve from the
  last recall (or creation) and is reinforced by recall count. Memories
  below ``prune_strength_threshold`` become PRUNE *candidates* — the LLM
  pass and the confidence gate still have to concur.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

WEIGHTS = {
    "relevance": 0.30,
    "frequency": 0.24,
    "query_diversity": 0.15,
    "recency": 0.15,
    "consolidation": 0.10,
    "richness": 0.06,
}

# Saturation constants: recall counts and query diversity are unbounded,
# so squash with 1 - exp(-x/k). k chosen so ~5 recalls ≈ 0.63 of max.
_FREQ_K = 5.0
_DIVERSITY_K = 3.0
_CONSOLIDATION_K = 3.0
_RICHNESS_MAX = 5.0  # concepts list is capped at 5 by the schema


def _saturate(x: float, k: float) -> float:
    return 1.0 - math.exp(-max(0.0, x) / k)


def _parse_ts(value) -> float:
    """Best-effort unix ts from an ISO string / epoch float / None."""
    if not value:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


def promotion_score(
    *,
    relevance: float = 0.0,
    recall_count: int = 0,
    unique_query_count: int = 0,
    last_recalled_at: float = 0.0,
    created_at: float = 0.0,
    merged_source_count: int = 0,
    concept_count: int = 0,
    now: float,
    recency_half_life_days: float = 14.0,
) -> float:
    """Weighted promotion score in [0, 1]."""
    anchor = last_recalled_at or created_at
    if anchor > 0:
        age_days = max(0.0, (now - anchor) / 86400.0)
        recency = 0.5 ** (age_days / recency_half_life_days)
    else:
        recency = 0.0
    return (
        WEIGHTS["relevance"] * max(0.0, min(1.0, relevance))
        + WEIGHTS["frequency"] * _saturate(recall_count, _FREQ_K)
        + WEIGHTS["query_diversity"] * _saturate(unique_query_count, _DIVERSITY_K)
        + WEIGHTS["recency"] * recency
        + WEIGHTS["consolidation"] * _saturate(merged_source_count, _CONSOLIDATION_K)
        + WEIGHTS["richness"] * min(1.0, concept_count / _RICHNESS_MAX)
    )


def retention_strength(
    *,
    base_confidence: float | None,
    recall_count: int,
    last_recalled_at: float,
    created_at: float,
    now: float,
    half_life_days: float = 45.0,
) -> float:
    """Ebbinghaus retention strength in [0, 1].

    ``strength = seed * reinforcement * decay`` where:

    - seed = stored confidence (default 0.7 when unset),
    - reinforcement = 1 + saturating bump per recall (max 2×),
    - decay = 0.5 ** (days_since_anchor / half_life); the anchor is the
      last recall when there is one, else creation.

    Computed at dream time from trace aggregates — no per-recall writes.
    """
    seed = 0.7 if base_confidence is None else max(0.0, min(1.0, base_confidence))
    reinforcement = 1.0 + _saturate(recall_count, _FREQ_K)  # ∈ [1, 2)
    anchor = last_recalled_at or created_at
    if anchor <= 0:
        return min(1.0, seed)  # no temporal anchor — never decay to zero silently
    age_days = max(0.0, (now - anchor) / 86400.0)
    decay = 0.5 ** (age_days / half_life_days)
    return max(0.0, min(1.0, seed * reinforcement * decay))


def score_memory(mem: dict, traces: dict, *, now: float | None = None,
                 strength_half_life_days: float = 45.0) -> dict:
    """Score one staged memory dict against its trace aggregates.

    ``mem`` needs: memory_id, created_at (ISO), confidence?, concepts?,
    related_memory_ids?, mean_relevance? (from trace-time scores when
    available). Returns ``{promotion_score, retention_strength}``.
    """
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    created = _parse_ts(mem.get("created_at"))
    t = traces.get(mem.get("memory_id") or "", {})
    return {
        "promotion_score": promotion_score(
            relevance=float(mem.get("mean_relevance") or 0.0),
            recall_count=t.get("recall_count", 0),
            unique_query_count=t.get("unique_query_count", 0),
            last_recalled_at=t.get("last_recalled_at", 0.0),
            created_at=created,
            merged_source_count=len(mem.get("related_memory_ids") or []),
            concept_count=len(mem.get("concepts") or []),
            now=now,
        ),
        "retention_strength": retention_strength(
            base_confidence=mem.get("confidence"),
            recall_count=t.get("recall_count", 0),
            last_recalled_at=t.get("last_recalled_at", 0.0),
            created_at=created,
            now=now,
            half_life_days=strength_half_life_days,
        ),
    }
