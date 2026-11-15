"""Service-facing hybrid capabilities. Disabled flags preserve original behavior."""
from __future__ import annotations

import logging
import threading
import hashlib
from functools import lru_cache

from config import settings
from graphiti_core.llm_client.hybrid_decision_client import HybridDecisions

logger = logging.getLogger(__name__)
_guard_lock = threading.Lock()


def decisions(*, telemetry=None) -> HybridDecisions:
    return HybridDecisions(settings.hybrid_inference_api_key, endpoint=settings.hybrid_inference_endpoint,
                        model=settings.hybrid_inference_model,
                        threshold=settings.hybrid_threshold, timeout=settings.hybrid_timeout_s,
                        telemetry=telemetry)


@lru_cache(maxsize=8)
def _extraction_guard(key_identity: str, model: str):
    from inference_guard import InferenceGuard
    return InferenceGuard()


def warn_if_inactive() -> list[str]:
    """Warn once at startup for every enabled flag that cannot activate.

    A flag with an empty endpoint or key falls back to the original path on
    every call; this makes that visible instead of silent. Flags off: no-op.
    """
    enabled = [f for f in ('graph', 'categories', 'extraction', 'rerank')
               if getattr(settings, f'hybrid_{f}_enabled')]
    missing = [name for name, value in (('HYBRID_INFERENCE_ENDPOINT', settings.hybrid_inference_endpoint),
                                        ('HYBRID_INFERENCE_API_KEY', settings.hybrid_inference_api_key))
               if enabled and not value]
    if missing:
        logger.warning('hybrid_inference_inactive flags=%s missing=%s; falling back to the original path',
                       ','.join(enabled), ','.join(missing))
    return missing


def warmup_extraction() -> None:
    """Local-only warmup before readiness: never spends tokens or sends data."""
    warn_if_inactive()
    if not settings.hybrid_extraction_enabled:
        return
    from source_extraction import _segmenter
    from graphiti_core.llm_client.hybrid_decision_client import _http_client
    try:
        _segmenter()
        _http_client()
    except Exception as exc:
        logger.warning('hybrid_warmup_failed %s', type(exc).__name__)


def extract_source_facts(messages: list[dict], *, telemetry=None,
                         category_evidence: dict | None = None) -> list[tuple[str, str]] | None:
    if not settings.hybrid_extraction_enabled:
        return None
    from source_extraction import select_facts

    decision = decisions(telemetry=telemetry)
    try:
        if not (settings.hybrid_inference_api_key and settings.hybrid_inference_endpoint):
            decision.emit({'provider': 'hybrid', 'task': 'source_extraction', 'fallback_reason': 'missing_key_or_endpoint'})
            return None
        def operation():
            evidence, events = {}, []
            original = decision.telemetry
            def record(event):
                events.append(event)
                if original:
                    original(event)
            decision.telemetry = record
            facts = select_facts(messages, decision, category_evidence=evidence)
            failed = any(e.get('attempted') and e.get('fallback_reason') for e in events)
            return (facts, evidence), failed

        key_identity = hashlib.sha256(settings.hybrid_inference_api_key.encode()).hexdigest()
        # lru_cache alone permits duplicate construction on concurrent misses.
        with _guard_lock:
            guard = _extraction_guard(key_identity, settings.hybrid_inference_model)
        result = guard.run(operation, timeout=settings.hybrid_extraction_deadline_s,
                           emit=lambda e: decision.emit({'provider': 'hybrid', 'task': 'source_extraction', **e}))
        if result is None:
            return None
        facts, evidence = result
        if facts is not None and category_evidence is not None:
            category_evidence.update(evidence)
        return facts
    except Exception as exc:
        # Tokenizer/runtime/optional provider failures preserve original flow.
        decision.emit({'provider': 'hybrid', 'task': 'source_extraction', 'fallback_reason': type(exc).__name__})
        return None


def classify_types(values: list[str], *, categories: dict[str, str] | None = None) -> dict[str, str] | None:
    if not settings.hybrid_categories_enabled:
        return None
    from schemas import CORE_MEMORY_CATEGORIES

    # Parser acceptance includes every registered adapter; routing does not.
    # Ordinary memory labels use the existing 13 buckets, not unrelated domains.
    return decisions().classify(values, categories if categories is not None else CORE_MEMORY_CATEGORIES)


def rerank(query: str, memories: list):
    """Only reorder already-authorized rows. No filtering or score overwrites.

    Original score remains the retrieval score, not mislabeled as decision-model confidence.
    Stable ties and any provider failure preserve the prior ranking.
    """
    if not settings.hybrid_rerank_enabled or len(memories) < 2:
        return memories
    count = min(len(memories), settings.hybrid_rerank_limit)
    # Egress gate: rows with a sensitivity stamp, a non-default visibility for
    # their category, or text matching a sensitivity class are never sent. They
    # keep their original position; only eligible rows are reordered among
    # the slots eligible rows already occupy.
    slots = [i for i in range(count) if _may_send(memories[i])]
    if len(slots) < 2:
        return memories
    candidates = [{k: getattr(memories[i], k, None) for k in (
        'memory', 'category', 'source_type', 'valid_at', 'invalid_at', 'occurred_at',
        'epistemic_level', 'visibility', 'category_evidence')} for i in slots]
    for candidate in candidates:
        evidence = candidate.pop('category_evidence')
        if evidence:
            # Hash/model bookkeeping belongs in storage, not every ranking call.
            candidate['category_evidence'] = {'probabilities': evidence['probabilities'], 'mode': evidence['mode']}
    policy = settings.hybrid_rerank_policy + (
        ' Category evidence contains competing relevance weights, not confirmed memberships or fact confidence. '
        'Use it only as supporting context; direct source evidence takes precedence.')
    scores = decisions().scores(query, candidates, policy)
    if scores is None:
        return memories
    ordered = [slots[j] for j in sorted(range(len(slots)), key=lambda j: -scores[j])]
    result = list(memories)
    for slot, source in zip(slots, ordered):
        result[slot] = memories[source]
    return result


def _may_send(row) -> bool:
    """Egress gate for one row: default sensitivity + default visibility + clean text."""
    from memory.sensitivity import classify_sensitivity
    from schemas import default_visibility_for_category

    if getattr(row, 'sensitivity', None):
        return False
    visibility = getattr(row, 'visibility', None)
    if visibility is not None and visibility != default_visibility_for_category(getattr(row, 'category', None) or '').value:
        return False
    return classify_sensitivity(getattr(row, 'memory', '') or '') is None
