"""Service-facing hybrid capabilities. Disabled flags preserve original behavior."""
from __future__ import annotations

import logging
import threading
import hashlib
from functools import lru_cache

from config import settings
from graphiti_core.llm_client.jev_client import JevDecisions

logger = logging.getLogger(__name__)
_guard_lock = threading.Lock()


def decisions(*, telemetry=None) -> JevDecisions:
    return JevDecisions(settings.typesafe_api_key, model=settings.jev_model,
                        threshold=settings.jev_threshold, timeout=settings.jev_timeout_s,
                        telemetry=telemetry)


@lru_cache(maxsize=8)
def _extraction_guard(key_identity: str, model: str):
    from inference_guard import InferenceGuard
    return InferenceGuard()


def warmup_extraction() -> None:
    """Local-only warmup before readiness: never spends tokens or sends data."""
    if not settings.jev_extraction_enabled:
        return
    from jev_extraction import _segmenter
    from graphiti_core.llm_client.jev_client import _http_client
    try:
        _segmenter()
        _http_client()
    except Exception as exc:
        logger.warning('hybrid_warmup_failed %s', type(exc).__name__)


def extract_source_facts(messages: list[dict], *, telemetry=None,
                         category_evidence: dict | None = None) -> list[tuple[str, str]] | None:
    if not settings.jev_extraction_enabled:
        return None
    from jev_extraction import select_facts

    decision = decisions(telemetry=telemetry)
    try:
        if not settings.typesafe_api_key:
            decision.emit({'provider': 'jev', 'task': 'source_extraction', 'fallback_reason': 'missing_key'})
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

        key_identity = hashlib.sha256(settings.typesafe_api_key.encode()).hexdigest()
        # lru_cache alone permits duplicate construction on concurrent misses.
        with _guard_lock:
            guard = _extraction_guard(key_identity, settings.jev_model)
        result = guard.run(operation, timeout=settings.jev_extraction_deadline_s,
                           emit=lambda e: decision.emit({'provider': 'jev', 'task': 'source_extraction', **e}))
        if result is None:
            return None
        facts, evidence = result
        if facts is not None and category_evidence is not None:
            category_evidence.update(evidence)
        return facts
    except Exception as exc:
        # Tokenizer/runtime/optional provider failures preserve original flow.
        decision.emit({'provider': 'jev', 'task': 'source_extraction', 'fallback_reason': type(exc).__name__})
        return None


def classify_types(values: list[str], *, categories: dict[str, str] | None = None) -> dict[str, str] | None:
    if not settings.jev_categories_enabled:
        return None
    from schemas import CORE_MEMORY_CATEGORIES

    # Parser acceptance includes every registered adapter; routing does not.
    # Ordinary memory labels use the existing 13 buckets, not unrelated domains.
    return decisions().classify(values, categories if categories is not None else CORE_MEMORY_CATEGORIES)


def rerank(query: str, memories: list):
    """Only reorder already-authorized rows. No filtering or score overwrites.

    Original score remains the retrieval score, not mislabeled as Jev confidence.
    Stable ties and any provider failure preserve the prior ranking.
    """
    if not settings.jev_rerank_enabled or len(memories) < 2:
        return memories
    count = min(len(memories), settings.jev_rerank_limit)
    candidates = [{k: getattr(m, k, None) for k in (
        'memory', 'category', 'source_type', 'valid_at', 'invalid_at', 'occurred_at',
        'epistemic_level', 'visibility', 'category_evidence')} for m in memories[:count]]
    for candidate in candidates:
        evidence = candidate.pop('category_evidence')
        if evidence:
            # Hash/model bookkeeping belongs in storage, not every ranking call.
            candidate['category_evidence'] = {'probabilities': evidence['probabilities'], 'mode': evidence['mode']}
    policy = settings.jev_rerank_policy + (
        ' Category evidence contains competing relevance weights, not confirmed memberships or fact confidence. '
        'Use it only as supporting context; direct source evidence takes precedence.')
    scores = decisions().scores(query, candidates, policy)
    if scores is None:
        return memories
    return [memories[i] for i in sorted(range(count), key=lambda i: -scores[i])] + memories[count:]
