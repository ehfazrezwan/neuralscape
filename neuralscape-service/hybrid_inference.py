"""Service-facing hybrid capabilities. Disabled flags preserve original behavior."""
from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
import hashlib
from functools import lru_cache
from pathlib import Path

from config import settings
from graphiti_core.llm_client.jev_client import JevDecisions

logger = logging.getLogger(__name__)
_needle_lock = threading.Lock()
_needle_model = None
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


def try_extract(text: str, *, telemetry=None) -> list[tuple[str, str]] | None:
    """Experimental Needle lane for ONE literal assertion, not conversations.

    Entire source text must survive as a quote. Multi-sentence content, custom
    guidance/ontologies and speaker/date extraction stay with the main model.
    Suppressed calls, ungrounded fields and missing confidence always abstain.
    """
    if not settings.needle_extraction_enabled:
        return None
    text = text.strip()
    if not text or len(text) > 600 or '\n' in text or len(re.findall(r'[.!?](?:\s|$)', text)) > 1:
        return None
    started = time.perf_counter()
    event = {'provider': 'needle', 'model': 'needle3', 'task': 'literal_extraction'}
    try:
        # Native base models share global state. Serialize init/reset/complete,
        # and reset on BOTH sides so tenant text never becomes history.
        with _needle_lock:
            global _needle_model
            if _needle_model is None:
                os.environ.setdefault('NEEDLE_TELEMETRY', '0')
                from needle import Needle
                from needle.agent import fetch
                from schemas import MEMORY_CATEGORIES

                # Downloads belong in deployment warmup, never the write path.
                cache = Path(fetch.cache_dir(3))
                if not (cache / fetch.base_weights(3)).is_file() or not (cache / fetch._lib_name()).is_file():
                    raise ValueError('model_not_prefetched')
                _needle_model = Needle(generation=3, auto_date=False, tools=[{
                    'name': 'extract_memory',
                    'description': 'Extract one durable factual memory. Copy the entire statement verbatim into quote and classify it.',
                    'parameters': {'type': 'object', 'properties': {
                        'quote': {'type': 'string', 'description': 'Exact entire source statement, without paraphrasing.'},
                        'category': {'type': 'string', 'enum': sorted(MEMORY_CATEGORIES)},
                    }, 'required': ['quote', 'category']},
                }])
                event['cold_start'] = True
            _needle_model.reset()
            try:
                response = _needle_model.complete(text, max_new_tokens=256)
            finally:
                _needle_model.reset()
        confidence = response.get('confidence')
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not settings.needle_min_confidence <= confidence <= 1:
            raise ValueError('uncertain')
        if response.get('suppressed_calls') or any((response.get('validation') or {}).values()):
            raise ValueError('ungrounded_or_suppressed')
        calls = response.get('function_calls') or []
        if len(calls) != 1 or calls[0].get('name') != 'extract_memory':
            raise ValueError('invalid_call')
        arguments = calls[0].get('arguments') or {}
        from schemas import MEMORY_CATEGORIES

        if set(arguments) != {'quote', 'category'} or arguments['category'] not in MEMORY_CATEGORIES or arguments['quote'] != text:
            raise ValueError('source_mismatch')
        event['accepted'] = True
        return [(arguments['category'], text)]
    except Exception as exc:
        # Optional dependency/native runtime failures must not lose memories.
        event['fallback_reason'] = type(exc).__name__
        return None
    finally:
        event['elapsed_ms'] = (time.perf_counter() - started) * 1000
        logger.info('hybrid_inference %s', event)
        if telemetry:
            try:
                telemetry(event)
            except Exception:
                logger.warning('hybrid_telemetry_failed')
