"""Soft category evidence is retrieval metadata, never an access policy.

The primary category retains the legacy contract. Evidence is bound to content
and primary category so edits cannot accidentally reuse stale predictions.
"""
from __future__ import annotations

import hashlib
import math

from schemas import CORE_MEMORY_CATEGORIES, GLOBAL_CATEGORIES, PROJECT_CATEGORIES, default_visibility_for_category

POLICY_VERSION = 'source-category-v2'


def _storage_policy(category: str) -> tuple:
    scope = 'project' if category in PROJECT_CATEGORIES else 'global' if category in GLOBAL_CATEGORIES else 'flexible'
    # Working context has a different lifetime/meaning from durable memories.
    return scope, default_visibility_for_category(category).value, category == 'task_context'


def category_mode(answer: dict, threshold: float) -> str | None:
    """Called only AFTER wire validation; soft acceptance is deliberately bounded.

    No evidence-based safety claim attaches to the exploratory numeric cutoffs.
    Keep the full distribution, including UNCERTAIN, rather than renormalizing.
    """
    primary = answer['choice']
    probabilities = answer['probabilities']
    if primary not in CORE_MEMORY_CATEGORIES:
        return None
    if answer['confidence'] >= threshold and probabilities[primary] >= threshold:
        return 'certain'
    ranked = sorted(((k, p) for k, p in probabilities.items() if p > 0 and k != 'UNCERTAIN'),
                    key=lambda item: -item[1])
    if not ranked or probabilities[primary] < .5 or probabilities.get('UNCERTAIN', 0) > .05:
        return None
    if sum(p for _, p in ranked[:3]) < .95 - 1e-9:
        return None
    # Even a small nonzero alternative with different scope/privacy must not
    # silently change storage behavior. No learned private/shared decisions.
    if any(k not in CORE_MEMORY_CATEGORIES or _storage_policy(k) != _storage_policy(primary) for k, _ in ranked):
        return None
    return 'soft'


def make_evidence(content: str, answer: dict, model: str, mode: str) -> dict:
    return {'probabilities': dict(answer['probabilities']), 'confidence': answer['confidence'],
            'primary_category': answer['choice'], 'model': model, 'mode': mode,
            'policy_version': POLICY_VERSION, 'content_sha256': hashlib.sha256(content.encode()).hexdigest()}


def validated_evidence(content: str, category: str | None, value) -> dict | None:
    """Ignore stale/malformed metadata on reads and legacy rows, fail closed."""
    if not isinstance(value, dict):
        return None
    try:
        probabilities = value['probabilities']
        expected = set(CORE_MEMORY_CATEGORIES) | {'UNCERTAIN'}
        if not isinstance(probabilities, dict) or set(probabilities) != expected:
            return None
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
            return None
        confidence = value['confidence']
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return None
        if (abs(sum(probabilities.values()) - 1) > .02 or category not in CORE_MEMORY_CATEGORIES
                or probabilities[category] < max(probabilities.values()) - 1e-6
                or value['primary_category'] != category
                or value['content_sha256'] != hashlib.sha256(content.encode()).hexdigest()
                or value['policy_version'] != POLICY_VERSION or value['mode'] not in ('certain', 'soft')
                or not isinstance(value['model'], str) or not value['model']):
            return None
        return {k: value[k] for k in ('probabilities', 'confidence', 'primary_category', 'model',
                                     'mode', 'policy_version', 'content_sha256')}
    except (KeyError, TypeError, ValueError):
        return None
