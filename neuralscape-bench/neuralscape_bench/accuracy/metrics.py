"""Retrieval attribution + score aggregation (pure — unit-tested).

Retrieval metric methodology: NS stores *distilled facts*, not raw turns, and
the REST search response does not echo the ingest ``run_id``. Each retrieved
memory is therefore attributed to the haystack session whose text best
lexically contains it (memories are extracted from exactly one session-sized
batch, so containment-argmax attribution is reliable). R@k = any of the
top-k retrieved memories attributes to a gold evidence session. Suites
without evidence annotations (DMR) skip the metric.
"""

from __future__ import annotations

import re
from collections import Counter

from neuralscape_bench.accuracy.schema import Conversation

_TOKEN = re.compile(r"[a-z0-9']+")

# Below this containment score the memory is likely paraphrased beyond
# lexical recognition (or matched a distractor) — attribution abstains.
MIN_ATTRIBUTION_SCORE = 0.25

_STOP = frozenset(
    "the a an and or of to in is are was were be been i you he she it we they "
    "my your his her its our their me him them this that these those with for "
    "on at by as do did does have has had not no so".split()
)


def _tokens(text: str) -> Counter:
    return Counter(t for t in _TOKEN.findall(text.lower()) if t not in _STOP)


def attribute_memory(memory_text: str, conversation: Conversation) -> tuple[str | None, float]:
    """Best-containment session for a distilled memory → (session_id, score).

    Score = fraction of the memory's content tokens (multiset) found in the
    session text. Returns (None, score) when below MIN_ATTRIBUTION_SCORE.
    """
    mem = _tokens(memory_text)
    if not mem:
        return None, 0.0
    total = sum(mem.values())
    best_id, best_score = None, 0.0
    for session in conversation.sessions:
        sess = _tokens(session.text())
        overlap = sum(min(c, sess.get(t, 0)) for t, c in mem.items())
        score = overlap / total
        if score > best_score:
            best_id, best_score = session.session_id, score
    if best_score < MIN_ATTRIBUTION_SCORE:
        return None, best_score
    return best_id, best_score


def recall_at_k(attributed: list[str | None], gold: tuple[str, ...] | list[str],
                k: int) -> bool | None:
    """True/False = gold session among the first-k attributions; None = no gold."""
    if not gold:
        return None
    gold_set = set(gold)
    return any(s in gold_set for s in attributed[:k] if s)


def aggregate(records: list[dict], *, k: int = 10) -> dict:
    """Aggregate per-question records into suite-level metrics.

    Record fields consumed: ``qtype``, ``correct`` (bool|None),
    ``is_abstention``, ``abstained`` (bool|None), ``retrieval_hit`` (bool|None).
    """
    def _acc(rows: list[dict]) -> dict:
        judged = [r for r in rows if r.get("correct") is not None]
        correct = sum(1 for r in judged if r["correct"])
        return {
            "n": len(rows),
            "judged": len(judged),
            "correct": correct,
            "accuracy": round(correct / len(judged), 4) if judged else None,
        }

    out: dict = {"overall": _acc(records), "by_type": {}}
    types = sorted({r.get("qtype", "unknown") for r in records})
    for t in types:
        out["by_type"][t] = _acc([r for r in records if r.get("qtype") == t])

    # Abstention behavior: on gold-abstention questions, did NS abstain or
    # (per judge) answer correctly that there is no information?
    abst = [r for r in records if r.get("is_abstention")]
    if abst:
        judged = [r for r in abst if r.get("correct") is not None]
        out["abstention"] = {
            "n": len(abst),
            "correct": sum(1 for r in judged if r["correct"]),
            "explicit_abstain_flag": sum(1 for r in abst if r.get("abstained")),
        }

    # Retrieval R@k over questions that have gold evidence.
    scored = [r for r in records if r.get("retrieval_hit") is not None]
    if scored:
        hits = sum(1 for r in scored if r["retrieval_hit"])
        out[f"retrieval_recall_at_{k}"] = {
            "n": len(scored),
            "hits": hits,
            "recall": round(hits / len(scored), 4),
        }
    return out
