"""
Correctness + first-hop-hit scoring for the token-savings benchmark.

We reuse the Track-Q normalization helpers (``_normalize_path``,
``_normalize_bare_symbol``, ``_parse_symbol_set``) so "did the agent reach the
right answer?" is judged the SAME way ICEBench judges recall — keeping the two
benchmarks consistent.

Two things are scored:

  * ``score_answer``   — did the agent's *submitted* answer reach the gold
    location / caller set? (correctness of a condition)
  * ``first_hop_hit``  — for the with-memory condition, did the *first* tool the
    agent used, ``code_memory``, return the gold location on its first call (so
    no file was read)? This is the "find-without-reading" signal.
"""

import re
from dataclasses import dataclass

from icebench.trackq.score import (
    _normalize_bare_symbol,
    _normalize_path,
    _parse_symbol_set,
)

# Line tolerance for symbol_lookup: a def may be reported at the decorator line,
# the `def`/`class` line, or one off; allow a small window.
LINE_TOLERANCE = 3
# Neighbors: treat as "correct" when the caller-set F1 clears this bar.
NEIGHBORS_F1_THRESHOLD = 0.5


@dataclass
class AnswerScore:
    correct: bool
    detail: dict


def _parse_location(loc: str) -> tuple[str, int | None]:
    """Parse an agent 'file:line' answer into (normalized_file, line|None)."""
    if not loc or not isinstance(loc, str):
        return "", None
    loc = loc.strip().strip("`").strip()
    m = re.match(r"^(.*?):(\d+)\s*$", loc)
    if m:
        return _normalize_path(m.group(1)), int(m.group(2))
    return _normalize_path(loc), None


def _set_prf(answer: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not answer and not gold:
        return 1.0, 1.0, 1.0
    if not answer or not gold:
        return 0.0, 0.0, 0.0
    tp = len(answer & gold)
    precision = tp / len(answer)
    recall = tp / len(gold)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_answer(op_class: str, answer: dict, gold: dict) -> AnswerScore:
    """Judge whether a submitted answer reached the gold target."""
    if answer.get("gave_up"):
        return AnswerScore(False, {"gave_up": True})

    if op_class in ("locate", "symbol_lookup"):
        ans_file, ans_line = _parse_location(answer.get("location", ""))
        gold_file = _normalize_path(gold.get("file", ""))
        file_ok = bool(ans_file) and ans_file == gold_file
        detail = {"ans_file": ans_file, "ans_line": ans_line, "gold_file": gold_file}
        if op_class == "locate":
            # nl_locate gold has no line -> file-level correctness (the agent was
            # asked for a location, not to name the symbol).
            return AnswerScore(file_ok, detail)
        # symbol_lookup: require file match AND (if a line was given) proximity.
        gold_line = gold.get("line")
        detail["gold_line"] = gold_line
        if not file_ok:
            return AnswerScore(False, detail)
        if ans_line is None or gold_line is None:
            return AnswerScore(True, detail)  # file matched, no line to check
        line_ok = abs(ans_line - int(gold_line)) <= LINE_TOLERANCE
        return AnswerScore(line_ok, detail)

    if op_class == "neighbors":
        ans = {_normalize_bare_symbol(c) for c in (answer.get("callers") or [])}
        ans.discard("")
        gld = {_normalize_bare_symbol(c) for c in gold.get("callers", [])}
        gld.discard("")
        p, r, f1 = _set_prf(ans, gld)
        return AnswerScore(
            f1 >= NEIGHBORS_F1_THRESHOLD,
            {"precision": p, "recall": r, "f1": f1, "n_ans": len(ans), "n_gold": len(gld)},
        )

    raise ValueError(f"unknown op_class: {op_class}")


def first_hop_hit(op_class: str, first_memory_raw: dict, gold: dict, k: int = 1) -> bool:
    """
    Did the first ``code_memory`` call return the gold target within top-k
    (locate/symbol_lookup) or clear the F1 bar (neighbors)?

    Args:
        first_memory_raw: the ``raw`` payload of the agent's first code_memory
            call (``{"ranked": [[file, symbol], ...]}`` or ``{"callers": [...]}``).
        gold: op-specific gold.
        k: rank cutoff for locate/symbol_lookup (1 or 5).
    """
    if not first_memory_raw:
        return False

    if op_class in ("locate", "symbol_lookup"):
        ranked = first_memory_raw.get("ranked") or []
        gold_norm = (_normalize_path(gold.get("file", "")), _normalize_bare_symbol(gold.get("symbol", "")))
        for file, sym in ranked[:k]:
            if (_normalize_path(file), _normalize_bare_symbol(sym)) == gold_norm:
                return True
        return False

    if op_class == "neighbors":
        callers = first_memory_raw.get("callers") or []
        ans = {_normalize_bare_symbol(c) for c in callers}
        ans.discard("")
        gld = {_normalize_bare_symbol(c) for c in gold.get("callers", [])}
        gld.discard("")
        _, _, f1 = _set_prf(ans, gld)
        return f1 >= NEIGHBORS_F1_THRESHOLD

    return False
