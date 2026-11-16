"""Deterministic modality post-check for distilled ingest facts (issue #444).

An LLM distilling "X should do Y" sometimes writes "X must do Y". Such a fact is not
"directly restating the document", so it must not carry the ``explicit`` label. No LLM
call: the fact's obligation words are compared with those of the source clause that
best matches the fact.

Verdicts (:func:`check_modality`):

- ``"ok"``       the fact makes no obligation claim, or the located source clause carries one;
- ``"upgraded"`` the located source clause carries no obligation (a ``should`` became ``must``);
- ``"unknown"``  the fact claims an obligation but no source clause matches it well enough to
                 compare. We cannot show the document states it, so it is not ``explicit``.

Upgraded and unknown facts are labelled :data:`MODALITY_INFERRED_LEVEL`.

Known limits: lexical (English), stem-overlap matching; a heavily paraphrased fact becomes
``unknown``; "obligation" is a word class, not a strength scale (``need`` vs ``required`` are
treated alike); a soft ``should`` is never obligation. A tie between equally good clauses
gives the fact the benefit of the doubt (``ok`` if any tied clause carries an obligation).
"""

from __future__ import annotations

import math
import re

# Label for a fact whose obligation is not shown by the source: extractor-inferred, and not
# "deductive" (the stronger norm is not entailed by the source).
MODALITY_INFERRED_LEVEL = "inductive"

# Negated "obligations" first: "need not", "not required", "no need" are the opposite of
# an obligation, so they are removed before the obligation words are looked for.
_NEGATED_RE = re.compile(
    r"\b(?:need|needs|needed)\s+not\b|\bneedn'?t\b|\bno\s+need\b|"
    r"\b(?:do|does|did)(?:\s+not|n'?t)\s+(?:need|have)\s+to\b|"
    r"\b(?:is|are|was|were)(?:\s+not|n'?t)\s+(?:be\s+)?(?:required|mandatory|compulsory|obligatory|necessary|obliged|obligated)\b|"
    r"\bnot\s+(?:be\s+)?(?:required|mandatory|compulsory|obligatory|necessary|obliged|obligated)\b|"
    r"\bun(?:necessary|required)\b",
    re.IGNORECASE,
)
_OBLIGATION_RE = re.compile(
    r"\b(?:must(?:n'?t)?|shall|require[ds]?|requiring|requirements?|mandatory|mandated|mandates?|"
    r"obligatory|obliged|obligated|obligations?|compulsory|necessary|essential|imperative|"
    r"need(?:s|ed)?|(?:has|have|had|got)\s+to|gotta|(?:is|are)\s+to)\b",
    re.IGNORECASE,
)
# A label/value or limit statement in the source ("Employees: wear badges", "Minimum length: 12",
# "at least 12 characters") is normative as written, so a fact restating it with "must" is faithful.
_LIMIT_RE = re.compile(
    r"\b(?:minimum|maximum|at\s+least|at\s+most|no\s+more\s+than|no\s+fewer\s+than|not\s+exceed)\b", re.IGNORECASE
)
# "Label: short value" (at most 3 label words, at most 8 value words): a field/rule line, not a
# "Speaker: long reported sentence".
_LABEL_VALUE_RE = re.compile(r"^\s*(?P<label>[^:\n]{1,40}):\s*(?P<value>\S[^\n]*)$")


def _is_normative_form(clause: str) -> bool:
    if _LIMIT_RE.search(clause):
        return True
    m = _LABEL_VALUE_RE.match(clause)
    return bool(m and len(m["label"].split()) <= 3 and len(m["value"].split()) <= 8)

# Sentences end at . ! ? plus whitespace, at ! or ? with no space, or at a newline; clauses at ';'.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=[!?])(?=\S)|\n+")
_CLAUSE_SPLIT_RE = re.compile(r";")
_WORD_RE = re.compile(r"[a-z]{4,}")
_STOP = frozenset(
    ("that this with from have they them their will would should could might there which about into than then when "
     "what while where been being were also each every only over such some more most other these those must shall "
     "require required requires requiring need needs needed mandatory obligatory compulsory necessary essential "
     "imperative recommend recommended suggest ideally").split()
)
MIN_SHARED_STEMS = 2
MIN_SHARED_FRACTION = 0.4


def _stems(text: str) -> frozenset[str]:
    return frozenset(w[:5] for w in _WORD_RE.findall(text.lower()) if w not in _STOP)


def has_obligation(text: str) -> bool:
    return _OBLIGATION_RE.search(_NEGATED_RE.sub(" negated ", text)) is not None


class ModalitySource:
    """A source document split into clauses, with stem sets computed once."""

    def __init__(self, text: str):
        self.clauses: list[tuple[frozenset[str], bool]] = []
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            for clause in _CLAUSE_SPLIT_RE.split(sentence):
                stems = _stems(clause)
                if stems:
                    normative = has_obligation(clause) or _is_normative_form(clause)
                    self.clauses.append((stems, normative))

    def check(self, fact: str) -> str:
        if not has_obligation(fact):
            return "ok"
        fact_stems = _stems(fact)
        if not fact_stems:
            return "unknown"
        best, tied = 0, []
        for stems, normative in self.clauses:
            overlap = len(fact_stems & stems)
            if overlap > best:
                best, tied = overlap, [normative]
            elif overlap == best and overlap:
                tied.append(normative)
        if best < max(MIN_SHARED_STEMS, math.ceil(MIN_SHARED_FRACTION * len(fact_stems))):
            return "unknown"
        return "ok" if any(tied) else "upgraded"


def check_modality(fact: str, source: "str | ModalitySource") -> str:
    return (source if isinstance(source, ModalitySource) else ModalitySource(source)).check(fact)


def modality_upgraded(fact: str, source: "str | ModalitySource") -> bool:
    """True when ``fact`` must not be labelled ``explicit`` (``upgraded`` or ``unknown``)."""
    return check_modality(fact, source) != "ok"
