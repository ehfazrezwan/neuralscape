"""Normalized data model shared by every accuracy suite (pure — unit-tested).

Each suite parser maps its native JSON into these types so the ingest /
answer / judge / report runners are suite-agnostic. Evidence annotations are
kept at whatever granularity the dataset provides (session ids, turn ids,
verbatim evidence text) — the retrieval metric degrades gracefully when a
suite lacks them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_USER_ID_SAFE = re.compile(r"[^a-zA-Z0-9_.\-]+")


def bench_user_id(suite: str, conv_id: str, namespace: str | None = None) -> str:
    """Per-conversation NS user id, e.g. ``bench-locomo-conv-26``.

    Must satisfy the service's ``_ID_PATTERN`` (``^[a-zA-Z0-9_.\\-]+$``) and
    its 100-char cap — unsafe chars collapse to ``-``.

    ``namespace`` (optional) prefixes the id so a per-PR mini-ingest lands in
    its own user space — e.g. ``bench-t11-locomo-conv-26`` — without colliding
    with the shared baseline store. Default (``None``) leaves ids unchanged.
    """
    prefix = f"bench-{namespace}-" if namespace else "bench-"
    raw = f"{prefix}{suite}-{conv_id}"
    return _USER_ID_SAFE.sub("-", raw)[:100].strip("-")


@dataclass(frozen=True)
class Turn:
    """One utterance. ``role`` is the NS conversation role (user/assistant);
    ``speaker`` preserves the dataset's display name for two-human corpora."""

    role: str                       # "user" | "assistant"
    content: str
    speaker: str | None = None
    turn_id: str | None = None      # dataset-native id (dia_id, mid, …)
    has_answer: bool = False        # turn-level evidence flag (LongMemEval)


@dataclass(frozen=True)
class Session:
    session_id: str
    turns: tuple[Turn, ...]
    date: str | None = None         # dataset-native timestamp string

    def text(self) -> str:
        """Plain-text render used for lexical session attribution."""
        return "\n".join(t.content for t in self.turns)


@dataclass(frozen=True)
class Conversation:
    """One ingestion unit → one NS user (``bench-<suite>-<conv_id>``)."""

    conv_id: str
    sessions: tuple[Session, ...]

    def session(self, session_id: str) -> Session | None:
        for s in self.sessions:
            if s.session_id == session_id:
                return s
        return None


@dataclass(frozen=True)
class QAItem:
    qa_id: str
    conv_id: str
    question: str
    gold_answer: str
    qtype: str                                  # suite-native category label
    evidence_session_ids: tuple[str, ...] = ()
    evidence_turn_ids: tuple[str, ...] = ()
    evidence_texts: tuple[str, ...] = ()        # verbatim evidence utterances
    is_abstention: bool = False                 # gold behavior = "I don't know"
    choices: tuple[tuple[str, str], ...] = ()   # multiple-choice (MemBench)
    rubric: tuple[str, ...] = ()                # grading rubric lines (BEAM)
    question_date: str | None = None            # asked-at date (LongMemEval)


@dataclass
class SuiteData:
    """Everything a runner needs for one suite: conversations + QA pairs."""

    suite: str
    conversations: list[Conversation] = field(default_factory=list)
    qa_items: list[QAItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def conversation(self, conv_id: str) -> Conversation | None:
        for c in self.conversations:
            if c.conv_id == conv_id:
                return c
        return None

    def stats(self) -> dict:
        n_sessions = sum(len(c.sessions) for c in self.conversations)
        n_turns = sum(len(s.turns) for c in self.conversations for s in c.sessions)
        n_chars = sum(
            len(t.content) for c in self.conversations for s in c.sessions for t in s.turns
        )
        by_type: dict[str, int] = {}
        for qa in self.qa_items:
            by_type[qa.qtype] = by_type.get(qa.qtype, 0) + 1
        return {
            "conversations": len(self.conversations),
            "sessions": n_sessions,
            "turns": n_turns,
            "conversation_chars": n_chars,
            "qa_items": len(self.qa_items),
            "qa_by_type": dict(sorted(by_type.items())),
        }
