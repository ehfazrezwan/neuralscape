"""LongMemEval (ICLR 2025, xiaowu0162/LongMemEval) — Zep/MemPalace/Honcho suite.

Canonical download: HuggingFace ``xiaowu0162/longmemeval-cleaned`` (the
2025-09 cleaned release; MIT):

- ``longmemeval_s_cleaned.json``  (~277 MB, ~40 sessions / ~115k tokens per question)
- ``longmemeval_m_cleaned.json``  (~2.7 GB, ~500 sessions per question)
- ``longmemeval_oracle.json``     (~15 MB, evidence sessions only)

500 questions each, six types: single-session-user, single-session-assistant,
single-session-preference, temporal-reasoning, knowledge-update,
multi-session. Question ids ending ``_abs`` are abstention instances.

Structure per instance: ``haystack_sessions`` (lists of {role, content}
turns, evidence turns flagged ``has_answer: true``), ``haystack_session_ids``,
``haystack_dates``, ``answer_session_ids``, ``question_date``.

Every question gets its OWN haystack → one NS user per question
(``bench-<suite>-<question_id>``). That makes LongMemEval by far the most
expensive suite to ingest (500 × ~40 sessions for S) — see costs.py.
"""

from __future__ import annotations

from pathlib import Path

from neuralscape_bench.accuracy.download import fetch_file, hf_resolve_url, read_json
from neuralscape_bench.accuracy.sampling import stratified_sample
from neuralscape_bench.accuracy.schema import Conversation, QAItem, Session, SuiteData, Turn
from neuralscape_bench.accuracy.suites import Suite

HF_REPO = "xiaowu0162/longmemeval-cleaned"
FILES = {
    "longmemeval_s": "longmemeval_s_cleaned.json",
    "longmemeval_m": "longmemeval_m_cleaned.json",
    "longmemeval_oracle": "longmemeval_oracle.json",
}

QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
)


def _fetch_variant(variant: str):
    def fetch(dest_dir: Path) -> dict:
        filename = FILES[variant]
        path = fetch_file(hf_resolve_url(HF_REPO, filename), dest_dir / filename)
        data = read_json(path)
        if not isinstance(data, list) or len(data) != 500:
            raise ValueError(f"{filename}: expected 500 instances, got {type(data)}/{len(data) if isinstance(data, list) else '?'}")
        return {"file": str(path), "questions": len(data)}
    return fetch


def parse(data: list[dict], *, variant: str) -> SuiteData:
    """Pure parser over a loaded LongMemEval JSON (fixture-tested)."""
    out = SuiteData(suite=variant)
    for inst in data:
        qid = str(inst.get("question_id", ""))
        session_ids = [str(s) for s in inst.get("haystack_session_ids", [])]
        dates = inst.get("haystack_dates", [])
        raw_sessions = inst.get("haystack_sessions", [])
        answer_ids = {str(s) for s in inst.get("answer_session_ids", [])}

        sessions: list[Session] = []
        evidence_turns: list[str] = []
        for i, raw in enumerate(raw_sessions):
            sid = session_ids[i] if i < len(session_ids) else f"session_{i}"
            turns: list[Turn] = []
            for j, t in enumerate(raw if isinstance(raw, list) else []):
                content = (t.get("content") or "").strip()
                if not content:
                    continue
                has_answer = bool(t.get("has_answer"))
                turn_id = f"{sid}#{j}"
                if has_answer:
                    evidence_turns.append(turn_id)
                turns.append(Turn(
                    role=t.get("role") or "user",
                    content=content,
                    turn_id=turn_id,
                    has_answer=has_answer,
                ))
            if turns:
                sessions.append(Session(
                    session_id=sid,
                    turns=tuple(turns),
                    date=str(dates[i]) if i < len(dates) else None,
                ))

        out.conversations.append(Conversation(conv_id=qid, sessions=tuple(sessions)))
        out.qa_items.append(QAItem(
            qa_id=qid,
            conv_id=qid,
            question=str(inst.get("question", "")),
            gold_answer=str(inst.get("answer", "")),
            qtype=str(inst.get("question_type", "unknown")),
            evidence_session_ids=tuple(s for s in session_ids if s in answer_ids),
            evidence_turn_ids=tuple(evidence_turns),
            is_abstention=qid.endswith("_abs"),
            question_date=str(inst.get("question_date")) if inst.get("question_date") else None,
        ))
    out.notes.append(
        "retrieval metric skips the 30 *_abs abstention instances (no ground-truth location)"
    )
    return out


def _load_variant(variant: str):
    def load(dest_dir: Path, *, sample: int | None = None, seed: int = 42,
             options: dict | None = None) -> SuiteData:
        data = parse(read_json(dest_dir / FILES[variant]), variant=variant)
        if sample is not None:
            data.qa_items = stratified_sample(data.qa_items, sample, seed=seed)
            keep = {qa.conv_id for qa in data.qa_items}
            # Each question owns its haystack — drop conversations for
            # unsampled questions so ingest cost tracks the sample size.
            data.conversations = [c for c in data.conversations if c.conv_id in keep]
        return data
    return load


SUITES = [
    Suite(
        name="longmemeval_s",
        display="LongMemEval_S",
        fetch=_fetch_variant("longmemeval_s"),
        load=_load_variant("longmemeval_s"),
        source=f"huggingface.co/datasets/{HF_REPO} ({FILES['longmemeval_s']})",
        license_note="MIT (upstream repo).",
    ),
    Suite(
        name="longmemeval_m",
        display="LongMemEval_M",
        fetch=_fetch_variant("longmemeval_m"),
        load=_load_variant("longmemeval_m"),
        source=f"huggingface.co/datasets/{HF_REPO} ({FILES['longmemeval_m']})",
        license_note="MIT (upstream repo). 2.7 GB download.",
    ),
]
