"""BEAM (vectorize-io/agent-memory-benchmark) — Honcho scores on it.

Canonical dataset: ``data/beam/<tier>/{queries,documents}.json.gz`` in
github.com/vectorize-io/agent-memory-benchmark (HF mirror:
``Mohammadta/BEAM``). Tiers: 100k / 500k / 1m / 10m context tokens.

Schema (verified against the 100k tier):
- ``queries.json.gz``: ``{id, query, gold_answers[], gold_ids[], user_id,
  meta: {question_category, conversation_id, rubric[], why_unanswerable?}}``
  — 400 queries at 100k (20 users × 20), 10 categories × 40.
- ``documents.json.gz``: ``{id, content, user_id, timestamp}`` where content
  is a dialogue with ``[<date> | Turn <n>] <Role>: <text>`` markers.

Haystack assembly: one NS conversation per BEAM ``user_id``; each document
becomes one session (turns parsed from the markers). ``gold_ids`` are
document ids → session-level retrieval evidence.
"""

from __future__ import annotations

import re
from pathlib import Path

from neuralscape_bench.accuracy.download import fetch_file, read_json
from neuralscape_bench.accuracy.sampling import stratified_sample
from neuralscape_bench.accuracy.schema import Conversation, QAItem, Session, SuiteData, Turn
from neuralscape_bench.accuracy.suites import Suite

BASE = "https://raw.githubusercontent.com/vectorize-io/agent-memory-benchmark/main/data/beam"
TIERS = ("100k", "500k", "1m", "10m")
DEFAULT_TIER = "100k"

# "[March-15-2024 | Turn 0] User: text" — the date is omitted after Turn 0.
_MARKER = re.compile(
    r"\[(?:(?P<date>[^\]|]+)\s*\|\s*)?Turn\s+(?P<turn>\d+)\]\s*(?P<role>\w+):",
)


def _tier(options: dict | None) -> str:
    tier = (options or {}).get("tier", DEFAULT_TIER)
    if tier not in TIERS:
        raise ValueError(f"Unknown BEAM tier {tier!r}; expected one of {TIERS}")
    return tier


def fetch(dest_dir: Path, *, tier: str = DEFAULT_TIER) -> dict:
    paths = {}
    for name in ("queries", "documents"):
        fn = f"{tier}-{name}.json.gz"
        paths[name] = fetch_file(f"{BASE}/{tier}/{name}.json.gz", dest_dir / fn)
    queries = read_json(paths["queries"])
    documents = read_json(paths["documents"])
    if not queries or not documents:
        raise ValueError(f"BEAM {tier}: empty queries or documents")
    missing = {k for q in queries for k in ("id", "query", "user_id") if k not in q}
    if missing:
        raise ValueError(f"BEAM {tier}: queries missing keys {missing}")
    return {"tier": tier, "queries": len(queries), "documents": len(documents)}


def parse_document_content(content: str) -> tuple[list[Turn], str | None]:
    """Split a BEAM document's marked-up dialogue into turns (fixture-tested).

    Returns (turns, first_date). Unmarked leading text is discarded (the
    upstream format always opens with a Turn 0 marker).
    """
    turns: list[Turn] = []
    first_date: str | None = None
    matches = list(_MARKER.finditer(content))
    for i, m in enumerate(matches):
        date = (m.group("date") or "").strip() or None
        if date and first_date is None:
            first_date = date
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[m.end():end].strip()
        if not text:
            continue
        role_raw = m.group("role").lower()
        turns.append(Turn(
            role="user" if role_raw == "user" else "assistant",
            content=text,
            turn_id=m.group("turn"),
        ))
    return turns, first_date


def parse(queries: list[dict], documents: list[dict], *, tier: str) -> SuiteData:
    """Pure parser over loaded BEAM queries + documents (fixture-tested)."""
    out = SuiteData(suite="beam")
    by_user: dict[str, list[dict]] = {}
    for doc in documents:
        by_user.setdefault(str(doc.get("user_id")), []).append(doc)

    for user_id in sorted(by_user):
        sessions = []
        for doc in by_user[user_id]:
            turns, first_date = parse_document_content(doc.get("content") or "")
            if turns:
                sessions.append(Session(
                    session_id=str(doc.get("id")),
                    turns=tuple(turns),
                    date=str(doc.get("timestamp") or first_date or "") or None,
                ))
        out.conversations.append(Conversation(conv_id=user_id, sessions=tuple(sessions)))

    for q in queries:
        meta = q.get("meta") or {}
        category = str(meta.get("question_category", "unknown"))
        gold_answers = [str(a) for a in (q.get("gold_answers") or []) if a]
        out.qa_items.append(QAItem(
            qa_id=str(q.get("id")),
            conv_id=str(q.get("user_id")),
            question=str(q.get("query", "")),
            gold_answer=" | ".join(gold_answers),
            qtype=category,
            evidence_session_ids=tuple(str(g) for g in (q.get("gold_ids") or [])),
            is_abstention=category == "abstention",
            rubric=tuple(str(r) for r in (meta.get("rubric") or [])),
        ))
    out.notes.append(f"tier={tier}")
    return out


def _fetch(dest_dir: Path) -> dict:
    return fetch(dest_dir, tier=DEFAULT_TIER)


def load(dest_dir: Path, *, sample: int | None = None, seed: int = 42,
         options: dict | None = None) -> SuiteData:
    tier = _tier(options)
    if not (dest_dir / f"{tier}-queries.json.gz").exists():
        fetch(dest_dir, tier=tier)
    queries = read_json(dest_dir / f"{tier}-queries.json.gz")
    documents = read_json(dest_dir / f"{tier}-documents.json.gz")
    data = parse(queries, documents, tier=tier)
    if sample is not None:
        data.qa_items = stratified_sample(data.qa_items, sample, seed=seed)
        keep = {qa.conv_id for qa in data.qa_items}
        data.conversations = [c for c in data.conversations if c.conv_id in keep]
    return data


SUITES = [
    Suite(
        name="beam",
        display="BEAM",
        fetch=_fetch,
        load=load,
        source="github.com/vectorize-io/agent-memory-benchmark (data/beam/<tier>/)",
        license_note="See upstream repo license.",
        default_options={"tier": DEFAULT_TIER},
    ),
]
