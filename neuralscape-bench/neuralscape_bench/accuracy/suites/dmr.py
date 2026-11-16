"""DMR — Deep Memory Retrieval (MemGPT paper; used by the Zep/Graphiti paper).

Canonical dataset: HuggingFace ``MemGPT/MSC-Self-Instruct``
(``msc_self_instruct.jsonl``, ~8.5 MB, Apache-2.0) — 500 samples built on
Multi-Session Chat. Each record holds:

- ``previous_dialogs``: 4 earlier sessions (turns are bare ``{"text": ...}``,
  strictly alternating Speaker 1 / Speaker 2; ``time_back`` like "3 days");
- ``dialog``: the final (5th) session, turns carry ``id`` = "Speaker 1"/"2";
- ``self_instruct``: {"B": question asked to Speaker 2's persona,
  "A": gold answer}.

The MemGPT paper scored ROUGE-L recall + an LLM judge; the Zep paper used an
LLM judge (their reported DMR: MemGPT 93.4%, Zep 94.8%). No evidence
annotations exist → retrieval R@k is skipped for this suite.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from neuralscape_bench.accuracy.download import fetch_file, hf_resolve_url, read_jsonl
from neuralscape_bench.accuracy.sampling import stratified_sample
from neuralscape_bench.accuracy.schema import Conversation, QAItem, Session, SuiteData, Turn
from neuralscape_bench.accuracy.suites import Suite

HF_REPO = "MemGPT/MSC-Self-Instruct"
FILENAME = "msc_self_instruct.jsonl"

# Deterministic event-time anchor for the FINAL session of every DMR
# conversation. A fixed past constant (not ingest-time "now") so occurred_at
# — which participates in the service's deterministic enqueue job key — is
# identical across resumes and re-runs. Previous sessions sit time_back
# earlier, per dataset semantics ("N hours/days ago" relative to the final
# session).
DMR_ANCHOR = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# "ago" optional: the dataset writes "3 days ago", some records just "3 days".
_TIME_BACK = re.compile(r"(\d+)\s*(hour|day|week|month|year)s?(?:\s*ago)?", re.I)
_UNIT_HOURS = {"hour": 1, "day": 24, "week": 24 * 7, "month": 24 * 30, "year": 24 * 365}


def _occurred_at_from(time_back: str | None) -> str:
    """Anchor-relative ISO event time; unparseable/absent ⇒ the anchor itself."""
    m = _TIME_BACK.search(str(time_back or ""))
    if not m:
        return DMR_ANCHOR.isoformat()
    hours = int(m.group(1)) * _UNIT_HOURS[m.group(2).lower()]
    return (DMR_ANCHOR - timedelta(hours=hours)).isoformat()


def fetch(dest_dir: Path) -> dict:
    path = fetch_file(hf_resolve_url(HF_REPO, FILENAME), dest_dir / FILENAME)
    records = read_jsonl(path)
    if len(records) != 500:
        raise ValueError(f"{FILENAME}: expected 500 records, got {len(records)}")
    return {"file": str(path), "records": len(records)}


def _turns_from(dialog: list[dict]) -> list[Turn]:
    """MSC turns alternate Speaker 1 / Speaker 2; some carry an ``id``."""
    turns: list[Turn] = []
    for i, t in enumerate(dialog):
        text = (t.get("text") or "").strip()
        if not text:
            continue
        speaker = t.get("id") or ("Speaker 1" if i % 2 == 0 else "Speaker 2")
        # Identity travels on the role field only. Prefixing the text with
        # "Speaker N:" too gave the extractor a second, literal channel it
        # sometimes latched onto, distilling unresolvable "Speaker 2 ..."
        # memories (run-3 failure analysis, 2026-07-05).
        turns.append(Turn(
            role="user" if speaker == "Speaker 1" else "assistant",
            content=text,
            speaker=speaker,
        ))
    return turns


def parse(records: list[dict]) -> SuiteData:
    """Pure parser over msc_self_instruct records (fixture-tested)."""
    out = SuiteData(suite="dmr")
    for idx, rec in enumerate(records):
        conv_id = str(rec.get("metadata", {}).get("initial_data_id") or f"dmr-{idx}")
        sessions: list[Session] = []
        for si, prev in enumerate(rec.get("previous_dialogs", [])):
            turns = _turns_from(prev.get("dialog", []))
            if turns:
                time_back = prev.get("time_back")
                sessions.append(Session(
                    session_id=f"s{si + 1}",
                    turns=tuple(turns),
                    date=f"{time_back} before the final session" if time_back else None,
                    occurred_at=_occurred_at_from(time_back),
                ))
        final = _turns_from(rec.get("dialog", []))
        if final:
            sessions.append(Session(
                session_id=f"s{len(sessions) + 1}", turns=tuple(final),
                occurred_at=DMR_ANCHOR.isoformat(),
            ))

        si_pair = rec.get("self_instruct", {}) or {}
        question = (si_pair.get("B") or "").strip()
        gold = (si_pair.get("A") or "").strip()
        if not question:
            continue
        out.conversations.append(Conversation(conv_id=conv_id, sessions=tuple(sessions)))
        out.qa_items.append(QAItem(
            qa_id=conv_id,
            conv_id=conv_id,
            question=question,
            gold_answer=gold,
            qtype="dmr",
        ))
    out.notes.append("no evidence annotations → retrieval R@k not computed for DMR")
    return out


def load(dest_dir: Path, *, sample: int | None = None, seed: int = 42,
         options: dict | None = None) -> SuiteData:
    data = parse(read_jsonl(dest_dir / FILENAME))
    if sample is not None:
        data.qa_items = stratified_sample(data.qa_items, sample, seed=seed)
        keep = {qa.conv_id for qa in data.qa_items}
        data.conversations = [c for c in data.conversations if c.conv_id in keep]
    return data


SUITES = [
    Suite(
        name="dmr",
        display="DMR (MSC-Self-Instruct)",
        fetch=fetch,
        load=load,
        source=f"huggingface.co/datasets/{HF_REPO}",
        license_note="Apache-2.0.",
    ),
]
