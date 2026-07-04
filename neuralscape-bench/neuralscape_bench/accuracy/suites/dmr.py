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

from pathlib import Path

from neuralscape_bench.accuracy.download import fetch_file, hf_resolve_url, read_jsonl
from neuralscape_bench.accuracy.sampling import stratified_sample
from neuralscape_bench.accuracy.schema import Conversation, QAItem, Session, SuiteData, Turn
from neuralscape_bench.accuracy.suites import Suite

HF_REPO = "MemGPT/MSC-Self-Instruct"
FILENAME = "msc_self_instruct.jsonl"


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
        turns.append(Turn(
            role="user" if speaker == "Speaker 1" else "assistant",
            content=f"{speaker}: {text}",
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
                ))
        final = _turns_from(rec.get("dialog", []))
        if final:
            sessions.append(Session(session_id=f"s{len(sessions) + 1}", turns=tuple(final)))

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
