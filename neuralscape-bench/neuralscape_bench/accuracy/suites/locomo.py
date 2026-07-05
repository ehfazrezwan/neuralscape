"""LoCoMo (ACL 2024, snap-research/locomo) — mem0's headline benchmark.

Dataset: ``data/locomo10.json`` in the upstream repo — 10 very long
two-human conversations (~19-35 sessions each) with ~200 QA pairs per
conversation across 5 categories:

    1 multi-hop  2 temporal  3 open-domain/knowledge  4 single-hop
    5 adversarial (unanswerable; gold behavior is to say so)

Quirks handled here (verified against the real file):
- ``qa.category`` is sometimes an int, sometimes a str ("2");
- ``qa.evidence`` is sometimes a list, sometimes a str repr of one ("['D1:3']");
- ``qa.answer`` may be an int, or absent for category 5 (which carries
  ``adversarial_answer`` instead);
- image turns carry ``blip_caption`` (images themselves are not released) —
  the caption is folded into the turn text like mem0's own eval does.
"""

from __future__ import annotations

import ast
from pathlib import Path

from neuralscape_bench.accuracy.download import fetch_file, read_json
from neuralscape_bench.accuracy.sampling import stratified_sample
from neuralscape_bench.accuracy.schema import Conversation, QAItem, Session, SuiteData, Turn
from neuralscape_bench.accuracy.suites import Suite

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
FILENAME = "locomo10.json"

CATEGORY_LABELS = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def fetch(dest_dir: Path) -> dict:
    path = fetch_file(URL, dest_dir / FILENAME)
    data = read_json(path)
    if not isinstance(data, list) or len(data) != 10:
        raise ValueError(f"locomo10.json: expected a list of 10 conversations, got {type(data)}/{len(data)}")
    return {"file": str(path), "conversations": len(data)}


def _coerce_evidence(raw) -> list[str]:
    """Evidence is a list of dia_ids — or a str repr of one. Normalize."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return [raw] if raw else []
    if isinstance(raw, list):
        return [str(e) for e in raw if e]
    return [str(raw)]


def _dia_session(dia_id: str) -> str | None:
    """``D12:3`` → session id ``12`` (None when malformed)."""
    if ":" not in dia_id:
        return None
    head = dia_id.split(":", 1)[0].lstrip("Dd")
    return head or None


def _turn_text(turn: dict) -> str:
    text = (turn.get("text") or "").strip()
    caption = (turn.get("blip_caption") or "").strip()
    if caption:
        text = f"{text} [shares a photo: {caption}]" if text else f"[shares a photo: {caption}]"
    return text


def parse(data: list[dict]) -> SuiteData:
    """Pure parser over the loaded locomo10 JSON (fixture-tested)."""
    out = SuiteData(suite="locomo")
    for sample in data:
        conv_id = str(sample.get("sample_id", f"conv-{len(out.conversations)}"))
        conv = sample.get("conversation", {})
        speaker_a = conv.get("speaker_a", "Speaker A")

        session_nums = sorted(
            int(k.split("_")[1])
            for k, v in conv.items()
            if k.startswith("session_") and not k.endswith("date_time") and isinstance(v, list)
        )
        sessions: list[Session] = []
        for n in session_nums:
            turns: list[Turn] = []
            for t in conv.get(f"session_{n}", []):
                text = _turn_text(t)
                if not text:
                    continue
                speaker = t.get("speaker") or "?"
                turns.append(Turn(
                    role="user" if speaker == speaker_a else "assistant",
                    content=f"{speaker}: {text}",
                    speaker=speaker,
                    turn_id=t.get("dia_id"),
                ))
            if turns:
                sessions.append(Session(
                    session_id=str(n),
                    turns=tuple(turns),
                    date=conv.get(f"session_{n}_date_time"),
                ))
        out.conversations.append(Conversation(conv_id=conv_id, sessions=tuple(sessions)))

        for i, qa in enumerate(sample.get("qa", [])):
            try:
                cat = int(qa.get("category", 0))
            except (TypeError, ValueError):
                cat = 0
            is_adv = cat == 5
            answer = qa.get("adversarial_answer") if is_adv else qa.get("answer")
            if answer is None:
                answer = "Not mentioned in the conversation" if is_adv else ""
            evidence = _coerce_evidence(qa.get("evidence"))
            ev_sessions = tuple(dict.fromkeys(
                s for s in (_dia_session(e) for e in evidence) if s
            ))
            out.qa_items.append(QAItem(
                qa_id=f"{conv_id}-qa{i}",
                conv_id=conv_id,
                question=str(qa.get("question", "")),
                gold_answer=str(answer),
                qtype=f"{cat}-{CATEGORY_LABELS.get(cat, 'unknown')}",
                evidence_session_ids=ev_sessions,
                evidence_turn_ids=tuple(evidence),
                is_abstention=is_adv,
            ))
    out.notes.append("category 5 (adversarial) gold behavior is abstention")
    return out


def load(dest_dir: Path, *, sample: int | None = None, seed: int = 42,
         options: dict | None = None) -> SuiteData:
    data = parse(read_json(dest_dir / FILENAME))
    if sample is not None:
        data.qa_items = stratified_sample(data.qa_items, sample, seed=seed)
    return data


SUITES = [
    Suite(
        name="locomo",
        display="LoCoMo",
        fetch=fetch,
        load=load,
        source="github.com/snap-research/locomo (data/locomo10.json)",
        license_note="See LICENSE.txt in the upstream repo before redistribution.",
    ),
]
