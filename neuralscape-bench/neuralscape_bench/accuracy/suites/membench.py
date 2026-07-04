"""MemBench (ACL 2025 Findings) — reproduced by MemPalace.

Canonical dataset: github.com/import-myself/Membench, ``MemData/FirstAgent/``
— one JSON file per category (highlevel, lowlevel_rec, RecMultiSession,
knowledge_update, comparative, conditional, noisy, aggregative, …). Files
range 10–70 MB, so the fetcher pulls a configurable category subset
(default: the three smallest, ~39 MB total).

File shape (verified against ``highlevel.json``):
``{"<topic>": [ {tid, message_list: [[{mid, user, assistant, time, place}…]…],
QA: {qid, question, answer, choices: {A..D}, ground_truth,
target_step_id: [[step, list_idx]…]}} ]}``

Each item becomes one NS conversation; each inner ``message_list`` list is a
session whose steps expand to a user turn + an assistant turn. MemBench is
multiple-choice — the judge compares the free-form NS answer against the
gold ``answer`` text (choice letters are carried along for reference).
"""

from __future__ import annotations

from pathlib import Path

from neuralscape_bench.accuracy.download import fetch_file, read_json
from neuralscape_bench.accuracy.sampling import stratified_sample
from neuralscape_bench.accuracy.schema import Conversation, QAItem, Session, SuiteData, Turn
from neuralscape_bench.accuracy.suites import Suite

BASE = "https://raw.githubusercontent.com/import-myself/Membench/master/MemData/FirstAgent"
ALL_CATEGORIES = (
    "highlevel", "lowlevel_rec", "RecMultiSession", "knowledge_update",
    "comparative", "conditional", "noisy", "aggregative", "highlevel_rec",
)
DEFAULT_CATEGORIES = ("highlevel", "lowlevel_rec", "RecMultiSession")


def _categories(options: dict | None) -> tuple[str, ...]:
    cats = (options or {}).get("categories") or DEFAULT_CATEGORIES
    if cats == "all" or cats == ["all"]:
        return ALL_CATEGORIES
    unknown = set(cats) - set(ALL_CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown MemBench categories {sorted(unknown)}; known: {ALL_CATEGORIES}")
    return tuple(cats)


def fetch(dest_dir: Path, *, categories: tuple[str, ...] = DEFAULT_CATEGORIES) -> dict:
    fetched = {}
    for cat in categories:
        path = fetch_file(f"{BASE}/{cat}.json", dest_dir / f"{cat}.json")
        payload = read_json(path)
        n_items = sum(len(v) for v in payload.values() if isinstance(v, list))
        if n_items == 0:
            raise ValueError(f"MemBench {cat}.json: no items found")
        fetched[cat] = n_items
    return {"categories": fetched}


def parse_file(payload: dict, *, category: str) -> SuiteData:
    """Pure parser over one MemBench category file (fixture-tested)."""
    out = SuiteData(suite="membench")
    for topic, items in payload.items():
        if not isinstance(items, list):
            continue
        for item in items:
            tid = item.get("tid", len(out.conversations))
            conv_id = f"{category}-{topic}-{tid}"
            sessions: list[Session] = []
            for li, steps in enumerate(item.get("message_list", [])):
                turns: list[Turn] = []
                for step in steps if isinstance(steps, list) else []:
                    mid = step.get("mid")
                    time = (step.get("time") or "").strip()
                    place = (step.get("place") or "").strip()
                    ctx = " / ".join(x for x in (time, place) if x)
                    user = (step.get("user") or "").strip()
                    assistant = (step.get("assistant") or "").strip()
                    if user:
                        content = f"({ctx}) {user}" if ctx else user
                        turns.append(Turn(role="user", content=content,
                                          turn_id=f"s{li}.t{mid}"))
                    if assistant:
                        turns.append(Turn(role="assistant", content=assistant,
                                          turn_id=f"s{li}.t{mid}"))
                if turns:
                    sessions.append(Session(session_id=f"s{li}", turns=tuple(turns)))
            qa = item.get("QA") or {}
            question = (qa.get("question") or "").strip()
            if not question or not sessions:
                continue

            evidence_turns: list[str] = []
            evidence_sessions: list[str] = []
            for pair in qa.get("target_step_id") or []:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    step, li = pair[0], pair[1]
                elif isinstance(pair, int):
                    step, li = pair, 0
                else:
                    continue
                evidence_turns.append(f"s{li}.t{step}")
                if f"s{li}" not in evidence_sessions:
                    evidence_sessions.append(f"s{li}")

            choices = qa.get("choices") or {}
            gold = str(qa.get("answer", "")).strip()
            ground_truth = str(qa.get("ground_truth", "")).strip()
            if ground_truth and ground_truth in choices and not gold:
                gold = str(choices[ground_truth])

            out.conversations.append(Conversation(conv_id=conv_id, sessions=tuple(sessions)))
            out.qa_items.append(QAItem(
                qa_id=f"{conv_id}-q{qa.get('qid', 0)}",
                conv_id=conv_id,
                question=question,
                gold_answer=gold,
                qtype=category,
                evidence_session_ids=tuple(evidence_sessions),
                evidence_turn_ids=tuple(evidence_turns),
                choices=tuple(sorted((str(k), str(v)) for k, v in choices.items())),
            ))
    return out


def load(dest_dir: Path, *, sample: int | None = None, seed: int = 42,
         options: dict | None = None) -> SuiteData:
    data = SuiteData(suite="membench")
    for cat in _categories(options):
        f = dest_dir / f"{cat}.json"
        if not f.exists():
            continue
        part = parse_file(read_json(f), category=cat)
        data.conversations.extend(part.conversations)
        data.qa_items.extend(part.qa_items)
    data.notes.append("multiple-choice suite; judged as free-form answer vs gold answer text")
    if sample is not None:
        data.qa_items = stratified_sample(data.qa_items, sample, seed=seed)
        keep = {qa.conv_id for qa in data.qa_items}
        data.conversations = [c for c in data.conversations if c.conv_id in keep]
    return data


def _fetch(dest_dir: Path) -> dict:
    return fetch(dest_dir, categories=DEFAULT_CATEGORIES)


SUITES = [
    Suite(
        name="membench",
        display="MemBench",
        fetch=_fetch,
        load=load,
        source="github.com/import-myself/Membench (MemData/FirstAgent)",
        license_note="See upstream repo license.",
        default_options={"categories": list(DEFAULT_CATEGORIES)},
    ),
]
