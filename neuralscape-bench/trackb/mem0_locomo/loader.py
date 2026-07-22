"""LoCoMo dataset loader (reuses existing parser or reads directly)."""

from __future__ import annotations

import json
from pathlib import Path

# Read-only reuse of existing bench schemas
from neuralscape_bench.accuracy.schema import (
    Conversation,
    QAItem,
    Session,
    SuiteData,
    Turn,
)


CATEGORY_LABELS = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def load_locomo(dataset_path: Path) -> SuiteData:
    """Load LoCoMo dataset from locomo10.json.

    Parses 10 long conversations with ~200 QA pairs per conversation.
    Categories: 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial.

    Args:
        dataset_path: Path to locomo10.json

    Returns:
        SuiteData with conversations and QA items
    """
    with open(dataset_path, "r") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected list of conversations, got {type(raw_data)}")

    if len(raw_data) == 0:
        raise ValueError("Expected at least 1 conversation, got 0")

    data = SuiteData(suite="mem0-locomo")

    for sample in raw_data:
        conv_id = str(sample.get("sample_id", f"conv-{len(data.conversations)}"))
        conv_dict = sample.get("conversation", {})
        speaker_a = conv_dict.get("speaker_a", "Speaker A")

        # Parse sessions
        session_nums = sorted(
            int(k.split("_")[1])
            for k in conv_dict.keys()
            if k.startswith("session_") and not k.endswith("date_time")
            and isinstance(conv_dict.get(k), list)
        )

        sessions: list[Session] = []
        for n in session_nums:
            turns: list[Turn] = []
            for turn_dict in conv_dict.get(f"session_{n}", []):
                text = (turn_dict.get("text") or "").strip()
                caption = (turn_dict.get("blip_caption") or "").strip()
                if caption:
                    text = f"{text} [shares a photo: {caption}]" if text else f"[shares a photo: {caption}]"
                if not text:
                    continue

                speaker = turn_dict.get("speaker") or "?"
                turns.append(Turn(
                    role="user" if speaker == speaker_a else "assistant",
                    content=f"{speaker}: {text}",
                    speaker=speaker,
                    turn_id=turn_dict.get("dia_id"),
                ))

            if turns:
                sessions.append(Session(
                    session_id=str(n),
                    turns=tuple(turns),
                    date=conv_dict.get(f"session_{n}_date_time"),
                ))

        data.conversations.append(Conversation(conv_id=conv_id, sessions=tuple(sessions)))

        # Parse QA items
        for i, qa_dict in enumerate(sample.get("qa", [])):
            try:
                cat = int(qa_dict.get("category", 0))
            except (TypeError, ValueError):
                cat = 0

            is_adversarial = cat == 5
            answer = qa_dict.get("adversarial_answer") if is_adversarial else qa_dict.get("answer")
            if answer is None:
                answer = "Not mentioned in the conversation" if is_adversarial else ""

            # Evidence parsing (can be list or string repr)
            evidence = qa_dict.get("evidence") or []
            if isinstance(evidence, str):
                try:
                    import ast
                    evidence = ast.literal_eval(evidence)
                except (ValueError, SyntaxError):
                    evidence = [evidence] if evidence else []
            if not isinstance(evidence, list):
                evidence = [str(evidence)]
            evidence = [str(e) for e in evidence if e]

            # Extract session IDs from evidence (D1:3 -> session "1")
            ev_sessions = []
            for e in evidence:
                if ":" in e:
                    head = e.split(":", 1)[0].lstrip("Dd")
                    if head:
                        ev_sessions.append(head)

            data.qa_items.append(QAItem(
                qa_id=f"{conv_id}-qa{i}",
                conv_id=conv_id,
                question=str(qa_dict.get("question", "")),
                gold_answer=str(answer),
                qtype=f"{cat}-{CATEGORY_LABELS.get(cat, 'unknown')}",
                evidence_session_ids=tuple(dict.fromkeys(ev_sessions)),
                evidence_turn_ids=tuple(evidence),
                is_abstention=is_adversarial,
            ))

    data.notes.append("mem0 LoCoMo evaluation (Track B)")
    data.notes.append("category 5 (adversarial) gold behavior is abstention")

    return data
