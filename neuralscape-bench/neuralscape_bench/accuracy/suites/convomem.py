"""ConvoMem (Salesforce) — reproduced by MemPalace.

Canonical dataset: HuggingFace ``Salesforce/ConvoMem``, path
``core_benchmark/evidence_questions/<category>/<n>_evidence/<uuid>_<persona>.json``.
~75k QA pairs across 6 evidence categories:

    user_evidence, assistant_facts_evidence, changing_evidence,
    abstention_evidence, preference_evidence, implicit_connection_evidence

Each file is ``{"evidence_items": [...]}``; an item holds ``question``,
``answer``, ``message_evidences`` ([{speaker, text}]), and ``conversations``
([{id, containsEvidence, messages: [{speaker, text}]}]).

The full corpus is thousands of ~1 MB files — the fetcher downloads a
deterministic per-category file subset (``files_per_category``, default 2)
listed via the public HF tree API, which already yields hundreds of QA
items. Each evidence item becomes one NS conversation (its ``conversations``
are the sessions); evidence sessions are those with ``containsEvidence``.
"""

from __future__ import annotations

import json
from pathlib import Path

from neuralscape_bench.accuracy.download import fetch_file, hf_list_tree, hf_resolve_url
from neuralscape_bench.accuracy.sampling import stratified_sample
from neuralscape_bench.accuracy.schema import Conversation, QAItem, Session, SuiteData, Turn
from neuralscape_bench.accuracy.suites import Suite

HF_REPO = "Salesforce/ConvoMem"
BASE_PATH = "core_benchmark/evidence_questions"
CATEGORIES = (
    "user_evidence",
    "assistant_facts_evidence",
    "changing_evidence",
    "abstention_evidence",
    "preference_evidence",
    "implicit_connection_evidence",
)
DEFAULT_FILES_PER_CATEGORY = 2


def _lowest_bucket(cat: str) -> str:
    """Smallest ``<n>_evidence`` bucket available for a category.

    Buckets vary per category (e.g. ``changing_evidence`` starts at
    ``2_evidence`` — a "changing" fact needs at least two mentions).
    """
    tree = hf_list_tree(HF_REPO, f"{BASE_PATH}/{cat}")
    buckets = sorted(
        (int(name.split("_")[0]), name)
        for name in (Path(e["path"]).name for e in tree)
        if name.endswith("_evidence") and name.split("_")[0].isdigit()
    )
    if not buckets:
        raise ValueError(f"ConvoMem: no evidence buckets listed under {cat}")
    return buckets[0][1]


def fetch(dest_dir: Path, *, files_per_category: int = DEFAULT_FILES_PER_CATEGORY) -> dict:
    """Download a deterministic per-category subset (sorted-order head)."""
    fetched: dict[str, list[str]] = {}
    for cat in CATEGORIES:
        bucket = _lowest_bucket(cat)
        tree = hf_list_tree(HF_REPO, f"{BASE_PATH}/{cat}/{bucket}")
        files = sorted(
            e["path"] for e in tree
            if e.get("type") == "file" and e["path"].endswith(".json")
        )[:files_per_category]
        if not files:
            raise ValueError(f"ConvoMem: no files listed under {cat}/{bucket}")
        fetched[cat] = []
        for path_in_repo in files:
            local = dest_dir / cat / Path(path_in_repo).name
            fetch_file(hf_resolve_url(HF_REPO, path_in_repo), local)
            fetched[cat].append(local.name)
    return {"files_per_category": files_per_category, "files": fetched}


def parse_file(payload: dict, *, category: str, file_stem: str) -> SuiteData:
    """Pure parser over one ConvoMem evidence file (fixture-tested)."""
    out = SuiteData(suite="convomem")
    for i, item in enumerate(payload.get("evidence_items", [])):
        # Category-namespaced: file stems repeat across categories (the known
        # cross-category collision), and conv_id doubles as the NS user id —
        # a collision ingests two unrelated conversations into one user's
        # memory space and breaks qa_id joins/resume.
        conv_id = f"{category}-{file_stem}-{i}"
        sessions: list[Session] = []
        evidence_sessions: list[str] = []
        for conv in item.get("conversations", []):
            sid = str(conv.get("id", len(sessions)))
            turns = []
            for m in conv.get("messages", []):
                text = (m.get("text") or "").strip()
                if not text:
                    continue
                speaker = (m.get("speaker") or "").strip()
                turns.append(Turn(
                    role="user" if speaker.lower() == "user" else "assistant",
                    content=text,
                    speaker=speaker or None,
                ))
            if not turns:
                continue
            sessions.append(Session(session_id=sid, turns=tuple(turns)))
            if conv.get("containsEvidence"):
                evidence_sessions.append(sid)

        question = (item.get("question") or "").strip()
        if not question or not sessions:
            continue
        out.conversations.append(Conversation(conv_id=conv_id, sessions=tuple(sessions)))
        out.qa_items.append(QAItem(
            qa_id=conv_id,
            conv_id=conv_id,
            question=question,
            gold_answer=str(item.get("answer", "")),
            qtype=category,
            evidence_session_ids=tuple(evidence_sessions),
            evidence_texts=tuple(
                (e.get("text") or "").strip()
                for e in item.get("message_evidences", []) if e.get("text")
            ),
            is_abstention=category == "abstention_evidence",
        ))
    return out


def load(dest_dir: Path, *, sample: int | None = None, seed: int = 42,
         options: dict | None = None) -> SuiteData:
    data = SuiteData(suite="convomem")
    for cat in CATEGORIES:
        cat_dir = dest_dir / cat
        if not cat_dir.is_dir():
            continue
        for f in sorted(cat_dir.glob("*.json")):
            part = parse_file(json.loads(f.read_text()), category=cat, file_stem=f.stem[:40])
            data.conversations.extend(part.conversations)
            data.qa_items.extend(part.qa_items)
    data.notes.append(
        "subset: lowest evidence bucket per category, deterministic file head "
        "(full corpus is ~75k QA across thousands of files)"
    )
    if sample is not None:
        data.qa_items = stratified_sample(data.qa_items, sample, seed=seed)
        keep = {qa.conv_id for qa in data.qa_items}
        data.conversations = [c for c in data.conversations if c.conv_id in keep]
    return data


def _fetch(dest_dir: Path) -> dict:
    return fetch(dest_dir)


SUITES = [
    Suite(
        name="convomem",
        display="ConvoMem",
        fetch=_fetch,
        load=load,
        source=f"huggingface.co/datasets/{HF_REPO} ({BASE_PATH})",
        license_note="See dataset card on HuggingFace.",
        default_options={"files_per_category": DEFAULT_FILES_PER_CATEGORY},
    ),
]
