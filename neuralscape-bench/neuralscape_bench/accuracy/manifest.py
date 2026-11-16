"""Resume manifests: which conversations/sessions/questions are already done.

A manifest is a plain JSON file under ``neuralscape-bench/state/`` (gitignored)
keyed by suite + stack fingerprint, so re-running after an interruption skips
completed work instead of re-storing (writes are async — poll, never
re-store). Pure logic (load/mark/should-skip) is unit-tested.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent.parent / "state"


class IngestManifest:
    """Tracks per-conversation session ingestion for one suite + target."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict = {"conversations": {}, "stats": {}}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        self._data.setdefault("conversations", {})
        self._data.setdefault("stats", {})

    @classmethod
    def for_run(cls, suite: str, target_label: str, state_dir: Path = STATE_DIR) -> "IngestManifest":
        state_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in target_label)
        return cls(state_dir / f"ingest-{suite}-{safe}.json")

    # ── queries ──
    def sessions_done(self, conv_id: str) -> set[str]:
        return set(self._data["conversations"].get(conv_id, {}).get("sessions_done", []))

    def is_conversation_done(self, conv_id: str, session_ids: list[str]) -> bool:
        return set(session_ids) <= self.sessions_done(conv_id)

    # ── mutations (each saves — cheap, and crash-safe by design) ──
    def mark_session(self, conv_id: str, session_id: str, *, task_id: str | None = None,
                     elapsed_s: float | None = None, est_tokens: int | None = None) -> None:
        conv = self._data["conversations"].setdefault(conv_id, {"sessions_done": []})
        if session_id not in conv["sessions_done"]:
            conv["sessions_done"].append(session_id)
        meta = conv.setdefault("sessions_meta", {})
        meta[session_id] = {
            "task_id": task_id,
            "elapsed_s": round(elapsed_s, 2) if elapsed_s is not None else None,
            "est_tokens": est_tokens,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

    def record_stats(self, **kw) -> None:
        self._data["stats"].update(kw)
        self.save()

    def session_metas(self, pairs: set[tuple[str, str]] | None = None) -> list[dict]:
        """Per-session ingest records (``elapsed_s`` is None when not recorded).

        ``pairs`` restricts to ``(conv_id, session_id)`` pairs, so a shared
        state file never leaks sessions of other datasets or samples.
        """
        out = []
        for conv_id, c in self._data["conversations"].items():
            for session_id, meta in c.get("sessions_meta", {}).items():
                if pairs is None or (conv_id, session_id) in pairs:
                    out.append({**meta, "conv_id": conv_id, "session_id": session_id})
        return out

    def totals(self, pairs: set[tuple[str, str]] | None = None) -> dict:
        """Ingest totals. A total with any unrecorded component is ``None``
        (unknown), never a partial sum or zero."""
        metas = self.session_metas(pairs)
        done = {(cid, sid) for cid, c in self._data["conversations"].items()
                for sid in c.get("sessions_done", [])
                if pairs is None or (cid, sid) in pairs}
        wall = [m.get("elapsed_s") for m in metas]
        toks = [m.get("est_tokens") for m in metas]
        return {
            "conversations": len({cid for cid, _ in done}),
            "sessions": len(done),
            "ingest_wall_s": None if None in wall else round(sum(wall), 1),
            "est_tokens": None if None in toks else sum(toks),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1))
        tmp.replace(self.path)


def load_done_qa_ids(jsonl_path: Path) -> set[str]:
    """qa_ids already present in an answers/judged JSONL (resume support)."""
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("qa_id"):
                done.add(rec["qa_id"])
    return done


def append_jsonl(jsonl_path: Path, record: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl_records(jsonl_path: Path) -> list[dict]:
    out: list[dict] = []
    if not jsonl_path.exists():
        return out
    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out
