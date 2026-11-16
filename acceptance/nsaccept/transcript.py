"""A Claude Code session transcript (JSONL) in the shape the client really writes.

Real entries are ``{type, uuid, parentUuid, sessionId, cwd, timestamp,
isSidechain, message: {role, content}}`` where ``content`` is a plain string for
a typed prompt and a block list (``text``, ``thinking``, ``tool_use``,
``tool_result``) for everything else. The plugin reads exactly that shape
(PR #452). Contents here are synthetic: structure only, never real text.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def thinking_block(text: str = "synthetic reasoning that is not conversation text") -> dict:
    return {"type": "thinking", "thinking": text}


def tool_use_block(tool_id: str, name: str, tool_input: dict) -> dict:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


class Transcript:
    """An append-only transcript file; every method writes its entry immediately."""

    def __init__(self, path: Path, session_id: str, cwd: str) -> None:
        self.path = Path(path)
        self.session_id = session_id
        self.cwd = cwd
        self._n = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():  # another Session over the same sandbox must not wipe an existing transcript
            self.path.write_text("")

    # ── entry builders ────────────────────────────────────────
    def _common(self, **extra) -> dict:
        self._n += 1
        return {
            "uuid": f"uuid-{self._n}",
            "parentUuid": f"uuid-{self._n - 1}" if self._n > 1 else None,
            "sessionId": self.session_id,
            "cwd": self.cwd,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "isSidechain": False,
            **extra,
        }

    def _write(self, entry: dict) -> dict:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return entry

    def user(self, text: str, **flags) -> dict:
        """A typed prompt (string content). ``flags``: isMeta, isSidechain, isCompactSummary, ..."""
        return self._write({"type": "user", **self._common(**flags), "message": {"role": "user", "content": text}})

    def assistant(self, *blocks, **flags) -> dict:
        """One assistant entry; plain strings become text blocks."""
        content = [text_block(b) if isinstance(b, str) else b for b in blocks]
        return self._write(
            {
                "type": "assistant",
                **self._common(**flags),
                "message": {"role": "assistant", "model": "synthetic-model", "content": content},
            }
        )

    def tool_result(self, tool_id: str, output: str) -> dict:
        """The user-role entry that carries a tool's result (no text of its own)."""
        return self._write(
            {
                "type": "user",
                **self._common(),
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": output}],
                },
            }
        )

    # ── whole turns ───────────────────────────────────────────
    def turn(self, prompt: str, answer: str, *, tool: tuple[str, dict, str] | None = None) -> None:
        """prompt, [tool call + result], final answer: what one real turn looks like.

        ``tool`` = (tool name, tool input, tool output).
        """
        self.user(prompt)
        if tool is not None:
            name, tool_input, output = tool
            tool_id = f"toolu_synthetic_{self._n}"
            self.assistant(thinking_block(), tool_use_block(tool_id, name, tool_input))
            self.tool_result(tool_id, output)
        self.assistant(answer)

    def lines(self) -> list[dict]:
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
