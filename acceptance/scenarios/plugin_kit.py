"""Drive the plugin's BUILT hook scripts the way Claude Code does, for the plugin scenarios.

One ``Session`` is one Claude Code session: a sandbox directory set that
survives across hook calls (observation buffers, cursors, snapshots live in
CLAUDE_PLUGIN_DATA), a transcript file in the real shape, and the real hook
payloads (``hook_event_name``, ``session_id``, ``transcript_path``, ``cwd``,
``tool_name`` / ``tool_input`` / ``tool_response`` ...). The URL option points at a
``RecordingProxy`` so scenarios can assert on exactly what a hook sent.

Nothing here imports plugin source: it runs ``node scripts/<hook>.js``.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from nsaccept import eventually
from nsaccept.client import NSClient
from nsaccept.hooks import HookResult, HookSandbox, run_hook
from nsaccept.proxy import FLUSH, RecordingProxy
from nsaccept.runctx import Ctx
from nsaccept.transcript import Transcript

# The Stop hook waits up to 3 s for the transcript to hold the Stop event's final
# answer; our transcripts are complete before the hook runs.
NO_LAG_WAIT = {"NEURALSCAPE_TRANSCRIPT_LAG_WAIT_MS": "0"}


def option(name: str, value: str) -> dict[str, str]:
    """A plugin option as Claude Code exposes it to hooks."""
    return {f"CLAUDE_PLUGIN_OPTION_{name}": value}


class Session:
    def __init__(
        self,
        ctx: Ctx,
        *,
        project_id: str | None,
        proxy: RecordingProxy | None = None,
        url: str | None = None,
        user_id: str | None = None,
        token: str | None = "",
        session_id: str | None = None,
        sandbox: HookSandbox | None = None,
        extra_env: dict[str, str] | None = None,
        work_name: str = "work",
        cwd: Path | None = None,
    ) -> None:
        self.ctx = ctx
        self.project_id = project_id
        self.proxy = proxy
        self.url = url or (proxy.url if proxy else ctx.base_url)
        self.user_id = user_id or ctx.user_a
        self.token = ctx.token_a if token == "" else token  # explicit None = no API_KEY at all
        self.session_id = session_id or f"acc-{ctx.run_id}-{uuid.uuid4().hex[:8]}"
        self._own_sandbox = sandbox is None
        self.sandbox = sandbox or HookSandbox(work_name)
        self.extra_env = {**NO_LAG_WAIT, **(extra_env or {})}
        self.cwd = Path(cwd) if cwd else self.sandbox.work  # the project directory Claude Code reports
        self.transcript = Transcript(self.sandbox.data_dir.parent / f"{self.session_id}.jsonl", self.session_id, str(self.cwd))
        self.results: list[HookResult] = []

    def close(self) -> None:
        if self._own_sandbox:
            self.sandbox.close()

    # ── plumbing ──────────────────────────────────────────────
    def payload(self, event: str, **fields) -> dict:
        return {
            "session_id": self.session_id,
            "transcript_path": str(self.transcript.path),
            "cwd": str(self.cwd),
            "permission_mode": "default",
            "hook_event_name": event,
            **fields,
        }

    def hook(self, script: str, payload: dict, *, env: dict[str, str] | None = None, raw_stdin: str | None = None) -> HookResult:
        res = run_hook(
            script,
            payload,
            base_url=self.url,
            token=self.token,
            user_id=self.user_id,
            project_id=self.project_id,
            extra_env={**self.extra_env, **(env or {})},
            sandbox=self.sandbox,
            raw_stdin=raw_stdin,
        )
        self.results.append(res)
        return res

    # ── the hooks, with the payloads Claude Code sends ────────
    def session_start(self, source: str = "startup", **env) -> HookResult:
        return self.hook("session-start", self.payload("SessionStart", source=source, model="synthetic-model"), env=env or None)

    def user_prompt(self, prompt: str, **env) -> HookResult:
        return self.hook("user-prompt-submit", self.payload("UserPromptSubmit", prompt=prompt), env=env or None)

    def post_tool(self, tool_name: str, tool_input: dict, tool_response, **env) -> HookResult:
        return self.hook(
            "post-tool-use",
            self.payload(
                "PostToolUse",
                tool_name=tool_name,
                tool_input=tool_input,
                tool_response=tool_response,
                tool_use_id=f"toolu_{uuid.uuid4().hex[:12]}",
            ),
            env=env or None,
        )

    def bash(self, command: str, stdout: str, **env) -> HookResult:
        """PostToolUse for a Bash call: tool_response is an object, as the client sends it."""
        return self.post_tool(
            "Bash",
            {"command": command, "description": "synthetic command"},
            {"stdout": stdout, "stderr": "", "interrupted": False, "isImage": False},
            **env,
        )

    def stop(self, last_assistant_message: str | None = None, **env) -> HookResult:
        fields: dict = {"stop_hook_active": False}
        if last_assistant_message is not None:
            fields["last_assistant_message"] = last_assistant_message
        return self.hook("session-end", self.payload("Stop", **fields), env=env or None)

    def session_end(self, reason: str = "other", **env) -> HookResult:
        return self.hook("session-summary", self.payload("SessionEnd", reason=reason), env=env or None)

    def pre_compact(self, trigger: str = "manual", **env) -> HookResult:
        return self.hook(
            "pre-compact", self.payload("PreCompact", trigger=trigger, custom_instructions=""), env=env or None
        )

    def pre_read(self, file_path: str | Path, **env) -> HookResult:
        return self.hook(
            "pre-tool-use",
            self.payload(
                "PreToolUse",
                tool_name="Read",
                tool_input={"file_path": str(file_path)},
                tool_use_id=f"toolu_{uuid.uuid4().hex[:12]}",
            ),
            env=env or None,
        )

    # ── the plugin's local state ──────────────────────────────
    @property
    def buffer_path(self) -> Path:
        return self.sandbox.observations / f"{self.session_id}.jsonl"

    def buffer_rows(self) -> list[dict]:
        if not self.buffer_path.is_file():
            return []
        return [json.loads(line) for line in self.buffer_path.read_text().splitlines() if line.strip()]

    @property
    def offset_path(self) -> Path:
        return Path(str(self.transcript.path) + ".neuralscape-offset")

    def offset(self) -> int | None:
        try:
            return int(self.offset_path.read_text().strip())
        except (OSError, ValueError):
            return None


# ── reading hook output ───────────────────────────────────────


def context_of(res: HookResult) -> str:
    """The additionalContext a hook injected ('' when it injected nothing)."""
    assert res.returncode == 0, f"hook exited {res.returncode}: {res.stderr[:200]}"
    out = (res.data or {}).get("hookSpecificOutput") or {}
    return out.get("additionalContext") or ""


def assert_continues(res: HookResult) -> None:
    assert res.returncode == 0, f"hook exited {res.returncode}: {res.stderr[:200]}"
    assert res.data is not None and res.data.get("continue") is True, f"hook did not answer continue: {res.stdout[:200]!r}"


# ── reading what reached the wire ─────────────────────────────


def flushes_with(proxy: RecordingProxy, marker: str) -> list[dict]:
    """Bodies of the /flush requests whose user or assistant text carries ``marker``."""
    out = []
    for call in proxy.calls(FLUSH, "POST"):
        body = call.body
        if marker in (body.get("user_message") or "") or marker in (body.get("assistant_response") or ""):
            out.append(body)
    return out


def check_sent_exactly_once(proxy: RecordingProxy, marker: str) -> None:
    n = len(flushes_with(proxy, marker))
    assert n == 1, f"{marker} was sent to /flush {n} times (want exactly 1)"


def everything_sent(proxy: RecordingProxy) -> str:
    """Every request line + body the hooks put on the wire, as one string (for 'never appears' checks)."""
    return "\n".join(f"{c.method} {c.path} {c.raw_body}" for c in proxy.requests)


# ── reading what the stack stored ─────────────────────────────


def task_completed(client: NSClient, task_id: str, timeout: float = 240.0) -> dict:
    """Poll a task the HOOK enqueued (its id comes from the proxy's record of the 202)."""
    assert task_id and not task_id.startswith("stub-"), f"not a real task id: {task_id!r}"
    status = client.poll_task(task_id, timeout=timeout)
    assert status.get("status") == "completed", f"task {task_id} ended as {status.get('status')!r}: {str(status.get('error'))[:200]}"
    return status


def memories_with(client: NSClient, project_id: str, marker: str) -> list[dict]:
    resp = client.get("/v1/memories", params={"project_id": project_id, "limit": 500})
    assert resp.status_code == 200, f"GET /v1/memories -> {resp.status_code}: {resp.text[:160]}"
    return [r for r in resp.json() if marker in (r.get("memory") or "")]


def check_recalled(client: NSClient, project_id: str, *markers: str) -> dict:
    """Some stored memory of this user carries one of ``markers`` AND /v1/search returns it for its own text."""
    row = None
    for marker in markers:
        found = memories_with(client, project_id, marker)
        if found:
            row = found[0]
            break
    assert row is not None, f"none of {list(markers)} found among {client.user_id}'s memories of {project_id}"
    hits = client.search(row["memory"], project_id=project_id)
    assert any(h.get("id") == row["id"] for h in hits), f"memory {row['id']} stored but not returned by /v1/search for its own text"
    return row


def check_never_stored(client: NSClient, project_id: str, marker: str) -> None:
    """The marker is in no memory of the project, by listing and by search."""
    assert not memories_with(client, project_id, marker), f"{marker} was stored"
    hits = client.search(marker, project_id=project_id)
    assert not any(marker in (h.get("memory") or "") for h in hits), f"{marker} returned by /v1/search"


def wait_recalled(client: NSClient, project_id: str, *markers: str, timeout: float = 150.0) -> dict:
    return eventually(lambda: check_recalled(client, project_id, *markers), timeout=timeout, interval=3, what=f"recall of {markers}")
