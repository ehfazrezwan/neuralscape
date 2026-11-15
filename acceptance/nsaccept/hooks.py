"""Run the plugin's BUILT hook scripts exactly as Claude Code does."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# acceptance/nsaccept/hooks.py -> repo root is two levels above acceptance/
REPO_ROOT = Path(os.environ.get("NS_REPO_ROOT") or Path(__file__).resolve().parents[2])
DEFAULT_SCRIPTS_DIR = REPO_ROOT / "neuralscape-plugin" / "scripts"


class HookSandbox:
    """A throwaway CLAUDE_PLUGIN_DATA / HOME / work directory that outlives one hook run.

    ``run_hook`` without a sandbox gives every call its own; hooks that share
    state on disk (observation buffers, cursors, snapshots) need the SAME
    sandbox across calls. ``work`` is the hook's cwd (a Claude Code project
    directory). Use as a context manager or call ``close()``.
    """

    def __init__(self, work_name: str = "work") -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ns-accept-hook-")
        root = Path(self._tmp.name)
        self.data_dir = root / "plugin-data"
        self.home = root / "home"
        self.work = root / work_name
        for d in (self.data_dir, self.home, self.work):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def observations(self) -> Path:
        return self.data_dir / "observations"

    def close(self) -> None:
        self._tmp.cleanup()

    def __enter__(self) -> "HookSandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class HookResult:
    returncode: int
    stdout: str
    stderr: str
    data: dict | None = field(default=None)


def run_hook(
    name: str,
    payload: dict,
    *,
    base_url: str,
    token: str | None,
    user_id: str,
    project_id: str | None = None,
    extra_env: dict[str, str] | None = None,
    scripts_dir: Path | None = None,
    timeout: float = 60.0,
    sandbox: HookSandbox | None = None,
    raw_stdin: str | None = None,
) -> HookResult:
    """``node <scripts>/<name>.js`` with ``payload`` on stdin.

    The hook sees only plugin option env vars (URL, API_KEY = the bearer
    token, USER_ID, PROJECT_ID) and a throwaway CLAUDE_PLUGIN_DATA / HOME, so
    it can neither read the developer's real plugin config nor leave buffers
    behind.

    ``sandbox``: reuse one directory set across calls (default: a fresh one per
    call, removed afterwards). ``raw_stdin``: send this text instead of the
    JSON payload (malformed-input cases).
    """
    script = (scripts_dir or DEFAULT_SCRIPTS_DIR) / f"{name}.js"
    if not script.is_file():
        raise FileNotFoundError(f"hook script not found: {script}")
    owned = sandbox is None
    box = sandbox or HookSandbox()
    try:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(box.home),
            "CLAUDE_PLUGIN_DATA": str(box.data_dir),
            "CLAUDE_PLUGIN_OPTION_URL": base_url,
            "CLAUDE_PLUGIN_OPTION_USER_ID": user_id,
        }
        if token:
            env["CLAUDE_PLUGIN_OPTION_API_KEY"] = token
        if project_id:
            env["CLAUDE_PLUGIN_OPTION_PROJECT_ID"] = project_id
        env.update(extra_env or {})
        payload = {"cwd": str(box.work), **payload}
        proc = subprocess.run(
            ["node", str(script)],
            input=json.dumps(payload) if raw_stdin is None else raw_stdin,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(box.work),
            timeout=timeout,
        )
    finally:
        if owned:
            box.close()
    result = HookResult(proc.returncode, proc.stdout, proc.stderr)
    try:
        parsed = json.loads(proc.stdout) if proc.stdout.strip() else None
        result.data = parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        result.data = None
    return result
