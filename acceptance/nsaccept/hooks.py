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
) -> HookResult:
    """``node <scripts>/<name>.js`` with ``payload`` on stdin.

    The hook sees only plugin option env vars (URL, API_KEY = the bearer
    token, USER_ID, PROJECT_ID) and a throwaway CLAUDE_PLUGIN_DATA / HOME, so
    it can neither read the developer's real plugin config nor leave buffers
    behind.
    """
    script = (scripts_dir or DEFAULT_SCRIPTS_DIR) / f"{name}.js"
    if not script.is_file():
        raise FileNotFoundError(f"hook script not found: {script}")
    with tempfile.TemporaryDirectory(prefix="ns-accept-hook-") as tmp:
        data_dir = Path(tmp) / "plugin-data"
        home = Path(tmp) / "home"
        work = Path(tmp) / "work"
        for d in (data_dir, home, work):
            d.mkdir()
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(home),
            "CLAUDE_PLUGIN_DATA": str(data_dir),
            "CLAUDE_PLUGIN_OPTION_URL": base_url,
            "CLAUDE_PLUGIN_OPTION_USER_ID": user_id,
        }
        if token:
            env["CLAUDE_PLUGIN_OPTION_API_KEY"] = token
        if project_id:
            env["CLAUDE_PLUGIN_OPTION_PROJECT_ID"] = project_id
        env.update(extra_env or {})
        payload = {"cwd": str(work), **payload}
        proc = subprocess.run(
            ["node", str(script)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=env,
            cwd=str(work),
            timeout=timeout,
        )
    result = HookResult(proc.returncode, proc.stdout, proc.stderr)
    try:
        parsed = json.loads(proc.stdout) if proc.stdout.strip() else None
        result.data = parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        result.data = None
    return result
