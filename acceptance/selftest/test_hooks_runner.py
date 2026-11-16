import json
import shutil

import pytest

from nsaccept.hooks import run_hook

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")

FAKE = """
let raw = '';
process.stdin.on('data', d => raw += d);
process.stdin.on('end', () => {
  const env = {};
  for (const k of Object.keys(process.env)) if (k.startsWith('CLAUDE_PLUGIN_')) env[k] = process.env[k];
  process.stdout.write(JSON.stringify({ stdin: JSON.parse(raw), env, home: process.env.HOME }));
  process.exit(Number(process.env.FAKE_EXIT || 0));
});
"""


def test_run_hook_passes_stdin_and_plugin_env(tmp_path):
    (tmp_path / "fake.js").write_text(FAKE)
    res = run_hook(
        "fake", {"hook_event_name": "SessionStart", "source": "startup"},
        base_url="http://127.0.0.1:9", token="a.b", user_id="u1", project_id="p1",
        scripts_dir=tmp_path,
    )
    assert res.returncode == 0 and res.data is not None
    env = res.data["env"]
    assert env["CLAUDE_PLUGIN_OPTION_URL"] == "http://127.0.0.1:9"
    assert env["CLAUDE_PLUGIN_OPTION_API_KEY"] == "a.b"
    assert env["CLAUDE_PLUGIN_OPTION_USER_ID"] == "u1"
    assert env["CLAUDE_PLUGIN_OPTION_PROJECT_ID"] == "p1"
    assert env["CLAUDE_PLUGIN_DATA"].startswith("/") and "plugin-data" in env["CLAUDE_PLUGIN_DATA"]
    assert res.data["stdin"]["hook_event_name"] == "SessionStart" and res.data["stdin"]["cwd"]
    assert res.data["home"] != __import__("os").environ.get("HOME")


def test_run_hook_reports_exit_code_and_non_json(tmp_path):
    (tmp_path / "bad.js").write_text("process.stdout.write('not json'); process.exit(2);")
    res = run_hook("bad", {}, base_url="http://127.0.0.1:9", token=None, user_id="u", scripts_dir=tmp_path)
    assert res.returncode == 2 and res.data is None and res.stdout == "not json"


def test_missing_script_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_hook("nope", {}, base_url="http://127.0.0.1:9", token=None, user_id="u", scripts_dir=tmp_path)
