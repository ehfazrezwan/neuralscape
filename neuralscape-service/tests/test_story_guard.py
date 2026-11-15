"""The integrated story skips only when its modules are absent; a present but broken module fails the run."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SERVICE_DIR.parent


# Loaded as a plugin from the scratch directory: the tests' conftest puts the service directory first on the
# import path, so the scratch directory is put back in front just before each module is collected (and the
# story is imported in append mode), and a planted module wins even when the service ships one of that name.
_SCRATCH_FIRST = """
import sys
from pathlib import Path

SCRATCH = str(Path(__file__).resolve().parent)
PLANTED = [path.stem for path in Path(SCRATCH).glob("*.py") if path.stem != "story_guard_scratch_first"]


def pytest_pycollect_makemodule(module_path, parent):
    if SCRATCH in sys.path:
        sys.path.remove(SCRATCH)
    sys.path.insert(0, SCRATCH)
    for name in PLANTED:
        sys.modules.pop(name, None)
"""


def _run_story(extra_path: Path | None) -> tuple[int, str]:
    env = dict(os.environ)
    command = [sys.executable, "-m", "pytest", "tests/test_integrated_reference_story.py", "-q", "-p", "no:cacheprovider", "-rs"]
    if extra_path is not None:
        (extra_path / "story_guard_scratch_first.py").write_text(_SCRATCH_FIRST)
        env["PYTHONPATH"] = str(extra_path) + os.pathsep + env.get("PYTHONPATH", "")
        command += ["-p", "story_guard_scratch_first", "--import-mode=append"]  # (prepend would put the service first again)
    done = subprocess.run(command, cwd=SERVICE_DIR, env=env, capture_output=True, text=True, timeout=300)
    return done.returncode, done.stdout + done.stderr


def test_a_present_but_broken_story_dependency_is_an_error_not_a_skip(tmp_path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # a module that exists (so it is not absent) but needs a file that did not ship
    (scratch / "contracts_policy_reference.py").write_text("import a_dependency_that_did_not_ship\n")
    code, output = _run_story(scratch)
    assert code != 0, output
    assert "error" in output.lower() and "skipped" not in output.split("\n")[-2].lower(), output[-600:]


def test_the_skip_names_exactly_the_absent_modules_when_none_is_present() -> None:
    import importlib.util

    required = (
        "contracts_lifecycle", "contracts_policy", "contracts_policy_reference", "contracts_freshness",
        "contracts_inference", "contracts_portability",
    )
    absent = [name for name in required if importlib.util.find_spec(name) is None]
    code, output = _run_story(None)
    if absent:
        assert code in (0, 5) and "absent module(s): " + ", ".join(absent) in output  # 5: nothing ran, the module was skipped, output[-600:]
    else:
        assert code == 0 and "passed" in output
