"""Every tracked top-level directory under neuralscape-service/ reaches the image stages that need it.

The Dockerfile enumerates its COPY paths per directory, so a directory that is
not listed is silently absent from the image: host tests pass and the image
crash-loops, or an integration test that reads a file from the tree errors at
setup inside the packaged image (``supabase/allowlist_hook.sql`` was missing
from every stage until this test existed).

Repo rule (CLAUDE.md, Container Build Gate): a new top-level directory goes into
all three stages (builder, runtime, test). The few directories that
deliberately do not are listed in ``EXPECTED_STAGES`` with the reason.

Only the service tree exists in the packaged test image (no Dockerfile, no git
checkout), so the test returns early there; the host CI job runs it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]
DOCKERFILE = SERVICE_DIR / "Dockerfile"
ALL_STAGES = frozenset({"builder", "runtime", "test"})

# Directories that deliberately skip a stage, and why.
EXPECTED_STAGES: dict[str, frozenset[str]] = {
    "docs": frozenset(),  # documentation only; never read by code or tests in the image
    "tests": frozenset({"test"}),  # the unit suite runs only in the test stage
    "scripts": frozenset({"runtime", "test"}),  # operator scripts; nothing is built from them
    "vendor": frozenset({"builder", "test"}),  # wheels installed into the venv the runtime copies
}


def _tracked_top_level_dirs() -> set[str] | None:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", "."],
            cwd=SERVICE_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {line.split("/", 1)[0] for line in out.splitlines() if "/" in line}


def _stages_copying() -> dict[str, set[str]]:
    copied: dict[str, set[str]] = {}
    stage = None
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^FROM\s+\S+\s+AS\s+(\S+)", line)
        if match:
            stage = match.group(1)
            continue
        match = re.match(r"^COPY\s+(?:--\S+\s+)*neuralscape-service/([^/\s*]+)/\s", line)
        if match and stage in ALL_STAGES:
            copied.setdefault(match.group(1), set()).add(stage)
    return copied


def test_every_tracked_top_level_directory_is_copied_into_the_stages_it_needs():
    if not DOCKERFILE.exists():
        return  # packaged image: no Dockerfile here
    dirs = _tracked_top_level_dirs()
    if not dirs:
        return  # no git checkout
    copied = _stages_copying()
    problems = []
    for name in sorted(dirs):
        expected = EXPECTED_STAGES.get(name, ALL_STAGES)
        actual = frozenset(copied.get(name, set()))
        if actual != expected:
            problems.append(
                f"neuralscape-service/{name}/: copied into {sorted(actual) or 'no stage'}, "
                f"expected {sorted(expected) or 'no stage'}"
            )
    assert not problems, "Dockerfile COPY lists out of step with the tree:\n" + "\n".join(problems)


def test_supabase_hook_sql_reaches_the_test_image():
    """integration_tests/test_supabase_allowlist_hook_pg.py reads this file inside the test image."""
    if not DOCKERFILE.exists():
        return
    assert (SERVICE_DIR / "supabase" / "allowlist_hook.sql").is_file()
    assert "test" in _stages_copying().get("supabase", set())
