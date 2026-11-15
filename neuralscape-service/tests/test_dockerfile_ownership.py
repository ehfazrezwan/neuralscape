"""The runtime and test image stages assign ownership at COPY time.

A trailing ``chown -R appuser:appuser /data /app`` rewrites every file already
in the image into a new layer, roughly doubling the runtime image. Creating the
user first and copying with ``--chown`` avoids the duplicate layer.

The Dockerfile is copied into the test stage only (the runtime image does not
carry it), so this file is read through a helper that fails loudly, never
skips, when it is absent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"
OWNER = "--chown=appuser:appuser"
STAGES = ("runtime", "test")


def _read_dockerfile() -> str:
    if not DOCKERFILE.is_file():
        pytest.fail(
            f"{DOCKERFILE} is missing. The test stage of the Dockerfile must "
            "COPY neuralscape-service/Dockerfile so this test can read it."
        )
    return DOCKERFILE.read_text()


def _stages() -> dict[str, list[str]]:
    """Stage name -> instruction lines (continuations joined, comments dropped)."""
    stages: dict[str, list[str]] = {}
    current: list[str] | None = None
    buf = ""
    for raw in _read_dockerfile().splitlines():
        line = raw.strip()
        if not buf and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        instr = (buf + line).strip()
        buf = ""
        m = re.match(r"^FROM .* AS (\w+)$", instr, re.IGNORECASE)
        if m:
            current = stages.setdefault(m.group(1), [])
            continue
        if current is not None:
            current.append(instr)
    return stages


def _stage(name: str) -> list[str]:
    stages = _stages()
    assert name in stages, f"stage {name!r} not found in {DOCKERFILE}"
    return stages[name]


def _is_copy(instr: str) -> bool:
    return instr.upper().startswith("COPY ")


def _is_uv_binary_copy(instr: str) -> bool:
    return _is_copy(instr) and "--from=ghcr.io/astral-sh/uv" in instr


def _copy_dests(instrs: list[str]) -> set[str]:
    dests = set()
    for i in instrs:
        if _is_copy(i) and "--from=" not in i:
            args = [a for a in i.split()[1:] if not a.startswith("--")]
            dests.add(args[-1].rstrip("/"))
    return dests


@pytest.mark.parametrize("stage", STAGES)
def test_no_recursive_chown(stage):
    for instr in _stage(stage):
        assert not re.search(r"chown\s+(-\w*R\w*|--recursive)\b", instr), instr


@pytest.mark.parametrize("stage", STAGES)
def test_user_created_before_first_copy(stage):
    instrs = _stage(stage)
    useradd = next((n for n, i in enumerate(instrs) if "useradd" in i), None)
    assert useradd is not None, f"{stage}: no useradd"
    first_copy = next(
        (n for n, i in enumerate(instrs) if _is_copy(i) and not _is_uv_binary_copy(i)),
        None,
    )
    assert first_copy is not None
    assert useradd < first_copy, f"{stage}: useradd must precede the first COPY"


@pytest.mark.parametrize("stage", STAGES)
def test_every_copy_sets_ownership(stage):
    for instr in _stage(stage):
        if _is_copy(instr) and not _is_uv_binary_copy(instr):
            assert OWNER in instr, f"{stage}: COPY without {OWNER}: {instr}"


@pytest.mark.parametrize("stage", STAGES)
def test_runs_as_appuser(stage):
    instrs = _stage(stage)
    user = next((n for n, i in enumerate(instrs) if i == "USER appuser"), None)
    assert user is not None, f"{stage}: USER appuser missing"
    if stage == "test":
        sync = next(
            (n for n, i in enumerate(instrs) if i.startswith("RUN") and "uv sync" in i),
            None,
        )
        assert sync is not None
        assert user < sync, "test: USER appuser must precede uv sync"


def test_runtime_copy_destinations_mirrored_in_test_stage():
    # Modest check: source packages the runtime serves are also present in the
    # test stage (which additionally has tests/, vendor/ and the Dockerfile).
    # Copies from another stage (the builder's venv) are not mirrored: the test
    # stage builds its own venv with dev dependencies.
    runtime = _copy_dests(_stage("runtime"))
    test = _copy_dests(_stage("test"))
    core = {d for d in runtime if not d.endswith("/scripts")}
    assert core <= test | {"graphiti", "mem0"}, sorted(core - test)
