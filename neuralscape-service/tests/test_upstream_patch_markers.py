"""scripts/check-upstream-patches.sh: every vendored-subtree patch keeps its marker.

A ``git subtree pull`` can silently revert NeuralScape's patches to graphiti and
mem0. ``scripts/upstream-patches.txt`` lists the patched files; the check fails
loudly when one of them lost its ``NEURALSCAPE PATCH`` marker.

The packaged test image carries only the service tree (no repo-level
``scripts/``), so these tests return early there, as the CI skip gate's
manifest test does.
"""

import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check-upstream-patches.sh"
MANIFEST = REPO / "scripts" / "upstream-patches.txt"
MARKER = "NEURALSCAPE PATCH"


def _present() -> bool:
    return SCRIPT.is_file() and MANIFEST.is_file()


def _listed() -> list[str]:
    return [
        line.strip()
        for line in MANIFEST.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _run(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), str(root)], capture_output=True, text=True, timeout=60
    )


def test_repo_passes_the_marker_check():
    if not _present():
        return
    result = _run(REPO)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK:" in result.stdout


def test_manifest_covers_every_marked_file_in_the_subtrees():
    """A new patch must be added to the manifest or the next sync cannot protect it."""
    if not _present():
        return
    marked = set()
    for sub in ("graphiti", "mem0"):
        for path in (REPO / sub).rglob("*.py"):
            if MARKER in path.read_text(errors="ignore"):
                marked.add(path.relative_to(REPO).as_posix())
    assert marked - set(_listed()) == set(), "marked files missing from upstream-patches.txt"
    assert set(_listed()) - marked == set(), "manifest lists files with no marker"


def test_manifest_includes_the_known_patched_files():
    if not _present():
        return
    listed = set(_listed())
    for expected in (
        "graphiti/graphiti_core/driver/neo4j_driver.py",
        "graphiti/graphiti_core/llm_client/gemini_client.py",
        "mem0/mem0/embeddings/openai.py",
        "mem0/mem0/configs/embeddings/base.py",
    ):
        assert expected in listed


def test_check_fails_when_a_marker_is_removed(tmp_path):
    if not _present():
        return
    (tmp_path / "scripts").mkdir()
    shutil.copy(MANIFEST, tmp_path / "scripts" / "upstream-patches.txt")
    for rel in _listed():
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dest)
    assert _run(tmp_path).returncode == 0  # the copy is clean

    victim = tmp_path / "graphiti/graphiti_core/driver/neo4j_driver.py"
    victim.write_text(victim.read_text().replace(MARKER, "removed"))
    result = _run(tmp_path)
    assert result.returncode != 0
    assert "LOST MARKER" in result.stderr
    assert "neo4j_driver.py" in result.stderr


def test_check_fails_when_a_listed_file_is_missing(tmp_path):
    if not _present():
        return
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "upstream-patches.txt").write_text("graphiti/gone.py\n")
    result = _run(tmp_path)
    assert result.returncode != 0
    assert "MISSING FILE" in result.stderr
