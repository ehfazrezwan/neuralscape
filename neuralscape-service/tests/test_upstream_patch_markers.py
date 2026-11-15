"""scripts/check-upstream-patches.sh: every vendored-subtree patch keeps its marker.

A ``git subtree pull`` can silently revert NeuralScape's patches to graphiti and
mem0. ``scripts/upstream-patches.txt`` lists the patched files; the check fails
loudly when one of them lost its ``NEURALSCAPE PATCH`` marker.

The git-derived coverage tests (``graphiti-base`` / ``mem0-base`` diffs) need
the base commits, i.e. a full clone; on a shallow checkout (CI's default
``fetch-depth: 1``) they return early, so there they verify nothing. The marker
and manifest-consistency tests do run everywhere the repo-level files exist.

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


def _entries() -> list[tuple[str, str]]:
    out = []
    for line in MANIFEST.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            kind, path = line.split(" ", 1)
            out.append((kind, path.strip()))
    return out


def _listed(kind: str | None = None) -> list[str]:
    return [p for k, p in _entries() if kind is None or k == kind]


def _base(name: str) -> str:
    for line in MANIFEST.read_text().splitlines():
        if line.startswith(f"#   {name}:"):
            return line.split(":", 1)[1].split()[0]
    raise AssertionError(f"{name} missing from manifest header")


def _git(*args: str) -> list[str] | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.split() if r.returncode == 0 else None


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


def test_marked_and_graphiti_mentioning_mem0_files_are_in_the_manifest():
    """A new patch must be listed or the next sync cannot protect it."""
    if not _present():
        return
    listed = set(_listed())
    found = set()
    for sub in ("graphiti", "mem0"):
        for path in (REPO / sub).rglob("*.py"):
            text = path.read_text(errors="ignore")
            if MARKER in text or (sub == "mem0" and "GraphStoreFactory" in text):
                found.add(path.relative_to(REPO).as_posix())
    assert found - listed == set(), "marked files missing from upstream-patches.txt"
    assert set(_listed("patched")) <= found, "patched entry without a marker"


def test_graphiti_patched_set_equals_diff_against_upstream_base():
    """Derived from git, so an unmarked NS edit to graphiti is caught."""
    if not _present():
        return
    base = _base("graphiti-base")
    diff = _git("diff", "--name-only", f"{base}:graphiti_core", "HEAD:graphiti/graphiti_core")
    if diff is None:  # shallow clone / base object unavailable
        return
    expected = {f"graphiti/graphiti_core/{f}" for f in diff}
    listed = {p for p in _listed("patched") if p.startswith("graphiti/")}
    assert listed == expected, (
        f"graphiti patched list drifted from the upstream diff: "
        f"unlisted={sorted(expected - listed)} stale={sorted(listed - expected)}"
    )


def test_mem0_files_changed_since_last_sync_are_in_the_manifest():
    if not _present():
        return
    base = _base("mem0-base")
    changed = _git("diff", "--name-only", base, "HEAD", "--", "mem0/mem0")
    if changed is None:
        return
    assert set(changed) - set(_listed()) == set(), "NS-edited mem0 files not in the manifest"


def test_owned_entries_are_not_also_patched():
    if not _present():
        return
    assert not set(_listed("owned")) & set(_listed("patched"))


def test_manifest_includes_the_known_patched_files():
    if not _present():
        return
    listed = set(_listed())
    for expected in (
        "graphiti/graphiti_core/driver/neo4j_driver.py",
        "graphiti/graphiti_core/llm_client/gemini_client.py",
        "mem0/mem0/embeddings/openai.py",
        "mem0/mem0/configs/embeddings/base.py",
        "mem0/mem0/utils/factory.py",
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


def test_removed_marker_is_ok_for_owned_but_missing_owned_file_fails(tmp_path):
    if not _present():
        return
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "upstream-patches.txt").write_text("owned mem0/gone.py\n")
    result = _run(tmp_path)
    assert result.returncode != 0 and "MISSING FILE" in result.stderr


def test_check_fails_when_a_listed_file_is_missing(tmp_path):
    if not _present():
        return
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "upstream-patches.txt").write_text("patched graphiti/gone.py\n")
    result = _run(tmp_path)
    assert result.returncode != 0
    assert "MISSING FILE" in result.stderr
