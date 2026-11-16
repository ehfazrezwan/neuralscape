"""Repo-level migration scripts: retired wiki layout migration + group-id cypher.

``scripts/`` lives at the repo root; the packaged test image carries only the
service tree, so every test here returns early when it is absent (same
precedent as ``test_upstream_patch_markers.py``).
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WIKI_SCRIPT = REPO / "scripts" / "migrate_wiki_layout.py"
CYPHER = REPO / "scripts" / "migrate-group-ids.cypher"
GRAPHITI_NODE_QUERIES = REPO / "graphiti" / "graphiti_core" / "models" / "nodes" / "node_db_queries.py"


# ── migrate_wiki_layout.py (retired) ─────────────────────────────────────────


def _old_layout_vault(tmp_path: Path) -> Path:
    page = tmp_path / "Wiki" / "Project" / "Architecture" / "shared.md"
    page.parent.mkdir(parents=True)
    page.write_text("legacy page")
    return tmp_path


def _run_wiki(*args: str):
    return subprocess.run(
        [sys.executable, str(WIKI_SCRIPT), *args],
        capture_output=True, text=True, timeout=60,
    )


def test_wiki_migration_runs_without_importing_removed_package(tmp_path):
    """The script used to crash at import (extensions.wiki_synthesizer is gone)."""
    if not WIKI_SCRIPT.is_file():
        return
    vault = _old_layout_vault(tmp_path)
    proc = _run_wiki("--vault", str(vault), "--dry-run")
    assert "Traceback" not in proc.stderr
    assert "ModuleNotFoundError" not in proc.stderr


def test_wiki_migration_exits_with_retirement_message_and_touches_nothing(tmp_path):
    if not WIKI_SCRIPT.is_file():
        return
    vault = _old_layout_vault(tmp_path)
    before = sorted(p.relative_to(vault) for p in vault.rglob("*"))
    for argv in (["--vault", str(vault)], ["--vault", str(vault), "--dry-run"], []):
        proc = _run_wiki(*argv)
        assert proc.returncode == 2
        assert "retired" in proc.stderr
        assert "780b933a" in proc.stderr
        assert "migrate_wiki_to_topics.py" in proc.stderr
    assert sorted(p.relative_to(vault) for p in vault.rglob("*")) == before
    assert (vault / "Wiki" / "Project" / "Architecture" / "shared.md").read_text() == "legacy page"


def test_wiki_migration_successor_exists_and_removed_package_is_gone():
    if not WIKI_SCRIPT.is_file():
        return
    assert (REPO / "neuralscape-service" / "scripts" / "migrate_wiki_to_topics.py").is_file()
    assert not (REPO / "neuralscape-service" / "extensions" / "wiki_synthesizer").exists()


# ── migrate-group-ids.cypher ─────────────────────────────────────────────────


def _statements() -> list[str]:
    body = "\n".join(
        line for line in CYPHER.read_text().splitlines() if not line.lstrip().startswith("//")
    )
    return [s.strip() for s in body.split(";") if s.strip()]


def _migrations() -> list[str]:
    return [s for s in _statements() if "SET " in s]


def _rewrite(statement: str, group_id: str, target_prefix: str, label: str | None = None):
    """Simulate one migration statement's WHERE/SET on a single group_id string."""
    legacy = re.findall(r"STARTS WITH '([^']+)'", statement.split("SET ")[0])
    assert legacy, "statement has no legacy-prefix predicate"
    if not any(group_id.startswith(p) for p in legacy):
        return group_id
    sizes = re.findall(r"substring\(\w+\.group_id, size\('([^']+)'\)\)", statement)
    # CASE WHEN <starts with p> THEN substring(.., size(p)) ... ELSE substring(.., size(q))
    whens = re.findall(r"WHEN \w+\.group_id STARTS WITH '([^']+)' THEN", statement)
    assert "$target_prefix" in statement
    for p in whens:
        if group_id.startswith(p):
            return target_prefix + group_id[len(p):]
    # fall through: ELSE branch (last size prefix)
    return target_prefix + group_id[len(sizes[-1]):]


def test_cypher_targets_real_graphiti_labels():
    if not CYPHER.is_file():
        return
    text = "\n".join(_migrations())
    for stale in ("EpisodicNode", "EntityNode", "CommunityNode"):
        assert stale not in text, f"{stale} is not a Graphiti label"
    labels = set(re.findall(r"MATCH \(n:(\w+)\)", text))
    assert labels == {"Episodic", "Entity", "Community"}
    if GRAPHITI_NODE_QUERIES.is_file():
        queries = GRAPHITI_NODE_QUERIES.read_text()
        for label in labels:
            assert f"MERGE (n:{label} " in queries


def test_cypher_rewrite_matches_current_group_id_builder():
    if not CYPHER.is_file():
        return
    from memory.groups import _build_group_id

    migrations = _migrations()
    assert len(migrations) == 4  # Episodic, Entity, Community, relationships
    for pid in ("neuralscape", "my-app", "a:b"):
        shared_target = _build_group_id("shared", "u1", pid)[: -len(pid)]
        private_target = _build_group_id("private", "u1", pid)[: -len(pid)]
        assert shared_target == "shared--project--"
        assert private_target == "user--u1--project--"
        for legacy in (f"project:{pid}", f"project--{pid}"):
            for stmt in migrations:
                assert _rewrite(stmt, legacy, shared_target) == _build_group_id("shared", "u1", pid)
                assert _rewrite(stmt, legacy, private_target) == _build_group_id("private", "u1", pid)


def test_cypher_is_idempotent_and_leaves_current_ids_alone():
    if not CYPHER.is_file():
        return
    from memory.groups import _build_group_id

    current = [
        _build_group_id("shared", "u1", "p"),
        _build_group_id("private", "u1", "p"),
        _build_group_id("shared", "u1", "p", workspace="ref"),
        _build_group_id("standard", "u1", "p"),
        _build_group_id("shared", "u1"),
        _build_group_id("private", "u1"),
        "global",
    ]
    for stmt in _migrations():
        for gid in current:
            assert _rewrite(stmt, gid, "shared--project--") == gid
        migrated = _rewrite(stmt, "project:p", "shared--project--")
        assert _rewrite(stmt, migrated, "shared--project--") == migrated


def test_cypher_header_names_today_formats_and_the_parameter():
    if not CYPHER.is_file():
        return
    header = "\n".join(
        line for line in CYPHER.read_text().splitlines() if line.startswith("//")
    )
    assert "shared--project--{id}" in header
    assert "user--{user_id}--project--{id}" in header
    assert "target_prefix" in header
    assert "EpisodicNode" not in header
