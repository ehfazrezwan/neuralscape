"""Tests for the relocate_vault_to_raw.py migration script.

Smoke-test the forward, dry-run, and reverse paths against a temp vault.
"""

from pathlib import Path

import pytest

from scripts.relocate_vault_to_raw import RAW_DIRNAME, execute, plan_moves


@pytest.fixture
def populated_vault(tmp_path) -> Path:
    """A vault with one folder and one file at the top level."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Daily").mkdir()
    (vault / "Daily" / "2024-01-15.md").write_text("entry")
    (vault / "index.md").write_text("# Index")
    return vault


def test_plan_forward_finds_known_targets(populated_vault):
    pairs = plan_moves(populated_vault, reverse=False)
    sources = {src.name for src, _ in pairs}
    assert "Daily" in sources
    assert "index.md" in sources
    # All destinations live under _raw/
    for _, dst in pairs:
        assert dst.parent.name == RAW_DIRNAME or dst.parent.parent.name == RAW_DIRNAME


def test_dry_run_does_not_move(populated_vault):
    pairs = plan_moves(populated_vault, reverse=False)
    execute(pairs, dry_run=True)
    # Source still present, destination still absent
    assert (populated_vault / "Daily" / "2024-01-15.md").exists()
    assert not (populated_vault / RAW_DIRNAME).exists()


def test_apply_moves_into_raw(populated_vault):
    pairs = plan_moves(populated_vault, reverse=False)
    execute(pairs, dry_run=False)
    assert (populated_vault / RAW_DIRNAME / "Daily" / "2024-01-15.md").exists()
    assert (populated_vault / RAW_DIRNAME / "index.md").exists()
    assert not (populated_vault / "Daily").exists()
    assert not (populated_vault / "index.md").exists()


def test_reverse_restores_original_layout(populated_vault):
    # Forward
    execute(plan_moves(populated_vault, reverse=False), dry_run=False)
    # Reverse
    execute(plan_moves(populated_vault, reverse=True), dry_run=False)
    assert (populated_vault / "Daily" / "2024-01-15.md").exists()
    assert (populated_vault / "index.md").exists()
    # _raw/ may still exist as an empty directory; the entries we moved are gone.
    raw = populated_vault / RAW_DIRNAME
    if raw.exists():
        assert not (raw / "Daily").exists()
        assert not (raw / "index.md").exists()


def test_idempotent_forward(populated_vault):
    execute(plan_moves(populated_vault, reverse=False), dry_run=False)
    # Re-plan after migration: nothing left to move
    second_pass = plan_moves(populated_vault, reverse=False)
    assert second_pass == []


def test_never_overwrites_existing_destination(populated_vault):
    # Pre-create a colliding destination so the script must skip the move.
    (populated_vault / RAW_DIRNAME).mkdir()
    (populated_vault / RAW_DIRNAME / "Daily").mkdir()
    (populated_vault / RAW_DIRNAME / "Daily" / "different.md").write_text("preexisting")

    pairs = plan_moves(populated_vault, reverse=False)
    # The conflicting Daily folder is dropped from the plan.
    assert all(src.name != "Daily" for src, _ in pairs)
    # Original Daily is left alone, preexisting destination intact.
    assert (populated_vault / "Daily" / "2024-01-15.md").exists()
    assert (populated_vault / RAW_DIRNAME / "Daily" / "different.md").read_text() == "preexisting"
