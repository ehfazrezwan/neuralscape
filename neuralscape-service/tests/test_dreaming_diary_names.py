"""Diary page names are collision-free per pool (one pool's text never lands in another's file)."""

from __future__ import annotations

from extensions.dreaming import reflect
from okf.vault import refresh_bundle_indexes

A, B = "user--a_b", "user--a-b"               # collapse to one slug under the old _slugify
LONG1 = "shared--project--" + "x" * 70 + "-one"   # share an 80-char slug prefix
LONG2 = "shared--project--" + "x" * 70 + "-two"


def _write(dreams, pool, text):
    entry = reflect.render_diary_entry(
        pool=pool, run_id="r", applied=[], reported=[],
        insights=[{"lens": "p", "content": text, "source_memory_ids": ["m"]}],
    )
    return reflect.write_diary(dreams, pool, entry, source_memory_ids=["m"])


def test_colliding_pool_ids_get_distinct_files_and_text_stays_apart(tmp_path):
    dreams = tmp_path / "Dreams"
    for first, second in ((A, B), (LONG1, LONG2)):
        assert reflect.diary_page_path(dreams, first) != reflect.diary_page_path(dreams, second)
        _write(dreams, first, f"SECRET-{first}")
        _write(dreams, second, f"SECRET-{second}")
        t1 = reflect.diary_page_path(dreams, first).read_text()
        t2 = reflect.diary_page_path(dreams, second).read_text()
        assert f"SECRET-{first}" in t1 and f"SECRET-{second}" not in t1
        assert f"SECRET-{second}" in t2 and f"SECRET-{first}" not in t2


def test_case_differing_ids_get_distinct_files(tmp_path):
    """The hash must not fold case: distinct on case-insensitive filesystems too."""
    dreams = tmp_path / "Dreams"
    upper, lower = "user--Alice", "user--alice"
    pu, pl = reflect.diary_page_path(dreams, upper), reflect.diary_page_path(dreams, lower)
    assert pu != pl and pu.name.lower() != pl.name.lower()
    _write(dreams, upper, "SECRET-upper")
    _write(dreams, lower, "SECRET-lower")
    assert "SECRET-lower" not in pu.read_text() and "SECRET-upper" not in pl.read_text()


def test_name_is_stable_and_readable():
    p1 = reflect.diary_page_path(reflect.Path("/v/Dreams"), A)
    assert p1 == reflect.diary_page_path(reflect.Path("/v/Dreams"), A)
    assert p1.name.startswith("user-a-b-") and p1.name.endswith(".md")
    assert p1.name == "user-a-b-" + p1.name.split("-")[-1]  # slug + 16-hex digest
    assert len(p1.stem.rsplit("-", 1)[1]) == 16


def test_index_links_resolve(tmp_path):
    for pool in (A, B):
        _write(tmp_path / "Dreams", pool, "x")
    refresh_bundle_indexes(tmp_path)
    index = (tmp_path / "Dreams" / "index.md").read_text()
    for pool in (A, B):
        assert reflect.diary_page_path(tmp_path / "Dreams", pool).name in index
    import re

    for href in re.findall(r"\]\(([^)]+\.md)\)", index):
        if href != "index.md":
            assert (tmp_path / "Dreams" / href).exists(), href


def test_legacy_diary_of_this_pool_is_migrated_not_duplicated(tmp_path):
    dreams = tmp_path / "Dreams"
    _write(dreams, A, "old text")
    new = reflect.diary_page_path(dreams, A)
    legacy = reflect.legacy_diary_page_path(dreams, A)
    new.replace(legacy)                       # simulate a file written before the fix
    _write(dreams, A, "new text")
    assert new.exists() and not legacy.exists()
    text = new.read_text()
    assert "old text" in text and "new text" in text


def _migrated_page(dreams):
    _write(dreams, A, "old text")
    new = reflect.diary_page_path(dreams, A)
    legacy = reflect.legacy_diary_page_path(dreams, A)
    new.replace(legacy)
    return new, legacy


def test_migrated_page_carries_one_note_naming_old_file(tmp_path):
    dreams = tmp_path / "Dreams"
    new, legacy = _migrated_page(dreams)
    _write(dreams, A, "new text")
    text = new.read_text()
    lines = [ln for ln in text.splitlines() if "carried over from the pre-fix" in ln]
    assert len(lines) == 1
    assert f"`{legacy.name}`" in lines[0] and "collided" in lines[0]
    assert text.index(lines[0]) < text.index("## Dream")
    assert "old text" in text and "new text" in text


def test_fresh_page_has_no_migration_note(tmp_path):
    dreams = tmp_path / "Dreams"
    _write(dreams, A, "fresh")
    assert "pre-fix" not in reflect.diary_page_path(dreams, A).read_text()


def test_migration_note_not_repeated_on_later_writes(tmp_path):
    dreams = tmp_path / "Dreams"
    new, _ = _migrated_page(dreams)
    _write(dreams, A, "second")
    _write(dreams, A, "third")
    text = new.read_text()
    assert text.count("carried over from the pre-fix") == 1
    assert "third" in text and "old text" in text


def test_legacy_file_of_another_pool_never_receives_text(tmp_path):
    dreams = tmp_path / "Dreams"
    _write(dreams, B, "B-text")
    legacy = reflect.legacy_diary_page_path(dreams, A)      # same old name as B's
    reflect.diary_page_path(dreams, B).replace(legacy)      # B's page sits under the shared old name
    before = legacy.read_text()
    _write(dreams, A, "A-text")
    assert legacy.read_text() == before and "A-text" not in before
    assert "A-text" in reflect.diary_page_path(dreams, A).read_text()
    assert "B-text" not in reflect.diary_page_path(dreams, A).read_text()


def _strand_legacy(dreams, owner):
    """Write ``owner``'s diary, then park it under the shared pre-fix name."""
    _write(dreams, owner, f"PRIVATE-{owner}")
    legacy = reflect.legacy_diary_page_path(dreams, owner)
    reflect.diary_page_path(dreams, owner).replace(legacy)
    return legacy


def test_legacy_quote_lookalike_pool_is_not_migrated_into_other_pool(tmp_path):
    dreams = tmp_path / "Dreams"
    x, y = "user--a'", "user--a"
    legacy = _strand_legacy(dreams, x)
    assert reflect.legacy_diary_page_path(dreams, y) == legacy
    before = legacy.read_text()
    _write(dreams, y, "Y-text")
    assert legacy.read_text() == before
    assert "PRIVATE" not in reflect.diary_page_path(dreams, y).read_text()


def test_legacy_whitespace_lookalike_pool_is_not_migrated_into_other_pool(tmp_path):
    dreams = tmp_path / "Dreams"
    x, y = "user--p  q", "user--p q"
    legacy = _strand_legacy(dreams, x)
    assert reflect.legacy_diary_page_path(dreams, y) == legacy
    before = legacy.read_text()
    _write(dreams, y, "Y-text")
    assert legacy.read_text() == before
    assert "PRIVATE" not in reflect.diary_page_path(dreams, y).read_text()


def test_unmigratable_legacy_does_not_break_write(tmp_path):
    dreams = tmp_path / "Dreams"
    # lone surrogate cannot be utf-8 encoded: write_diary must return None, not raise
    assert _write(dreams, "user--bad\ud800", "x") is None
    # legacy name over 255 bytes makes exists() raise ENAMETOOLONG on Linux; must not raise
    long_pool = "user--" + "x" * 300
    _write(dreams, long_pool, "x")  # must not raise
