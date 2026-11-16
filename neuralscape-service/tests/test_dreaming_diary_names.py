"""Diary page names are collision-free per pool (one pool's text never lands in another's file)."""

from __future__ import annotations

import hashlib

import pytest

from extensions.dreaming import reflect
from okf import vault as okf_vault
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


def _legacy_path(dreams, pool):
    """The pre-fix slug-only name (the only name the old code ever wrote)."""
    from extensions.conversation_compiler.obsidian_writer import _slugify

    return dreams / f"{_slugify(pool) or 'pool'}.md"


def _make_legacy(dreams, pool, *, fm_pool=None, secret=None, raw_fm=None):
    """Write a pre-fix file for ``pool`` under the shared slug-only name."""
    dreams.mkdir(parents=True, exist_ok=True)
    path = _legacy_path(dreams, pool)
    secret = secret or f"LEGACY-SECRET-{pool!r}"
    fm = raw_fm if raw_fm is not None else (
        f"---\ntype: diary\ntitle: Dreams\npool: {fm_pool if fm_pool is not None else pool}\n---\n"
    )
    path.write_bytes((fm + f"\n# Dreams \u2014 {pool}\n\n## Dream 2026-01-01 (r)\n- {secret}\n").encode("utf-8"))
    return path, secret


# (legacy file's own pool id, pool that sweeps and shares the slug-only name)
LOOKALIKES = [
    ("user--a\n", "user--a"),
    ("user--a\r", "user--a"),
    ("user--a\u2028", "user--a"),
    ("user--a\x85", "user--a"),
    ("user--a'", "user--a"),
    ("user--p  q", "user--p q"),
    ("user--a_b", "user--a-b"),
]


@pytest.mark.parametrize("owner,sweeper", LOOKALIKES)
def test_legacy_file_is_never_moved_merged_or_rewritten(tmp_path, owner, sweeper):
    dreams = tmp_path / "Dreams"
    legacy, secret = _make_legacy(dreams, owner)
    assert _legacy_path(dreams, sweeper) == legacy
    before = legacy.read_bytes()
    _write(dreams, sweeper, "NEW-TEXT")
    assert legacy.read_bytes() == before
    page = reflect.diary_page_path(dreams, sweeper)
    assert page != legacy
    assert secret not in page.read_text() and "NEW-TEXT" in page.read_text()


@pytest.mark.parametrize(
    "fm_pool",
    ["user--a", "'user--a'", '"user--a"', "  user--a  ", "user--a ", ""],
)
def test_legacy_file_with_any_frontmatter_pool_stays_byte_identical(tmp_path, fm_pool):
    """Even frontmatter + H1 that name exactly the sweeping pool must not be moved."""
    dreams = tmp_path / "Dreams"
    pool = "user--a"
    legacy, secret = _make_legacy(dreams, pool, fm_pool=fm_pool)
    before = legacy.read_bytes()
    _write(dreams, pool, "NEW-TEXT")
    assert legacy.read_bytes() == before
    page = reflect.diary_page_path(dreams, pool).read_text()
    assert secret not in page and "NEW-TEXT" in page
    refresh_bundle_indexes(tmp_path)
    assert legacy.read_bytes() == before


def test_legacy_file_with_no_frontmatter_stays_byte_identical(tmp_path):
    dreams = tmp_path / "Dreams"
    legacy, secret = _make_legacy(dreams, "user--a", raw_fm="")
    before = legacy.read_bytes()
    _write(dreams, "user--a", "NEW-TEXT")
    assert legacy.read_bytes() == before
    assert secret not in reflect.diary_page_path(dreams, "user--a").read_text()


def test_write_diary_never_reads_legacy_content_into_pages(tmp_path):
    """Several pools sharing one pre-fix name: none of their pages gets that text."""
    dreams = tmp_path / "Dreams"
    legacy, secret = _make_legacy(dreams, A)
    for pool in (A, B, "user--a_b"):
        _write(dreams, pool, f"NEW-{pool}")
        assert secret not in reflect.diary_page_path(dreams, pool).read_text()
    assert secret in legacy.read_text()


def test_fresh_pool_page_has_no_carried_over_entries(tmp_path):
    dreams = tmp_path / "Dreams"
    _make_legacy(dreams, A)
    _write(dreams, A, "only-entry")
    text = reflect.diary_page_path(dreams, A).read_text()
    assert text.count("## Dream") == 1
    assert "pre-fix" not in text and "carried over" not in text
    assert "LEGACY-SECRET" not in text


def test_index_lists_each_legacy_file_once_under_heading(tmp_path):
    dreams = tmp_path / "Dreams"
    legacy_a, _ = _make_legacy(dreams, A)
    legacy_x, _ = _make_legacy(dreams, "user--x", fm_pool="whatever")
    _write(dreams, A, "x")
    _write(dreams, B, "y")
    refresh_bundle_indexes(tmp_path)
    index = (dreams / "index.md").read_text()
    heading = okf_vault.LEGACY_DIARY_HEADING
    assert heading == "Pre-fix diary files (may hold entries from pools whose names collided)"
    assert index.count(heading) == 1
    head_pos = index.index(heading)
    before_h, after_h = index[:head_pos], index[head_pos:]
    for legacy in (legacy_a, legacy_x):
        assert index.count(f"({legacy.name})") == 1
        assert f"({legacy.name})" in after_h and f"({legacy.name})" not in before_h
    for pool in (A, B):
        name = reflect.diary_page_path(dreams, pool).name
        assert index.count(f"({name})") == 1 and f"({name})" in before_h
    # idempotent
    refresh_bundle_indexes(tmp_path)
    assert (dreams / "index.md").read_text() == index


def test_index_without_legacy_files_has_no_prefix_heading(tmp_path):
    _write(tmp_path / "Dreams", A, "x")
    refresh_bundle_indexes(tmp_path)
    assert okf_vault.LEGACY_DIARY_HEADING not in (tmp_path / "Dreams" / "index.md").read_text()


def test_legacy_only_dreams_dir_still_gets_an_index(tmp_path):
    legacy, _ = _make_legacy(tmp_path / "Dreams", A)
    refresh_bundle_indexes(tmp_path)
    index = (tmp_path / "Dreams" / "index.md").read_text()
    assert okf_vault.LEGACY_DIARY_HEADING in index and legacy.name in index


def test_digest_pattern_matches_what_diary_page_path_writes():
    for pool in (A, B, "x", "user--Alice", LONG1, "user--bad\u2028", "user--" + "x" * 300):
        name = reflect.diary_page_path(reflect.Path("/v/Dreams"), pool).name
        assert okf_vault._DIARY_DIGEST_RE.search(name), name
        assert not okf_vault._DIARY_DIGEST_RE.search(_legacy_path(reflect.Path('/v/Dreams'), pool).name)


def test_unwritable_pool_ids_do_not_break_write(tmp_path):
    dreams = tmp_path / "Dreams"
    # lone surrogate cannot be utf-8 encoded: must return None, not raise
    assert _write(dreams, "user--bad\ud800", "x") is None
    # over-long ids: the slug is capped, so the page name stays within filesystem limits
    long_pool = "user--" + "x" * 300
    assert len(reflect.diary_page_path(dreams, long_pool).name.encode()) < 255
    assert _write(dreams, long_pool, "x") is not None
    assert hashlib.sha256(long_pool.encode()).hexdigest()[:16] in reflect.diary_page_path(dreams, long_pool).name
