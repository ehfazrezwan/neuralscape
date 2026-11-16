"""Diary page names are collision-free per pool (one pool's text never lands in another's file)."""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata

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
    digest = hashlib.sha256(A.encode("utf-8")).hexdigest()[:32]
    assert p1.name == f"user-a-b--{digest}.md"  # slug + "--" + 32-hex digest of the exact id


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
    assert hashlib.sha256(long_pool.encode()).hexdigest()[:32] in reflect.diary_page_path(dreams, long_pool).name


# ── cross-pool path without a hash collision: a pre-fix file named like a new page ──


def _sha(pool, n=32):
    return hashlib.sha256(pool.encode("utf-8")).hexdigest()[:n]


V = "shared--project--alpha"
O = V + "-" + _sha(V, 16)  # the pre-fix 16-hex name of V's page, as another pool's id


def test_prefix_file_named_like_another_pools_page_is_not_read_or_overwritten(tmp_path):
    """The reviewer's repro. O's pre-fix file is ``slug(V)-<16 hex of V>.md``,
    which the previous scheme used as V's page. It must stay untouched and
    its text must never reach V's page (nor V's text O's)."""
    assert O == "shared--project--alpha-c3df39628649efbe"
    dreams = tmp_path / "Dreams"
    legacy, secret = _make_legacy(dreams, O)
    assert legacy.name == "shared-project-alpha-c3df39628649efbe.md"
    before = legacy.read_bytes()
    _write(dreams, V, "V-ENTRY")
    _write(dreams, O, "O-ENTRY")
    _write(dreams, V, "V-ENTRY-2")
    pv = reflect.diary_page_path(dreams, V)
    po = reflect.diary_page_path(dreams, O)
    assert len({pv, po, legacy}) == 3
    assert legacy.read_bytes() == before
    tv, to = pv.read_text(), po.read_text()
    assert secret not in tv and secret not in to
    assert "V-ENTRY" in tv and "O-ENTRY" not in tv
    assert "O-ENTRY" in to and "V-ENTRY" not in to
    refresh_bundle_indexes(tmp_path)
    assert legacy.read_bytes() == before


def _legacy_names(pool):
    """Every name an earlier version of this code could have written for ``pool``."""
    from extensions.conversation_compiler.obsidian_writer import _slugify

    slug = _slugify(pool) or "pool"
    cap = slug[:48].strip("-") or "pool"
    return {
        f"{slug}.md",
        f"{cap}-{_sha(pool, 10)}.md",  # intermediate 10-hex form
        f"{cap}-{_sha(pool, 16)}.md",  # intermediate 16-hex form
    }


def _adversarial_pools():
    rng = random.Random(611)
    alphabet = "ab-_ .:A0\u00e9\u2028"
    pools = {V, O, A, B, LONG1, LONG2, "", "-", "--", "x"}
    for _ in range(150):
        pools.add("user--" + "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40))))
    for base in list(pools):
        for n in (10, 16, 32, 64):
            h = _sha(base, n)
            pools.update({f"{base}-{h}", f"{base}--{h}", f"{base}_{h}", f"{base} {h}", h, f"-{h}", f"--{h}"})
    return sorted(pools)


def test_legacy_style_names_never_equal_new_style_names(tmp_path):
    dreams = tmp_path / "Dreams"
    pools = _adversarial_pools()
    assert len(pools) > 1000
    legacy_names, new_names = set(), set()
    for pool in pools:
        legacy_names |= _legacy_names(pool)
        new_names.add(reflect.diary_page_path(dreams, pool).name)
    assert not (legacy_names & new_names)
    # the generator really produces the dangerous shape: some pool's pre-fix
    # name carries another pool's exact old-scheme digest suffix
    assert _legacy_names(O) & {f"shared-project-alpha-{_sha(V, 16)}.md"}
    # distinct ids -> distinct pages
    assert len(new_names) == len(pools)


def test_classifier_puts_every_legacy_name_prefix_and_every_new_name_new(tmp_path):
    pools = _adversarial_pools()
    new_names = {reflect.diary_page_path(tmp_path, p).name for p in pools}
    legacy_names = set().union(*(_legacy_names(p) for p in pools))
    for name in new_names:
        assert okf_vault._DIARY_DIGEST_RE.search(name), name
    for name in legacy_names:
        assert not okf_vault._DIARY_DIGEST_RE.search(name), name
    # and through the real index builder, on a sample of files
    sample_new = sorted(new_names)[:60]
    sample_legacy = sorted(legacy_names)[:60]
    dreams = tmp_path / "Dreams"
    dreams.mkdir()
    for name in sample_new + sample_legacy:
        if len(name.encode()) < 255:
            (dreams / name).write_text(f"---\ntitle: T-{name}\n---\n\n# body\n", encoding="utf-8")
    refresh_bundle_indexes(tmp_path)
    index = (dreams / "index.md").read_text()
    head = index.index(okf_vault.LEGACY_DIARY_HEADING)
    for name in sample_new:
        assert index.count(f"({name.replace(' ', '%20')})") == 1
        assert f"({name.replace(' ', '%20')})" in index[:head]
    for name in sample_legacy:
        assert f"({name.replace(' ', '%20')})" in index[head:]
        assert f"({name.replace(' ', '%20')})" not in index[:head]


def test_new_names_use_double_hyphen_and_32_hex():
    for pool in (V, O, A, "", "x" * 300, "user--\u212b"):
        name = reflect.diary_page_path(reflect.Path("/v/Dreams"), pool).name
        assert re.fullmatch(r"[^/]+--[0-9a-f]{32}\.md", name), name
        assert "--" not in name.rsplit("--", 1)[0]  # the slug itself has no "--"


# ── the EXACT id is hashed: no strip, no normalisation, no case fold ──

EXACT_PAIRS = [
    ("user--a\n", "user--a"),
    ("user--a\u2028", "user--a"),
    ("user--a ", "user--a"),
    (" user--a", "user--a"),
    ("user--\u212b", "user--\u00c5"),  # ANGSTROM SIGN vs A-ring: NFC folds them, slugs are equal
    ("user--\u212a", "user--K"),        # KELVIN SIGN vs K: NFC folds them, slugs are equal
    ("user--cafe\u0301", "user--caf\u00e9"),  # decomposed vs composed e-acute
    ("user--Alice", "user--alice"),
]


@pytest.mark.parametrize("p1,p2", EXACT_PAIRS)
def test_pools_differing_only_by_whitespace_or_normalisation_get_separate_pages(tmp_path, p1, p2):
    assert p1 != p2
    dreams = tmp_path / "Dreams"
    a, b = reflect.diary_page_path(dreams, p1), reflect.diary_page_path(dreams, p2)
    assert a != b and a.name.lower() != b.name.lower()
    _write(dreams, p1, "SECRET-ONE")
    _write(dreams, p2, "SECRET-TWO")
    assert "SECRET-TWO" not in a.read_text() and "SECRET-ONE" not in b.read_text()


def test_normalisation_pairs_really_equalise_under_the_mutated_transforms():
    """Guards the pair list: each mutation target must actually merge some pair."""
    from extensions.conversation_compiler.obsidian_writer import _slugify

    stripped = [(x, y) for x, y in EXACT_PAIRS if x.strip() == y.strip() and x != y]
    nfc = [(x, y) for x, y in EXACT_PAIRS
           if unicodedata.normalize("NFC", x) == unicodedata.normalize("NFC", y)]
    assert len(stripped) >= 3 and len(nfc) >= 3
    # NFC-only pairs whose slugs are equal: only the digest keeps these apart
    # (the e-acute pair differs in the slug too, so it cannot catch an NFC mutant)
    assert len([1 for x, y in nfc if _slugify(x) == _slugify(y)]) >= 2
    assert len([1 for x, y in stripped if _slugify(x) == _slugify(y)]) >= 3


# ── Dreams/index.md lists names and frontmatter titles only, never diary content ──


def test_index_carries_no_diary_or_legacy_body_text(tmp_path):
    dreams = tmp_path / "Dreams"
    legacy, legacy_secret = _make_legacy(dreams, O, secret="LEGACY-BODY-SECRET")
    _write(dreams, V, "PAGE-BODY-SECRET-V")
    _write(dreams, O, "PAGE-BODY-SECRET-O")
    refresh_bundle_indexes(tmp_path)
    index = (dreams / "index.md").read_text()
    for text in ("LEGACY-BODY-SECRET", "PAGE-BODY-SECRET-V", "PAGE-BODY-SECRET-O", "## Dream", "Reflections"):
        assert text not in index, text
    assert legacy.name in index  # names are listed
