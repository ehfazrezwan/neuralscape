"""Another user's private content reaches the operator vault only through Dreams/.

``Dreams/<pool>.md`` is written for every pool by design (single-operator vault,
owner-accepted). The OKF index files and Home must never repeat that content.
"""

from __future__ import annotations

from extensions.dreaming import reflect
from okf.vault import refresh_bundle_indexes

POOL = "user--someone-else"


def test_only_the_dream_diary_carries_another_users_private_text(tmp_path):
    entry = reflect.render_diary_entry(
        pool=POOL, run_id="sweep",
        applied=[{"type": "merge", "memory_ids": ["m1", "m2"], "reason": "Rust cheat sheet zqTOKEN"}],
        reported=[],
        insights=[{"lens": "pattern", "content": "prefers Rust, keeps a zqTOKEN cheat sheet",
                   "source_memory_ids": ["m1"]}],
    )
    reflect.write_diary(tmp_path / "Dreams", POOL, entry, source_memory_ids=["m1", "m2"])
    (tmp_path / "Home.md").write_text("---\ntitle: Home\n---\n# Home\n")
    refresh_bundle_indexes(tmp_path)

    carriers = sorted(
        str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")
        if p.is_file() and ("Rust" in p.read_text() or "zqTOKEN" in p.read_text())
    )
    assert carriers == ["Dreams/user-someone-else.md"], carriers


def test_dreams_index_is_not_a_diary_page_and_stays_clean(tmp_path):
    reflect.write_diary(tmp_path / "Dreams", POOL, "## Dream\nRust zqTOKEN\n", source_memory_ids=["m1"])
    refresh_bundle_indexes(tmp_path)
    index = (tmp_path / "Dreams" / "index.md").read_text()
    assert "zqTOKEN" not in index and "Rust" not in index


def _sweep_writes(vault, pool=POOL):
    """The vault writers a sweep runs for one pool, in sweep order."""
    entry = reflect.render_diary_entry(
        pool=pool, run_id="sweep", applied=[], reported=[],
        insights=[{"lens": "pattern", "content": "prefers Rust, keeps a zqTOKEN cheat sheet",
                   "source_memory_ids": ["m1"]}],
    )
    diary = reflect.write_diary(vault / "Dreams", pool, entry, source_memory_ids=["m1"])
    reflect.update_vault_log(vault, pool, diary_rel=diary, applied=0, insights=1)
    (vault / "Home.md").write_text("---\ntitle: Home\n---\n# Home\n")
    refresh_bundle_indexes(vault)


def test_real_layout_pool_id_is_only_in_diary_log_and_dreams_index(tmp_path):
    """Real vault layout: the pool NAME shows in log.md and Dreams/index.md, the pool's TEXT only in its diary."""
    from integration_tests._dreaming_support import marker_hits

    _sweep_writes(tmp_path)
    uid = POOL.split("--", 1)[1]
    id_files = sorted({h.split(":", 1)[0] for h in marker_hits(tmp_path, [uid])})
    assert id_files == ["Dreams/index.md", "Dreams/user-someone-else.md", "log.md"], id_files
    text_files = sorted({h.split(":", 1)[0] for h in marker_hits(tmp_path, ["zqTOKEN", "Rust"])})
    assert text_files == ["Dreams/user-someone-else.md"], text_files


def test_marker_hits_names_file_and_line_and_honours_skip(tmp_path):
    from integration_tests._dreaming_support import marker_hits

    _sweep_writes(tmp_path)
    hits = marker_hits(tmp_path, ["zqTOKEN"])
    assert len(hits) == 1 and hits[0].startswith("Dreams/user-someone-else.md:") and "zqTOKEN" in hits[0]
    assert marker_hits(tmp_path, ["zqTOKEN"], skip=lambda rel: rel.startswith("Dreams/")) == []
