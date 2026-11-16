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
