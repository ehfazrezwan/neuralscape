"""Vault link formats: topic pages use wikilinks to their hub; OKF index.md
files use standard markdown links and are not topic pages.

Pins the product contract the flag-on integration test relies on when it
selects a project's topic pages (it must skip reserved bundle filenames).
"""

from __future__ import annotations

from extensions.conversation_compiler.obsidian_writer import _atomic_write
from extensions.dreaming import librarian as lib
from okf.translate import is_reserved_filename
from okf.vault import refresh_bundle_indexes


def _vault(tmp_path):
    topic = lib.render_topic_page(
        title="Data Storage and Backend",
        pool="shared--project--alpha",
        summary="Storage stack",
        memory_ids=["a", "b"],
        categories=["architecture"],
        hub_link="alpha",
        version=1,
        index_card=[{"what": "Qdrant holds vectors", "entities": [], "source": "Decisions & Facts"}],
        sections={"Decisions & Facts": "- Qdrant holds vectors"},
        content_hash="deadbeef",
        faded_lines=[],
    )
    pdir = tmp_path / "Projects" / "alpha"
    _atomic_write(pdir / "Data Storage and Backend.md", topic)
    lib._write_hub(pdir, "alpha", _atomic_write)
    refresh_bundle_indexes(tmp_path)
    return pdir


def test_topic_pages_wikilink_their_hub_but_bundle_index_does_not(tmp_path):
    pdir = _vault(tmp_path)
    index = pdir / "index.md"
    assert index.exists() and is_reserved_filename(index.name)
    index_text = index.read_text(encoding="utf-8")
    assert "[[alpha]]" not in index_text
    assert "(Data%20Storage%20and%20Backend.md)" in index_text  # standard markdown links
    topics = [p for p in pdir.glob("*.md") if p.stem not in ("alpha", "Card") and not is_reserved_filename(p.name)]
    assert [p.name for p in topics] == ["Data Storage and Backend.md"]
    assert "[[alpha]]" in topics[0].read_text(encoding="utf-8")
