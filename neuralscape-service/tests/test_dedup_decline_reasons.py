"""The semantic dedup phase reports why it left a pair alone, and names each merged pair."""

import logging

from tests.test_dedup import _FakeQdrant, _merged, _row, service  # noqa: F401  (fixtures reused)

TEXT = "The nightly report job runs at 02:00 UTC on weekdays"


def test_merged_pair_is_named_in_an_info_line_and_the_summary_has_every_counter(service, caplog):
    _FakeQdrant(service, [
        _row("old", TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
        _row("new", TEXT, "2025-06-01T00:00:00", scope="global", visibility="private"),
    ])
    with caplog.at_level(logging.INFO):
        result = service.dedup_memories("u1")

    assert _merged(service) == {"old"}
    assert result["semantic_duplicates_removed"] == 1
    declined = result["semantic_declined"]
    assert set(declined) >= {
        "neighbour_other_partition", "neighbour_tombstoned", "neighbour_category_twin",
        "neighbour_other_pool", "neighbour_already_merged", "row_gone_before_compare",
        "row_without_text", "search_failed", "merge_failed",
    }
    assert not any(declined.values())
    text = caplog.text
    assert "dedup semantic merge user=u1 older=old newer=new score=" in text
    assert "mode=delete" in text or "mode=tombstone" in text
    assert "declined_neighbour_other_partition=0" in text and "semantic_pairs_merged=1" in text


def test_a_near_identical_row_in_another_partition_is_declined_and_counted(service):
    _FakeQdrant(service, [
        _row("a", TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
        # Global rows push no project filter down, so the neighbour comes back and the strict check declines it.
        _row("b", TEXT, "2025-06-01T00:00:00", scope="global", visibility="private", project_id="other"),
    ])
    result = service.dedup_memories("u1")

    assert result["semantic_duplicates_removed"] == 0
    assert result["semantic_declined"]["neighbour_other_partition"] >= 1


def test_a_failed_neighbour_search_is_counted(service):
    fake = _FakeQdrant(service, [
        _row("a", TEXT, "2025-01-01T00:00:00", scope="global", visibility="private"),
    ])
    service._memory.vector_store.search = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    result = service.dedup_memories("u1")

    assert result["semantic_duplicates_removed"] == 0
    assert result["semantic_declined"]["search_failed"] == 1
    del fake
