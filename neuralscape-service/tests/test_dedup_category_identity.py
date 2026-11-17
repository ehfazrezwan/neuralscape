"""The same text under another category is a distinct memory.

The content hash covers text only, and ``_find_by_content_hash`` used to ignore
the category, so a "convention" written after a "preference" of the same text
was absorbed into the old row. The lookup now filters on ``metadata.category``
(rows keep their hashes) on the explicit (raw) write path only: extraction stays
category-agnostic, and the dedup cron never merges a same-hash pair whose
categories differ.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from memory.delete import _dedup_partition, _dedup_partition_filters
from memory_service import MemoryService

TEXT = "Deploys freeze on Fridays."


def _row(pid, category, text=TEXT, created_at="2025-01-01T00:00:00", **meta):
    md = {"scope": "global", "visibility": "private", "category": category, **meta}
    return pid, {"user_id": "u1", "hash": "h-same", "data": text, "created_at": created_at, "metadata": md}


def _condition_value(scroll_filter, key):
    for cond in scroll_filter.must:
        if getattr(cond, "key", None) == key:
            return cond.match.value
    return None


@pytest.fixture
def service(monkeypatch):
    from tests.test_dedup import _UpRedis

    up = _UpRedis()  # dedup takes pool locks and does nothing destructive without Redis
    monkeypatch.setattr("extensions.dreaming.sweep._get_redis", lambda: up)
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._memory.embedding_model.embed.return_value = [0.1] * 8
    svc._memory.embedding_model.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 8 for _ in texts]
    store = {}

    def scroll(collection_name=None, scroll_filter=None, limit=1, **kw):
        # Honour the metadata.category condition the way Qdrant would.
        wanted = _condition_value(scroll_filter, "metadata.category")
        hits = [
            SimpleNamespace(id=pid, payload=payload)
            for pid, payload in store.items()
            if wanted is None or payload["metadata"].get("category") == wanted
        ]
        return hits[:1], None

    def insert(vectors=None, payloads=None, ids=None):
        for pid, payload in zip(ids, payloads):
            store[pid] = payload

    svc._memory.vector_store.client.scroll.side_effect = scroll
    svc._memory.vector_store.client.retrieve.return_value = []
    svc._memory.vector_store.insert.side_effect = insert
    svc.store = store
    return svc


def _put(svc, category, **kw):
    out, created = svc.store_raw(
        content=TEXT, user_id="u1", category=category, add_to_graph=False, return_created=True, **kw
    )
    return out[0].id, created


class TestRawPath:
    def test_same_text_under_two_categories_is_two_rows(self, service):
        first, c1 = _put(service, "preference")
        second, c2 = _put(service, "convention")
        assert c1 is True and c2 is True
        assert first != second
        assert {p["metadata"]["category"] for p in service.store.values()} == {"preference", "convention"}

    def test_same_text_same_category_still_dedups(self, service):
        first, _ = _put(service, "preference")
        again, created = _put(service, "preference")
        assert created is False and again == first
        assert len(service.store) == 1

    def test_lookup_filters_on_category_without_touching_the_hash(self, service):
        service._find_by_content_hash = MagicMock(return_value=None)
        service.store_raw(content=TEXT, user_id="u1", category="convention", add_to_graph=False)
        assert service._find_by_content_hash.call_args.kwargs["category"] == "convention"
        from memory.hashing import content_hash

        import inspect

        assert list(inspect.signature(content_hash).parameters) == ["content"]


class TestLookup:
    def test_category_condition_is_added_when_given(self, service):
        service._find_by_content_hash("u1", "h", "global", category="convention")
        flt = service._memory.vector_store.client.scroll.call_args.kwargs["scroll_filter"]
        assert _condition_value(flt, "metadata.category") == "convention"

    def test_no_category_condition_when_omitted(self, service):
        service._find_by_content_hash("u1", "h", "global")
        flt = service._memory.vector_store.client.scroll.call_args.kwargs["scroll_filter"]
        assert _condition_value(flt, "metadata.category") is None


class TestConversationPathStaysCategoryAgnostic:
    """A model-assigned category is a guess, not intent: extraction dedups on text alone."""

    def test_exact_duplicate_under_another_extracted_category_dedups(self, service):
        service._get_memory = MagicMock(return_value=service._memory)
        first = service._batch_store_facts(facts=[("preference", TEXT)], user_id="u1")
        again = service._batch_store_facts(facts=[("convention", TEXT)], user_id="u1")
        assert again[0].id == first[0].id
        assert len(service.store) == 1

    def test_same_text_two_categories_in_one_batch_collapse(self, service):
        service._get_memory = MagicMock(return_value=service._memory)
        out = service._batch_store_facts(facts=[("preference", TEXT), ("convention", TEXT)], user_id="u1")
        assert len({m.id for m in out}) == 1 and len(service.store) == 1


def _scroll_rows(service, rows):
    service._memory.vector_store.client.scroll.side_effect = None
    points = [SimpleNamespace(id=pid, payload=payload) for pid, payload in rows]
    service._memory.vector_store.client.scroll.return_value = (points, None)
    # the re-read after a pool lock sees the same rows
    service._memory.vector_store.client.retrieve.side_effect = (
        lambda collection_name=None, ids=None, **kw: [p for p in points if p.id in set(ids or [])]
    )


class TestDedupCron:
    def test_partition_is_unchanged_by_category(self):
        a = _row("a", "preference")[1]
        b = _row("b", "convention")[1]
        assert _dedup_partition(a) == _dedup_partition(b)
        assert "metadata.category" not in _dedup_partition_filters("u1", a)

    def test_exact_phase_keeps_a_same_hash_pair_under_two_categories(self, service):
        _scroll_rows(service, [
            _row("p1", "preference", created_at="2025-01-01T00:00:00"),
            _row("p2", "convention", created_at="2025-06-01T00:00:00"),
        ])
        service._memory.vector_store.search.return_value = []

        result = service.dedup_memories("u1", semantic=False)

        assert result["exact_duplicates_removed"] == 0
        service._memory.vector_store.delete.assert_not_called()

    def test_semantic_phase_keeps_a_same_hash_pair_under_two_categories(self, service):
        rows = [
            _row("p1", "preference", created_at="2025-01-01T00:00:00"),
            _row("p2", "convention", created_at="2025-06-01T00:00:00"),
        ]
        _scroll_rows(service, rows)
        # the search finds the twin with a perfect score: only the same-hash guard can stop the merge
        service._memory.vector_store.search.side_effect = lambda **kw: [
            SimpleNamespace(id=pid, score=1.0, payload=payload) for pid, payload in rows
        ]

        result = service.dedup_memories("u1", semantic=True)

        assert result["semantic_duplicates_removed"] == 0 and result["exact_duplicates_removed"] == 0
        service._memory.vector_store.delete.assert_not_called()

    def test_semantic_near_duplicates_with_different_hashes_still_merge_across_categories(self, service):
        older = ("p1", {**_row("p1", "preference", text="Deploys freeze on Fridays")[1], "hash": "h-1"})
        newer = ("p2", {**_row("p2", "convention", text="Deploys freeze on Fridays.",
                               created_at="2025-06-01T00:00:00")[1], "hash": "h-2"})
        _scroll_rows(service, [older, newer])
        service._memory.vector_store.search.side_effect = lambda **kw: [
            SimpleNamespace(id=pid, score=0.99, payload=payload) for pid, payload in (older, newer)
        ]

        result = service.dedup_memories("u1", semantic=True)

        assert result["semantic_duplicates_removed"] == 1
        service._memory.vector_store.delete.assert_called_once_with("p1")

    def test_cron_still_merges_same_text_same_category(self, service):
        _scroll_rows(service, [
            _row("p1", "preference", created_at="2025-01-01T00:00:00"),
            _row("p2", "preference", created_at="2025-06-01T00:00:00"),
        ])

        result = service.dedup_memories("u1", semantic=False)

        assert result["exact_duplicates_removed"] == 1
        service._memory.vector_store.delete.assert_called_once_with("p1")

    def test_legacy_rows_without_a_category_still_dedup(self, service):
        def legacy(pid, created):
            _, payload = _row(pid, None, created_at=created)
            payload["metadata"].pop("category")
            return pid, payload

        _scroll_rows(service, [legacy("p1", "2025-01-01T00:00:00"), legacy("p2", "2025-06-01T00:00:00")])
        result = service.dedup_memories("u1", semantic=False)
        assert result["exact_duplicates_removed"] == 1
