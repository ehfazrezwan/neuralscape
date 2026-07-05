"""Unit tests for retrieval economics (C1 index-first recall + batch get, C2 timeline).

No running services: mem0/Qdrant/Graphiti are mocked. Covers the title
heuristic (clipping, cleanup, garbage), token estimation, index-row shape +
token budget, write-time stamping, get_memories bounds / missing-id /
visibility enforcement (no cross-user private reads, no existence oracle),
and timeline anchor resolution / ordering / depth bounds / tombstone +
visibility filter construction.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import index_format
from config import settings
from index_format import (
    UNTITLED,
    distill_title,
    estimate_tokens,
    glyph_for,
    humanize_age,
    index_row,
)
from memory_service import (
    GET_MEMORIES_MAX_IDS,
    TIMELINE_MAX_DEPTH,
    MemoryService,
)
from schemas import MemoryResponse


NOW = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _point(mid: str, content: str, user_id: str, created_at: str, **meta):
    metadata = {
        "scope": "global",
        "category": "preference",
        "owner_user_id": user_id,
        "visibility": "private",
        **meta,
    }
    return SimpleNamespace(
        id=mid,
        payload={
            "data": content,
            "created_at": created_at,
            "user_id": user_id,
            "metadata": metadata,
        },
    )


@pytest.fixture
def service():
    """MemoryService with mocked internals (mirrors test_provenance.py)."""
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.embedding_model.embed.return_value = [0.1] * 768
    svc._memory.vector_store.client.scroll.return_value = ([], None)
    svc._memory.vector_store.client.retrieve.return_value = []
    return svc


# ── Title heuristic ──────────────────────────────────────────────────


class TestDistillTitle:
    def test_short_sentence_passes_through(self):
        assert distill_title("User prefers dark mode") == "User prefers dark mode"

    def test_clips_to_ten_words_with_ellipsis(self):
        content = " ".join(f"w{i}" for i in range(30))
        title = distill_title(content)
        assert title.endswith(" …")
        assert len(title.replace(" …", "").split()) == 10

    def test_takes_first_sentence_only(self):
        title = distill_title("Deploy uses blue-green strategy! The rest is history and more.")
        assert title == "Deploy uses blue-green strategy"

    def test_strips_trailing_period(self):
        assert distill_title("Simple fact.") == "Simple fact"

    def test_strips_markdown_noise(self):
        assert distill_title("## Setup notes\nmore text") == "Setup notes"
        assert distill_title("- bullet item one") == "bullet item one"
        assert distill_title("> quoted wisdom") == "quoted wisdom"

    def test_strips_wrapping_quotes(self):
        assert distill_title('"Quoted fact here"') == "Quoted fact here"

    def test_skips_blank_and_divider_lines(self):
        assert distill_title("\n\n----\n\nreal content here") == "real content here"

    def test_char_cap_on_pathological_word(self):
        title = distill_title("x" * 500)
        assert len(title) <= index_format.TITLE_MAX_CHARS + 2  # + " …"
        assert title.endswith("…")

    def test_garbage_falls_back_to_flattened_content(self):
        # First line is pure punctuation; flattened content has signal.
        assert "meaning" in distill_title("?!\nmeaning lives here")

    def test_pure_garbage_returns_untitled(self):
        assert distill_title("!!! ??? ***") == UNTITLED
        assert distill_title("") == UNTITLED
        assert distill_title("   \n  ") == UNTITLED
        assert distill_title(None) == UNTITLED


class TestEstimateTokens:
    def test_ceil_len_over_four(self):
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("abcde") == 2
        assert estimate_tokens("x" * 400) == 100

    def test_empty_floors_at_one(self):
        assert estimate_tokens("") == 1
        assert estimate_tokens(None) == 1


class TestHumanizeAge:
    @pytest.mark.parametrize(
        "delta,expected",
        [
            (timedelta(seconds=10), "now"),
            (timedelta(minutes=5), "5m"),
            (timedelta(hours=3), "3h"),
            (timedelta(days=2), "2d"),
            (timedelta(days=10), "1w"),
            (timedelta(days=90), "3mo"),
            (timedelta(days=800), "2y"),
        ],
    )
    def test_buckets(self, delta, expected):
        assert humanize_age(_iso(NOW - delta), now=NOW) == expected

    def test_unparseable_and_missing(self):
        assert humanize_age("not-a-date", now=NOW) == "?"
        assert humanize_age(None, now=NOW) == "?"

    def test_z_suffix_and_naive(self):
        assert humanize_age((NOW - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), now=NOW) == "1h"
        assert humanize_age((NOW - timedelta(hours=1)).replace(tzinfo=None).isoformat(), now=NOW) == "1h"


class TestGlyphs:
    def test_known_types_have_glyphs(self):
        assert glyph_for("bugfix") != index_format.DEFAULT_GLYPH
        assert glyph_for("reflection") != index_format.DEFAULT_GLYPH

    def test_unknown_and_none_default(self):
        assert glyph_for("no_such_type") == index_format.DEFAULT_GLYPH
        assert glyph_for(None) == index_format.DEFAULT_GLYPH


# ── Index row shape + token budget ───────────────────────────────────


class TestIndexRow:
    def _mem(self, **kw):
        base = dict(
            id=str(uuid.uuid4()),
            memory="User prefers concise answers with code samples first.",
            category="preference",
            created_at=_iso(NOW - timedelta(days=3)),
            observation_type="pattern",
            score=0.87654,
        )
        base.update(kw)
        return MemoryResponse(**base)

    def test_shape(self):
        row = index_row(self._mem(), now=NOW)
        assert set(row) == {"id", "title", "category", "glyph", "age", "tokens", "score"}
        assert row["age"] == "3d"
        assert row["score"] == 0.8765

    def test_anchor_marker(self):
        assert index_row(self._mem(), anchor=True, now=NOW)["anchor"] is True
        assert "anchor" not in index_row(self._mem(), now=NOW)

    def test_uses_stored_title_and_estimate(self):
        mem = self._mem(title="Stored title wins", token_estimate=42)
        row = index_row(mem, now=NOW)
        assert row["title"] == "Stored title wins"
        assert row["tokens"] == 42

    def test_legacy_memory_computes_on_the_fly(self):
        mem = self._mem(title=None, token_estimate=None)
        row = index_row(mem, now=NOW)
        assert row["title"].startswith("User prefers concise answers")
        assert row["tokens"] == estimate_tokens(mem.memory)

    def test_rendered_row_stays_under_token_budget(self):
        # Worst case: max-length title, long category, score, anchor marker.
        mem = self._mem(
            title="w" * index_format.TITLE_MAX_CHARS,
            token_estimate=99999,
            category="meeting_outcome",
        )
        rendered = json.dumps(index_row(mem, anchor=True, now=NOW), ensure_ascii=False)
        # ~4 chars/token → stay under ~100 tokens per row.
        assert len(rendered) < 400

    def test_none_values_dropped(self):
        mem = self._mem(category=None, score=None)
        row = index_row(mem, now=NOW)
        assert "category" not in row
        assert "score" not in row


# ── Write-time stamping ──────────────────────────────────────────────


class TestWriteTimeStamping:
    # E2: token_estimate is stamped via savings_meter.stamp_tokens — a REAL
    # tiktoken count when the meter is enabled (default), the len/4 heuristic
    # when it's off. Field name and stamping site are unchanged.
    def test_store_raw_stamps_title_and_estimate(self, service):
        from savings_meter import stamp_tokens

        content = "User prefers dark mode in every editor they use daily."
        [resp] = service.store_raw(
            content=content, user_id="u1", category="preference", add_to_graph=False
        )
        assert resp.title == distill_title(content)
        assert resp.token_estimate == stamp_tokens(content)

        payload = service._memory.vector_store.insert.call_args.kwargs["payloads"][0]
        assert payload["metadata"]["title"] == distill_title(content)
        assert payload["metadata"]["token_estimate"] == stamp_tokens(content)

    def test_batch_store_facts_stamps_title_and_estimate(self, service):
        from savings_meter import stamp_tokens

        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]
        [resp] = service._batch_store_facts(
            [("preference", "Terse commit messages are preferred here.")], user_id="u1"
        )
        assert resp.title == "Terse commit messages are preferred here"
        assert resp.token_estimate == stamp_tokens("Terse commit messages are preferred here.")
        payload = service._memory.vector_store.insert.call_args.kwargs["payloads"][0]
        assert payload["metadata"]["title"] == resp.title

    def test_mem_to_response_surfaces_fields(self, service):
        resp = service._mem_to_response(
            {
                "id": "m1",
                "memory": "content",
                "metadata": {"title": "T", "token_estimate": 7},
            }
        )
        assert resp.title == "T"
        assert resp.token_estimate == 7

    def test_legacy_memory_fields_null(self, service):
        resp = service._mem_to_response({"id": "m1", "memory": "content", "metadata": {}})
        assert resp.title is None
        assert resp.token_estimate is None


# ── get_memories: bounds, missing ids, visibility ────────────────────


class TestGetMemoriesByIds:
    def test_empty_rejected(self, service):
        with pytest.raises(ValueError):
            service.get_memories_by_ids([], "u1")

    def test_over_cap_rejected(self, service):
        ids = [str(uuid.uuid4()) for _ in range(GET_MEMORIES_MAX_IDS + 1)]
        with pytest.raises(ValueError, match=str(GET_MEMORIES_MAX_IDS)):
            service.get_memories_by_ids(ids, "u1")

    def test_missing_and_malformed_ids_reported(self, service):
        good = str(uuid.uuid4())
        gone = str(uuid.uuid4())
        service._memory.vector_store.client.retrieve.return_value = [
            _point(good, "hello world content", "u1", _iso(NOW))
        ]
        out = service.get_memories_by_ids([good, gone, "not-a-uuid"], "u1")
        assert [r.id for r in out["results"]] == [good]
        assert set(out["missing"]) == {gone, "not-a-uuid"}
        # Malformed id never reaches Qdrant (it would 400 the whole retrieve).
        sent = service._memory.vector_store.client.retrieve.call_args.kwargs["ids"]
        assert "not-a-uuid" not in sent

    def test_cannot_read_other_users_private_memory(self, service):
        mid = str(uuid.uuid4())
        service._memory.vector_store.client.retrieve.return_value = [
            _point(mid, "secret", "other-user", _iso(NOW), visibility="private")
        ]
        out = service.get_memories_by_ids([mid], "u1")
        # Reported exactly like not-found — no existence oracle.
        assert out["results"] == []
        assert out["missing"] == [mid]

    def test_shared_memory_readable_cross_user(self, service):
        mid = str(uuid.uuid4())
        service._memory.vector_store.client.retrieve.return_value = [
            _point(mid, "team fact", "other-user", _iso(NOW), visibility="shared")
        ]
        out = service.get_memories_by_ids([mid], "u1")
        assert [r.id for r in out["results"]] == [mid]
        assert out["results"][0].visibility == "shared"

    def test_standard_readable_only_when_enabled(self, service):
        mid = str(uuid.uuid4())
        service._memory.vector_store.client.retrieve.return_value = [
            _point(mid, "org rule", "dictator", _iso(NOW), visibility="standard")
        ]
        saved = settings.standards_enabled
        try:
            settings.standards_enabled = False
            assert service.get_memories_by_ids([mid], "u1")["missing"] == [mid]
            settings.standards_enabled = True
            assert [r.id for r in service.get_memories_by_ids([mid], "u1")["results"]] == [mid]
        finally:
            settings.standards_enabled = saved

    def test_own_private_readable_and_order_preserved(self, service):
        a, b = str(uuid.uuid4()), str(uuid.uuid4())
        service._memory.vector_store.client.retrieve.return_value = [
            _point(b, "second", "u1", _iso(NOW)),
            _point(a, "first", "u1", _iso(NOW)),
        ]
        out = service.get_memories_by_ids([a, b, a], "u1")  # dupe collapses
        assert [r.id for r in out["results"]] == [a, b]
        assert out["missing"] == []

    def test_legacy_row_without_visibility_stays_private(self, service):
        mid = str(uuid.uuid4())
        pt = _point(mid, "legacy", "other-user", _iso(NOW))
        del pt.payload["metadata"]["visibility"]
        del pt.payload["metadata"]["owner_user_id"]
        service._memory.vector_store.client.retrieve.return_value = [pt]
        assert service.get_memories_by_ids([mid], "u1")["missing"] == [mid]


# ── Timeline ─────────────────────────────────────────────────────────


def _configure_timeline_scrolls(service, before_points, after_points):
    """Route client.scroll calls to before/after fixtures by order direction."""
    from qdrant_client.models import Direction

    def _scroll(collection_name, scroll_filter=None, limit=10, order_by=None, **kw):
        if order_by is not None and order_by.direction == Direction.DESC:
            return before_points[:limit], None
        return after_points[:limit], None

    service._memory.vector_store.client.scroll.side_effect = _scroll


class TestTimeline:
    def _seed_anchor(self, service, mid, created_at):
        service._memory.vector_store.client.retrieve.return_value = [
            _point(mid, "anchor content", "u1", created_at)
        ]

    def test_anchor_by_id_window_ordering(self, service):
        anchor_id = str(uuid.uuid4())
        self._seed_anchor(service, anchor_id, _iso(NOW))
        b1 = _point(str(uuid.uuid4()), "older-1", "u1", _iso(NOW - timedelta(hours=2)))
        b2 = _point(str(uuid.uuid4()), "older-2", "u1", _iso(NOW - timedelta(hours=1)))
        a1 = _point(str(uuid.uuid4()), "newer-1", "u1", _iso(NOW + timedelta(hours=1)))
        # Before-scroll returns newest-first (DESC), service must flip to ASC.
        _configure_timeline_scrolls(service, [b2, b1], [a1])

        out = service.timeline(anchor_id, user_id="u1", depth=5)
        assert out["anchor_id"] == anchor_id
        contents = [m.memory for m in out["memories"]]
        assert contents == ["older-1", "older-2", "anchor content", "newer-1"]

    def test_anchor_by_query_uses_best_vector_hit(self, service):
        anchor_id = str(uuid.uuid4())
        graph_hit = MemoryResponse(id="edge-1", memory="graph fact", source="graph")
        vector_hit = MemoryResponse(
            id=anchor_id, memory="anchor content", source="vector",
            created_at=_iso(NOW), category="decision",
        )
        service.search = MagicMock(return_value=[graph_hit, vector_hit])
        _configure_timeline_scrolls(service, [], [])

        out = service.timeline("what changed in deploy", user_id="u1", depth=3)
        assert out["anchor_id"] == anchor_id
        service.search.assert_called_once()
        assert out["memories"][0].id == anchor_id

    def test_unresolvable_anchor_returns_none(self, service):
        service._memory.vector_store.client.retrieve.return_value = []
        assert service.timeline(str(uuid.uuid4()), user_id="u1") is None
        service.search = MagicMock(return_value=[])
        assert service.timeline("no hits for this", user_id="u1") is None

    def test_cannot_anchor_on_other_users_private_memory(self, service):
        mid = str(uuid.uuid4())
        service._memory.vector_store.client.retrieve.return_value = [
            _point(mid, "secret", "other-user", _iso(NOW), visibility="private")
        ]
        assert service.timeline(mid, user_id="u1") is None

    def test_depth_clamped(self, service):
        anchor_id = str(uuid.uuid4())
        self._seed_anchor(service, anchor_id, _iso(NOW))
        seen_limits = []

        def _scroll(collection_name, scroll_filter=None, limit=10, order_by=None, **kw):
            seen_limits.append(limit)
            return [], None

        service._memory.vector_store.client.scroll.side_effect = _scroll
        service.timeline(anchor_id, user_id="u1", depth=999)
        assert seen_limits == [TIMELINE_MAX_DEPTH, TIMELINE_MAX_DEPTH]
        seen_limits.clear()
        service.timeline(anchor_id, user_id="u1", depth=-3)
        assert seen_limits == [1, 1]

    def test_filter_excludes_tombstones_and_foreign_private(self, service):
        from qdrant_client.models import FieldCondition, Filter, HasIdCondition

        flt = service._timeline_filter("u1", None, "anchor-x", NOW, "before")
        # must_not: tombstone + anchor exclusion
        tombstone = [
            c for c in flt.must_not
            if isinstance(c, FieldCondition) and c.key == "metadata.dream_tombstoned"
        ]
        assert tombstone and tombstone[0].match.value is True
        anchor_excl = [c for c in flt.must_not if isinstance(c, HasIdCondition)]
        assert anchor_excl and anchor_excl[0].has_id == ["anchor-x"]
        # Visibility union: own rows OR shared (+ standard only when enabled)
        vis = next(c for c in flt.must if isinstance(c, Filter))
        keys = [(c.key, c.match.value) for c in vis.should]
        assert ("user_id", "u1") in keys
        assert ("metadata.visibility", "shared") in keys

    def test_filter_standard_pool_gated_by_setting(self, service):
        from qdrant_client.models import Filter

        saved = settings.standards_enabled
        try:
            settings.standards_enabled = True
            flt = service._timeline_filter("u1", None, None, NOW, "after")
            vis = next(c for c in flt.must if isinstance(c, Filter))
            assert ("metadata.visibility", "standard") in [
                (c.key, c.match.value) for c in vis.should
            ]
            settings.standards_enabled = False
            flt = service._timeline_filter("u1", None, None, NOW, "after")
            vis = next(c for c in flt.must if isinstance(c, Filter))
            assert ("metadata.visibility", "standard") not in [
                (c.key, c.match.value) for c in vis.should
            ]
        finally:
            settings.standards_enabled = saved

    def test_filter_project_dual_scope(self, service):
        from qdrant_client.models import Filter

        flt = service._timeline_filter("u1", "proj-x", None, NOW, "after")
        nested = [c for c in flt.must if isinstance(c, Filter)]
        assert len(nested) == 2  # visibility union + project/global union
        scope_should = [
            (c.key, c.match.value) for c in nested[1].should
        ]
        assert ("metadata.project_id", "proj-x") in scope_should
        assert ("metadata.scope", "global") in scope_should

    def test_order_by_failure_falls_back_to_python_sort(self, service):
        anchor_id = str(uuid.uuid4())
        self._seed_anchor(service, anchor_id, _iso(NOW))
        b1 = _point(str(uuid.uuid4()), "older-1", "u1", _iso(NOW - timedelta(hours=2)))
        b2 = _point(str(uuid.uuid4()), "older-2", "u1", _iso(NOW - timedelta(hours=1)))

        from qdrant_client.models import FieldCondition

        def _is_before(flt):
            for c in flt.must:
                if isinstance(c, FieldCondition) and c.key == "created_at":
                    return c.range.lt is not None
            return False

        def _scroll(collection_name, scroll_filter=None, limit=10, order_by=None, offset=None, **kw):
            if order_by is not None:
                raise RuntimeError("order_by requires a payload index")
            if offset is not None:
                return [], None
            # Emulate the range filter the mock store can't apply itself.
            return ([b1, b2] if _is_before(scroll_filter) else []), None

        service._memory.vector_store.client.scroll.side_effect = _scroll
        out = service.timeline(anchor_id, user_id="u1", depth=1)
        # depth=1 → nearest-before only (b2), anchor, nothing after.
        contents = [m.memory for m in out["memories"]]
        assert contents == ["older-2", "anchor content"]

    def test_unwindowable_anchor_degrades_to_anchor_only(self, service):
        anchor_id = str(uuid.uuid4())
        self._seed_anchor(service, anchor_id, "not-a-timestamp")
        out = service.timeline(anchor_id, user_id="u1")
        assert out["anchor_id"] == anchor_id
        assert [m.id for m in out["memories"]] == [anchor_id]

    def test_blank_anchor_rejected(self, service):
        with pytest.raises(ValueError):
            service.timeline("   ", user_id="u1")


# ── REST endpoints ───────────────────────────────────────────────────


@pytest.fixture
def rest_client():
    from fastapi.testclient import TestClient

    import main

    mock_svc = MagicMock(name="MemoryService")
    original = main._service
    main._service = mock_svc
    # No context manager: skip lifespan (which validates required env vars).
    client = TestClient(main.app, raise_server_exceptions=False)
    yield client, mock_svc
    main._service = original


class TestRestEndpoints:
    def _mem(self, mid=None, **kw):
        base = dict(
            id=mid or str(uuid.uuid4()),
            memory="User prefers concise answers.",
            category="preference",
            created_at=_iso(NOW - timedelta(days=1)),
            title="User prefers concise answers",
            token_estimate=8,
            score=0.9,
        )
        base.update(kw)
        return MemoryResponse(**base)

    def test_search_index_only_returns_compact_rows(self, rest_client):
        client, svc = rest_client
        svc.search.return_value = [self._mem()]
        resp = client.post(
            "/v1/search",
            json={"query": "prefs", "user_id": "u1", "index_only": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["index_only"] is True
        row = body["results"][0]
        assert row["title"] == "User prefers concise answers"
        assert row["tokens"] == 8
        assert "memory" not in row

    def test_search_default_returns_full_payloads(self, rest_client):
        client, svc = rest_client
        svc.search.return_value = [self._mem()]
        resp = client.post("/v1/search", json={"query": "prefs", "user_id": "u1"})
        assert resp.status_code == 200
        assert resp.json()["results"][0]["memory"] == "User prefers concise answers."

    def test_batch_get_roundtrip(self, rest_client):
        client, svc = rest_client
        mem = self._mem()
        svc.get_memories_by_ids.return_value = {"results": [mem], "missing": ["gone-id"]}
        resp = client.post(
            "/v1/memories/batch-get",
            json={"ids": [mem.id, "gone-id"], "user_id": "u1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"][0]["memory"] == mem.memory
        assert body["missing"] == ["gone-id"]

    def test_batch_get_validates_bounds(self, rest_client):
        client, _ = rest_client
        assert client.post(
            "/v1/memories/batch-get", json={"ids": [], "user_id": "u1"}
        ).status_code == 422
        assert client.post(
            "/v1/memories/batch-get",
            json={"ids": [str(uuid.uuid4()) for _ in range(51)], "user_id": "u1"},
        ).status_code == 422

    def test_timeline_marks_anchor(self, rest_client):
        client, svc = rest_client
        anchor = self._mem()
        other = self._mem()
        svc.timeline.return_value = {
            "anchor_id": anchor.id,
            "memories": [other, anchor],
        }
        resp = client.post(
            "/v1/timeline", json={"anchor": anchor.id, "user_id": "u1"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["anchor_id"] == anchor.id
        rows = body["results"]
        assert rows[0].get("anchor") is None
        assert rows[1]["anchor"] is True

    def test_timeline_404_when_unresolvable(self, rest_client):
        client, svc = rest_client
        svc.timeline.return_value = None
        resp = client.post(
            "/v1/timeline", json={"anchor": "no such thing", "user_id": "u1"}
        )
        assert resp.status_code == 404

    def test_timeline_depth_validated(self, rest_client):
        client, _ = rest_client
        assert client.post(
            "/v1/timeline", json={"anchor": "x", "depth": 0, "user_id": "u1"}
        ).status_code == 422
        assert client.post(
            "/v1/timeline", json={"anchor": "x", "depth": 51, "user_id": "u1"}
        ).status_code == 422


# ── MCP tools ────────────────────────────────────────────────────────


@pytest.fixture
def mcp_service():
    import mcp_server

    mock_svc = MagicMock(name="MemoryService")
    original = mcp_server._service
    mcp_server._service = mock_svc
    yield mock_svc
    mcp_server._service = original


class TestMcpTools:
    def _mem(self, **kw):
        base = dict(
            id=str(uuid.uuid4()),
            memory="A fact worth knowing about the deploy pipeline.",
            category="decision",
            created_at=_iso(NOW - timedelta(hours=4)),
            title="A fact worth knowing about the deploy pipeline",
            token_estimate=12,
        )
        base.update(kw)
        return MemoryResponse(**base)

    @pytest.mark.asyncio
    async def test_recall_index_only(self, mcp_service):
        import mcp_server

        mcp_service.search.return_value = [self._mem()]
        [res] = await mcp_server.call_tool(
            "recall_memories", {"query": "deploy", "user_id": "u1", "index_only": True}
        )
        body = json.loads(res.text)
        assert body["index_only"] is True
        assert body["results"][0]["tokens"] == 12
        assert "memory" not in body["results"][0]

    @pytest.mark.asyncio
    async def test_get_memories_tool(self, mcp_service):
        import mcp_server

        mem = self._mem()
        mcp_service.get_memories_by_ids.return_value = {"results": [mem], "missing": []}
        [res] = await mcp_server.call_tool(
            "get_memories", {"ids": [mem.id], "user_id": "u1"}
        )
        body = json.loads(res.text)
        assert body["results"][0]["memory"] == mem.memory
        assert body["missing"] == []

    @pytest.mark.asyncio
    async def test_get_memories_tool_rejects_bad_ids_arg(self, mcp_service):
        import mcp_server

        [res] = await mcp_server.call_tool("get_memories", {"ids": [], "user_id": "u1"})
        assert "error" in json.loads(res.text)
        [res] = await mcp_server.call_tool("get_memories", {"ids": "not-a-list", "user_id": "u1"})
        assert "error" in json.loads(res.text)

    @pytest.mark.asyncio
    async def test_timeline_tool_marks_anchor(self, mcp_service):
        import mcp_server

        anchor = self._mem()
        mcp_service.timeline.return_value = {
            "anchor_id": anchor.id,
            "memories": [anchor],
        }
        [res] = await mcp_server.call_tool(
            "timeline", {"anchor": anchor.id, "user_id": "u1"}
        )
        body = json.loads(res.text)
        assert body["anchor_id"] == anchor.id
        assert body["results"][0]["anchor"] is True

    @pytest.mark.asyncio
    async def test_timeline_tool_unresolvable_anchor(self, mcp_service):
        import mcp_server

        mcp_service.timeline.return_value = None
        [res] = await mcp_server.call_tool(
            "timeline", {"anchor": "nothing here", "user_id": "u1"}
        )
        assert "error" in json.loads(res.text)
