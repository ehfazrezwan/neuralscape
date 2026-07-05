"""Unit tests for `occurred_at` — the event-time envelope field.

NS stamps `created_at` at write time; `occurred_at` is the optional
event-time override for historical ingestion (imported journals, old chat
exports). Absence means "event time unknown — fall back to created_at";
it is NEVER defaulted to the storage time.

Covers: validation (ISO 8601, future-date rejection with clock-skew
allowance, naive→UTC), request-model exposure, write-path payload
stamping (raw / batch / conversation), absent-means-absent, response
surfacing, ask evidence rendering + recency-discipline text, and MCP
tool-schema exposure. All external services mocked (unit-test convention).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from memory_service import MemoryService
from schemas import (
    IngestTextRequest,
    MemoryResponse,
    RawMemoryRequest,
    StoreMemoryRequest,
    validate_occurred_at,
)


# ──────────────────────────────────────────────
# Piece 1: validation + schema exposure
# ──────────────────────────────────────────────


class TestValidateOccurredAt:
    def test_none_passes_through(self):
        assert validate_occurred_at(None) is None

    def test_aware_iso_canonicalized_to_utc(self):
        # Non-UTC offsets are converted, not preserved: one spelling per
        # instant keeps lexicographic recency sorting and the deterministic
        # enqueue job key stable across ISO spellings (Copilot, PR #130).
        out = validate_occurred_at("2023-05-01T10:00:00+02:00")
        assert out == "2023-05-01T08:00:00+00:00"

    def test_equivalent_spellings_normalize_identically(self):
        spellings = [
            "2023-05-01T08:00:00+00:00",
            "2023-05-01T08:00:00Z",
            "2023-05-01T10:00:00+02:00",
            "2023-05-01T08:00:00",
        ]
        assert len({validate_occurred_at(v) for v in spellings}) == 1

    def test_event_time_sort_key_is_offset_proof(self):
        from types import SimpleNamespace

        from ask import _event_time

        # 11:00+02:00 is 09:00 UTC; it must compare AFTER 08:30 UTC under the
        # sort key even though raw-string comparison across mixed offsets
        # would be meaningless.
        older = SimpleNamespace(occurred_at=None, created_at="2023-05-01T08:30:00+00:00")
        newer = SimpleNamespace(occurred_at="2023-05-01T11:00:00+02:00", created_at=None)
        assert _event_time(newer) > _event_time(older)
        # Unparseable values degrade to the raw string instead of raising.
        junk = SimpleNamespace(occurred_at="not-a-date", created_at=None)
        assert _event_time(junk) == "not-a-date"

    def test_z_suffix_accepted(self):
        out = validate_occurred_at("2023-05-01T10:00:00Z")
        assert out == "2023-05-01T10:00:00+00:00"

    def test_naive_assumes_utc(self):
        out = validate_occurred_at("2023-05-01T10:00:00")
        assert out == "2023-05-01T10:00:00+00:00"

    def test_date_only_accepted(self):
        # fromisoformat accepts a bare date; it normalizes to midnight UTC.
        out = validate_occurred_at("2023-05-01")
        assert out == "2023-05-01T00:00:00+00:00"

    def test_garbage_rejected(self):
        with pytest.raises(ValueError, match="occurred_at"):
            validate_occurred_at("last tuesday")

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="occurred_at"):
            validate_occurred_at("")

    def test_far_future_rejected(self):
        future = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
        with pytest.raises(ValueError, match="future"):
            validate_occurred_at(future)

    def test_future_within_clock_skew_accepted(self):
        near = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        assert validate_occurred_at(near) == near

    def test_datetime_input_accepted(self):
        dt = datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        assert validate_occurred_at(dt) == dt.isoformat()


class TestRequestModels:
    def test_raw_request_accepts_and_normalizes(self):
        req = RawMemoryRequest(
            content="x", category="preference",
            occurred_at="2023-05-01T10:00:00Z",
        )
        assert req.occurred_at == "2023-05-01T10:00:00+00:00"

    def test_raw_request_defaults_none(self):
        req = RawMemoryRequest(content="x", category="preference")
        assert req.occurred_at is None

    def test_raw_request_rejects_garbage(self):
        with pytest.raises(ValidationError, match="occurred_at"):
            RawMemoryRequest(content="x", category="preference", occurred_at="nope")

    def test_raw_request_rejects_far_future(self):
        future = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
        with pytest.raises(ValidationError):
            RawMemoryRequest(content="x", category="preference", occurred_at=future)

    def test_conversation_request_accepts(self):
        req = StoreMemoryRequest(
            messages=[{"role": "user", "content": "hi"}],
            occurred_at="2022-01-01T00:00:00+00:00",
        )
        assert req.occurred_at == "2022-01-01T00:00:00+00:00"

    def test_conversation_request_rejects_garbage(self):
        with pytest.raises(ValidationError, match="occurred_at"):
            StoreMemoryRequest(messages=[{"role": "user", "content": "hi"}],
                               occurred_at="whenever")

    def test_ingest_text_request_accepts(self):
        req = IngestTextRequest(content="ctx", occurred_at="2021-06-01T12:00:00Z")
        assert req.occurred_at == "2021-06-01T12:00:00+00:00"

    def test_ingest_text_request_rejects_garbage(self):
        with pytest.raises(ValidationError, match="occurred_at"):
            IngestTextRequest(content="ctx", occurred_at="not a time")

    def test_memory_response_has_field_defaulting_none(self):
        resp = MemoryResponse(id="m1", memory="x")
        assert resp.occurred_at is None


# ──────────────────────────────────────────────
# Piece 2: write-path payload stamping
# ──────────────────────────────────────────────


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
    return svc


class TestStoreRawOccurredAt:
    def test_stamps_metadata_and_response(self, service):
        result = service.store_raw(
            content="Moved to Berlin",
            user_id="u1",
            category="personal_fact",
            occurred_at="2019-03-15T00:00:00+00:00",
        )
        payload = service._memory.vector_store.insert.call_args[1]["payloads"][0]
        assert payload["metadata"]["occurred_at"] == "2019-03-15T00:00:00+00:00"
        assert result[0].occurred_at == "2019-03-15T00:00:00+00:00"
        # created_at is still the storage time, distinct from the event time.
        assert result[0].created_at != result[0].occurred_at

    def test_normalizes_naive_input(self, service):
        result = service.store_raw(
            content="Old journal entry",
            user_id="u1",
            category="personal_fact",
            occurred_at="2019-03-15T08:30:00",
        )
        payload = service._memory.vector_store.insert.call_args[1]["payloads"][0]
        assert payload["metadata"]["occurred_at"] == "2019-03-15T08:30:00+00:00"
        assert result[0].occurred_at == "2019-03-15T08:30:00+00:00"

    def test_absent_means_absent(self, service):
        """No occurred_at → key omitted from metadata, response None.

        Absence means "event time unknown, fall back to created_at" —
        it must NOT be defaulted to the storage time.
        """
        result = service.store_raw(
            content="A fact with unknown event time",
            user_id="u1",
            category="preference",
        )
        payload = service._memory.vector_store.insert.call_args[1]["payloads"][0]
        assert "occurred_at" not in payload["metadata"]
        assert result[0].occurred_at is None
        assert result[0].created_at is not None

    def test_invalid_occurred_at_raises(self, service):
        with pytest.raises(ValueError, match="occurred_at"):
            service.store_raw(
                content="x", user_id="u1", category="preference",
                occurred_at="not-a-time",
            )
        service._memory.vector_store.insert.assert_not_called()

    def test_batch_item_passthrough(self, service):
        n = 768
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * n]
        result = service.store_raw_batch([
            {
                "content": "Historic fact",
                "user_id": "u1",
                "category": "personal_fact",
                "occurred_at": "2020-06-01T00:00:00+00:00",
            }
        ])
        payload = service._memory.vector_store.insert.call_args[1]["payloads"][0]
        assert payload["metadata"]["occurred_at"] == "2020-06-01T00:00:00+00:00"
        assert result[0].occurred_at == "2020-06-01T00:00:00+00:00"


class TestConversationPathOccurredAt:
    def test_batch_store_facts_stamps_all(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768] * 2
        result = service._batch_store_facts(
            facts=[("preference", "Likes tea"), ("personal_fact", "Lived in Oslo")],
            user_id="u1",
            occurred_at="2021-02-03T00:00:00+00:00",
        )
        payloads = service._memory.vector_store.insert.call_args[1]["payloads"]
        assert all(
            p["metadata"]["occurred_at"] == "2021-02-03T00:00:00+00:00"
            for p in payloads
        )
        assert all(r.occurred_at == "2021-02-03T00:00:00+00:00" for r in result)

    def test_batch_store_facts_absent_means_absent(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]
        result = service._batch_store_facts(
            facts=[("preference", "Likes coffee")],
            user_id="u1",
        )
        payloads = service._memory.vector_store.insert.call_args[1]["payloads"]
        assert "occurred_at" not in payloads[0]["metadata"]
        assert result[0].occurred_at is None

    def test_batch_store_facts_rejects_invalid(self, service):
        with pytest.raises(ValueError, match="occurred_at"):
            service._batch_store_facts(
                facts=[("preference", "x")], user_id="u1", occurred_at="garbage",
            )


class TestResponseSurfacing:
    def test_mem_to_response_maps_occurred_at(self):
        svc = MemoryService()
        resp = svc._mem_to_response({
            "id": "m1",
            "memory": "x",
            "created_at": "2026-07-01T00:00:00+00:00",
            "metadata": {
                "category": "personal_fact",
                "occurred_at": "2018-01-01T00:00:00+00:00",
            },
        })
        assert resp.occurred_at == "2018-01-01T00:00:00+00:00"
        assert resp.created_at == "2026-07-01T00:00:00+00:00"

    def test_mem_to_response_legacy_null(self):
        svc = MemoryService()
        resp = svc._mem_to_response({"id": "m1", "memory": "x", "metadata": {}})
        assert resp.occurred_at is None

    def test_dedup_hit_response_carries_occurred_at(self, service):
        """_find_by_content_hash's response builder surfaces occurred_at."""
        point = MagicMock()
        point.id = "existing-id"
        point.payload = {
            "data": "x",
            "created_at": "2026-07-01T00:00:00+00:00",
            "metadata": {
                "category": "personal_fact",
                "scope": "global",
                "occurred_at": "2017-05-05T00:00:00+00:00",
            },
        }
        service._memory.vector_store.client.scroll.return_value = ([point], None)
        existing = service._find_by_content_hash(
            user_id="u1", content_hash="h", scope="global",
            project_id=None, visibility=None,
        )
        assert existing.occurred_at == "2017-05-05T00:00:00+00:00"


# ──────────────────────────────────────────────
# Piece 2b: enqueue plumbing (API → worker)
# ──────────────────────────────────────────────


@pytest.fixture
def tm():
    """A TaskManager with a mocked ARQ pool (mirrors test_task_manager.py)."""
    from unittest.mock import AsyncMock

    from task_manager import TaskManager

    manager = TaskManager()
    pool = MagicMock()
    pool.enqueue_job = AsyncMock()
    pool.aclose = AsyncMock()
    manager.pool = pool
    return manager


class TestEnqueuePlumbing:
    @pytest.mark.asyncio
    async def test_enqueue_raw_forwards_occurred_at_in_v2_extras(self, tm):
        job = MagicMock()
        job.job_id = "job-1"
        tm.pool.enqueue_job.return_value = job
        await tm.enqueue_raw(
            content="x", user_id="u1", category="personal_fact",
            occurred_at="2020-01-01T00:00:00+00:00",
        )
        positional = tm.pool.enqueue_job.call_args[0]
        v2_extras = positional[9]
        assert v2_extras["occurred_at"] == "2020-01-01T00:00:00+00:00"

    @pytest.mark.asyncio
    async def test_enqueue_raw_omits_when_absent(self, tm):
        job = MagicMock()
        job.job_id = "job-2"
        tm.pool.enqueue_job.return_value = job
        await tm.enqueue_raw(content="x", user_id="u1", category="personal_fact")
        v2_extras = tm.pool.enqueue_job.call_args[0][9]
        assert "occurred_at" not in v2_extras

    @pytest.mark.asyncio
    async def test_enqueue_store_forwards_occurred_at(self, tm):
        job = MagicMock()
        job.job_id = "job-3"
        tm.pool.enqueue_job.return_value = job
        await tm.enqueue_store(
            messages=[{"role": "user", "content": "hi"}],
            user_id="u1",
            occurred_at="2019-09-09T00:00:00+00:00",
        )
        positional = tm.pool.enqueue_job.call_args[0]
        # args: task_name, messages, user_id, project_id, agent_id, run_id, occurred_at
        assert positional[6] == "2019-09-09T00:00:00+00:00"

    @pytest.mark.asyncio
    async def test_worker_raw_task_passes_occurred_at_to_store(self, service):
        """process_memory_raw forwards v2_extras['occurred_at'] to store_raw."""
        from unittest.mock import AsyncMock

        import worker as worker_mod

        svc = MagicMock(name="MemoryService")
        svc.search.return_value = []
        stored_mem = MemoryResponse(
            id="m1", memory="x", occurred_at="2020-01-01T00:00:00+00:00",
        )
        svc.store_raw.return_value = ([stored_mem], True)
        ctx = {"service": svc, "redis": MagicMock(enqueue_job=AsyncMock())}
        await worker_mod.process_memory_raw(
            ctx, "x", "u1", "personal_fact",
            v2_extras={"occurred_at": "2020-01-01T00:00:00+00:00"},
        )
        assert (
            svc.store_raw.call_args[1]["occurred_at"]
            == "2020-01-01T00:00:00+00:00"
        )


# ──────────────────────────────────────────────
# Piece 3: ingest-text pipeline threading
# ──────────────────────────────────────────────


class _FakeIngestService:
    """Records store_raw calls (mirrors test_ingest_pipeline.FakeService)."""

    def __init__(self, facts=None):
        self.store_calls = []
        self._facts = facts or []

    def store_raw(self, **kwargs):
        self.store_calls.append(kwargs)
        responses = [MemoryResponse(id=f"id-{len(self.store_calls)}",
                                    memory=kwargs["content"])]
        if kwargs.get("return_created"):
            return responses, True
        return responses

    def extract_facts_only(self, text, extractor=None, user_id=None, project_id=None):
        return list(self._facts)


class TestIngestPipelineOccurredAt:
    _SOURCE = {"connector_id": "manual", "connector_type": "manual",
               "external_id": "hash-1"}

    def test_passages_and_facts_carry_occurred_at(self):
        from ingest.pipeline import IngestDoc, ingest_document

        svc = _FakeIngestService(facts=[("domain_knowledge", "A distilled fact.")])
        doc = IngestDoc(
            content="word " * 300,
            source=dict(self._SOURCE),
            user_id="u1",
            max_chars=200,
            overlap=20,
            occurred_at="2015-08-01T00:00:00+00:00",
        )
        ingest_document(svc, doc)
        assert svc.store_calls  # passages + fact
        assert all(
            c["occurred_at"] == "2015-08-01T00:00:00+00:00"
            for c in svc.store_calls
        )

    def test_absent_stays_absent(self):
        from ingest.pipeline import IngestDoc, ingest_document

        svc = _FakeIngestService(facts=[("domain_knowledge", "A distilled fact.")])
        doc = IngestDoc(
            content="word " * 300,
            source=dict(self._SOURCE),
            user_id="u1",
            max_chars=200,
            overlap=20,
        )
        ingest_document(svc, doc)
        assert all(c.get("occurred_at") is None for c in svc.store_calls)


# ──────────────────────────────────────────────
# Piece 4: ask evidence rendering + recency discipline
# ──────────────────────────────────────────────


def _ask_mem(mid, content, created_at="2026-07-01T00:00:00+00:00",
             occurred_at=None, score=0.9):
    return MemoryResponse(
        id=mid, memory=content, category="personal_fact", source="vector",
        created_at=created_at, occurred_at=occurred_at, score=score,
    )


class TestAskEvidenceOccurredAt:
    def test_render_shows_event_time_when_present(self):
        import ask as ask_mod

        row = _ask_mem("m1", "Moved to Berlin",
                       created_at="2026-07-01T00:00:00+00:00",
                       occurred_at="2019-03-15T00:00:00+00:00")
        rendered = ask_mod._render_evidence([row])
        assert "event: 2019-03-15T00:00:00+00:00" in rendered
        assert "stored: 2026-07-01T00:00:00+00:00" in rendered

    def test_render_falls_back_to_created_at_verbatim(self):
        """Rows without occurred_at keep the exact pre-existing format."""
        import ask as ask_mod

        row = _ask_mem("m1", "Likes tea")
        rendered = ask_mod._render_evidence([row])
        assert rendered == "[m1] (2026-07-01T00:00:00+00:00; personal_fact) Likes tea"

    def test_rows_sorted_by_event_time_when_present(self):
        """A row stored today about an old event sorts by its EVENT time."""
        import ask as ask_mod

        old_event_stored_today = _ask_mem(
            "old-event", "Lived in Oslo",
            created_at="2026-07-01T00:00:00+00:00",
            occurred_at="2015-01-01T00:00:00+00:00",
        )
        newer = _ask_mem("newer", "Moved to Berlin",
                         created_at="2020-01-01T00:00:00+00:00")
        evidence = {m.id: m for m in [old_event_stored_today, newer]}
        out = ask_mod._evidence_rows(evidence, [], False)
        # Chronological ascending by event time: 2015 event first, then 2020.
        assert [m.id for m in out] == ["old-event", "newer"]

    def test_recency_discipline_mentions_event_time(self):
        import ask as ask_mod

        assert "event time" in ask_mod._DISCIPLINES_FULL
        assert "event time" in ask_mod._DISCIPLINES_BRIEF


# ──────────────────────────────────────────────
# Piece 5: MCP tool-schema exposure
# ──────────────────────────────────────────────


@pytest.fixture
def mock_mcp_task_manager():
    """Patch the TaskManager in the MCP server module (mirrors test_mcp_tools)."""
    from unittest.mock import AsyncMock

    import mcp_server

    mock_tm = MagicMock(name="TaskManager")
    mock_tm.enqueue_raw = AsyncMock(return_value="task-oa")
    original = mcp_server._task_manager
    mcp_server._task_manager = mock_tm
    yield mock_tm
    mcp_server._task_manager = original


class TestMcpRememberOccurredAt:
    @pytest.mark.asyncio
    async def test_remember_schema_exposes_occurred_at(self):
        import mcp_server

        tools = {t.name: t for t in await mcp_server.list_tools()}
        props = tools["remember"].inputSchema["properties"]
        assert "occurred_at" in props
        assert props["occurred_at"]["type"] == "string"
        # Documented as the historical-ingestion event time.
        assert "happened" in props["occurred_at"]["description"]

    @pytest.mark.asyncio
    async def test_remember_forwards_occurred_at(self, mock_mcp_task_manager):
        import mcp_server

        await mcp_server.call_tool("remember", {
            "content": "Moved to Berlin",
            "category": "personal_fact",
            "user_id": "u1",
            "occurred_at": "2019-03-15T00:00:00Z",
        })
        kwargs = mock_mcp_task_manager.enqueue_raw.call_args[1]
        # Normalized at the tool boundary (Z → +00:00).
        assert kwargs["occurred_at"] == "2019-03-15T00:00:00+00:00"

    @pytest.mark.asyncio
    async def test_remember_rejects_invalid_occurred_at_before_enqueue(
        self, mock_mcp_task_manager
    ):
        import json as _json

        import mcp_server

        out = await mcp_server.call_tool("remember", {
            "content": "x",
            "category": "personal_fact",
            "user_id": "u1",
            "occurred_at": "not-a-date",
        })
        body = _json.loads(out[0].text)
        assert "occurred_at" in body.get("error", "")
        mock_mcp_task_manager.enqueue_raw.assert_not_called()
