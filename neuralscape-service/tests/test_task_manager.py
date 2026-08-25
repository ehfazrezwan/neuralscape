"""Tests for TaskManager — covers v2 enqueue paths and status helpers.

The Redis pool itself is mocked via AsyncMock; these tests verify only the
arguments TaskManager forwards to ``pool.enqueue_job`` and the shape of the
status responses it returns.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from task_manager import TaskManager, _generate_job_id


# ──────────────────────────────────────────────
# Job ID helper
# ──────────────────────────────────────────────


class TestGenerateJobId:
    def test_deterministic(self):
        a = _generate_job_id("hello", "u1")
        b = _generate_job_id("hello", "u1")
        assert a == b
        assert a.startswith("ns-")

    def test_different_user_different_id(self):
        a = _generate_job_id("hello", "u1")
        b = _generate_job_id("hello", "u2")
        assert a != b


# ──────────────────────────────────────────────
# TaskManager fixture
# ──────────────────────────────────────────────


@pytest.fixture
def tm():
    """A TaskManager with a mocked ARQ pool."""
    manager = TaskManager()
    pool = MagicMock()
    pool.enqueue_job = AsyncMock()
    pool.aclose = AsyncMock()
    manager.pool = pool
    return manager


# ──────────────────────────────────────────────
# enqueue_raw v2 fields forwarding (memory-model v2)
# ──────────────────────────────────────────────


class TestEnqueueRawV2:
    @pytest.mark.asyncio
    async def test_v2_fields_packed_into_extras(self, tm):
        """v2 fields are packed into a single v2_extras dict argument."""
        mock_job = MagicMock()
        mock_job.job_id = "job-xyz"
        tm.pool.enqueue_job.return_value = mock_job

        await tm.enqueue_raw(
            content="x",
            user_id="ehfaz",
            category="decision",
            scope="project",
            project_id="proj1",
            tags=["tag1"],
            domain="coding",
            observation_type="decision",
            concepts=["why-it-exists"],
            source_type="tool_extraction",
            confidence=0.9,
            expires_at="2026-12-01T00:00:00+00:00",
        )

        # The 9th positional arg (after content..run_id) is the v2_extras dict
        args, _ = tm.pool.enqueue_job.call_args[0], tm.pool.enqueue_job.call_args[1]
        positional = tm.pool.enqueue_job.call_args[0]
        assert positional[0] == "process_memory_raw"
        v2_extras = positional[9]
        assert v2_extras["domain"] == "coding"
        assert v2_extras["observation_type"] == "decision"
        assert v2_extras["concepts"] == ["why-it-exists"]
        assert v2_extras["source_type"] == "tool_extraction"
        assert v2_extras["confidence"] == 0.9
        assert v2_extras["expires_at"] == "2026-12-01T00:00:00+00:00"

    @pytest.mark.asyncio
    async def test_memory_kind_and_source_ref_forwarded(self, tm):
        """Connector provenance fields reach the worker via v2_extras (C1)."""
        tm.pool.enqueue_job.return_value = None  # simulate dup → returns job_id
        src = {"connector_id": "notion-x", "connector_type": "notion"}
        await tm.enqueue_raw(
            content="x",
            user_id="ehfaz",
            category="domain_knowledge",
            memory_kind="passage",
            source_ref=src,
        )
        v2_extras = tm.pool.enqueue_job.call_args[0][9]
        assert v2_extras["memory_kind"] == "passage"
        assert v2_extras["source_ref"] == src

    @pytest.mark.asyncio
    async def test_job_id_differs_by_visibility(self, tm):
        """Same text at different visibility tiers must get DISTINCT job ids, or
        ARQ coalesces a standard-promotion onto the earlier private/shared job and
        silently drops it before it reaches store_raw."""
        tm.pool.enqueue_job.return_value = None
        ids = {}
        for vis in ("private", "standard"):
            await tm.enqueue_raw(content="same text", user_id="d", category="convention", visibility=vis)
            ids[vis] = tm.pool.enqueue_job.call_args[1]["_job_id"]
        assert ids["private"] != ids["standard"]

    @pytest.mark.asyncio
    async def test_ingest_file_job_id_differs_by_page_offset(self, tm):
        """Same file re-uploaded with a corrected page_offset must be a NEW job —
        page_offset changes exemplar provenance content, so coalescing onto the
        earlier job's cached result would silently drop the correction."""
        tm.pool.enqueue_job.return_value = None
        ids = {}
        for offset in (0, 60):
            payload = {
                "filename": "slice.pdf",
                "user_id": "d",
                "source_ref": {"external_id": "samehash"},
                "options": {"visibility": "shared", "adapter": "trading_strategy",
                            **({"page_offset": offset} if offset else {})},
            }
            await tm.enqueue_ingest_file(payload)
            ids[offset] = tm.pool.enqueue_job.call_args.kwargs["_job_id"]
        assert ids[0] != ids[60]

    @pytest.mark.asyncio
    async def test_connector_sync_job_id_matches_cron_scheme(self, tm):
        """enqueue_connector_sync uses the same sync-<id> job id as the cron (C2)."""
        tm.pool.enqueue_job.return_value = None
        job_id = await tm.enqueue_connector_sync("notion-personal")
        assert job_id == "sync-notion-personal"
        assert tm.pool.enqueue_job.call_args.kwargs["_job_id"] == "sync-notion-personal"

    @pytest.mark.asyncio
    async def test_no_v2_fields_packs_empty_extras(self, tm):
        """Without v2 fields, extras dict carries only defaulted booleans
        (None values are dropped; `sensitivity_override` defaults to False,
        which survives the None-filter since False is not None)."""
        mock_job = MagicMock()
        mock_job.job_id = "job-xyz"
        tm.pool.enqueue_job.return_value = mock_job

        await tm.enqueue_raw(
            content="x",
            user_id="ehfaz",
            category="preference",
        )
        v2_extras = tm.pool.enqueue_job.call_args[0][9]
        assert v2_extras == {"sensitivity_override": False}

    @pytest.mark.asyncio
    async def test_duplicate_job_returns_existing_id(self, tm):
        """When ARQ returns None (deduplicated job), TaskManager returns the deterministic id."""
        tm.pool.enqueue_job.return_value = None
        result = await tm.enqueue_raw(
            content="x", user_id="ehfaz", category="preference",
        )
        assert result.startswith("ns-")


# ──────────────────────────────────────────────
# enqueue_raw_batch (memory-model v2)
# ──────────────────────────────────────────────


class TestEnqueueRawBatch:
    @pytest.mark.asyncio
    async def test_dispatches_single_job_with_items(self, tm):
        mock_job = MagicMock()
        mock_job.job_id = "batch-xyz"
        tm.pool.enqueue_job.return_value = mock_job

        items = [
            {"content": "A", "user_id": "ehfaz", "category": "preference"},
            {"content": "B", "user_id": "ehfaz", "category": "personal_fact"},
        ]
        result = await tm.enqueue_raw_batch(items=items)

        assert result == "batch-xyz"
        tm.pool.enqueue_job.assert_called_once()
        positional = tm.pool.enqueue_job.call_args[0]
        assert positional[0] == "process_memory_raw_batch"
        assert positional[1] == items
        # Deterministic job ID derived from concatenated content + first user_id
        assert tm.pool.enqueue_job.call_args[1]["_job_id"].startswith("ns-")

    @pytest.mark.asyncio
    async def test_duplicate_batch_returns_existing_id(self, tm):
        tm.pool.enqueue_job.return_value = None
        items = [{"content": "A", "user_id": "ehfaz", "category": "preference"}]
        result = await tm.enqueue_raw_batch(items=items)
        assert result.startswith("ns-")

    @pytest.mark.asyncio
    async def test_empty_batch_uses_fallback_user(self, tm):
        """Edge case: empty items still produces a valid job ID via 'batch' fallback."""
        mock_job = MagicMock()
        mock_job.job_id = "batch-empty"
        tm.pool.enqueue_job.return_value = mock_job

        result = await tm.enqueue_raw_batch(items=[])
        assert result == "batch-empty"
        # The fallback user_id "batch" was used to derive the job id
        job_id = tm.pool.enqueue_job.call_args[1]["_job_id"]
        assert job_id.startswith("ns-")

    @pytest.mark.asyncio
    async def test_pipe_in_content_does_not_collide(self, tm):
        """Distinct batches with `|` in content must produce distinct job IDs.

        Regression for CR-13: the old `"|".join(...)` made
        `["a", "b|c"]` and `["a|b", "c"]` indistinguishable.
        """
        mock_job = MagicMock()
        mock_job.job_id = "x"
        tm.pool.enqueue_job.return_value = mock_job
        captured_ids: list[str] = []

        async def capture(*args, **kwargs):
            captured_ids.append(kwargs["_job_id"])
            return mock_job

        tm.pool.enqueue_job.side_effect = capture

        await tm.enqueue_raw_batch(items=[
            {"content": "a", "user_id": "u", "category": "preference"},
            {"content": "b|c", "user_id": "u", "category": "preference"},
        ])
        await tm.enqueue_raw_batch(items=[
            {"content": "a|b", "user_id": "u", "category": "preference"},
            {"content": "c", "user_id": "u", "category": "preference"},
        ])
        assert len(captured_ids) == 2
        assert captured_ids[0] != captured_ids[1]

    @pytest.mark.asyncio
    async def test_same_items_produce_same_job_id(self, tm):
        """Determinism: re-submitting the identical batch must still dedup."""
        mock_job = MagicMock()
        mock_job.job_id = "x"
        tm.pool.enqueue_job.return_value = mock_job
        captured: list[str] = []

        async def capture(*args, **kwargs):
            captured.append(kwargs["_job_id"])
            return mock_job

        tm.pool.enqueue_job.side_effect = capture

        items = [{"content": "a", "user_id": "u", "category": "preference"}]
        await tm.enqueue_raw_batch(items=items)
        await tm.enqueue_raw_batch(items=items)
        assert captured[0] == captured[1]


# ──────────────────────────────────────────────
# enqueue_store
# ──────────────────────────────────────────────


class TestEnqueueRetag:
    _FILTERS = {"project_id": "neuralscape", "category": "decision"}
    _OPS = {"add_tags": ["project:bon002"]}

    @pytest.mark.asyncio
    async def test_dispatches_on_fast_queue(self, tm):
        tm.pool.enqueue_job.return_value = None
        await tm.enqueue_retag("robb", dict(self._FILTERS), dict(self._OPS))
        positional = tm.pool.enqueue_job.call_args[0]
        assert positional[0] == "process_memory_retag"
        assert positional[1] == "robb"
        assert positional[2] == self._FILTERS
        assert positional[3] == self._OPS
        # Fast queue: no _queue_name override
        assert "_queue_name" not in tm.pool.enqueue_job.call_args[1]

    @pytest.mark.asyncio
    async def test_job_id_deterministic_and_key_order_insensitive(self, tm):
        tm.pool.enqueue_job.return_value = None
        a = await tm.enqueue_retag("robb", {"category": "decision", "project_id": "p"}, dict(self._OPS))
        b = await tm.enqueue_retag("robb", {"project_id": "p", "category": "decision"}, dict(self._OPS))
        assert a == b  # canonical JSON: dict key order can't fork the id

    @pytest.mark.asyncio
    async def test_distinct_retags_get_distinct_ids(self, tm):
        tm.pool.enqueue_job.return_value = None
        a = await tm.enqueue_retag("robb", dict(self._FILTERS), {"add_tags": ["x"]})
        b = await tm.enqueue_retag("robb", dict(self._FILTERS), {"add_tags": ["y"]})
        c = await tm.enqueue_retag("javi", dict(self._FILTERS), {"add_tags": ["x"]})
        assert len({a, b, c}) == 3


class TestEnqueueGraphEnrichmentForEdit:
    @pytest.mark.asyncio
    async def test_dispatches_on_graph_queue(self, tm):
        from config import settings

        tm.pool.enqueue_job.return_value = None
        await tm.enqueue_graph_enrichment(
            "m1", "new content", "ehfaz", "p1", "shared", None
        )
        positional = tm.pool.enqueue_job.call_args[0]
        kwargs = tm.pool.enqueue_job.call_args[1]
        assert positional[0] == "process_graph_enrichment"
        assert positional[1:8] == ("m1", "new content", "ehfaz", "p1", "shared", None, None)
        assert kwargs["_queue_name"] == settings.graph_queue_name

    @pytest.mark.asyncio
    async def test_job_id_keys_on_target_state(self, tm):
        """Editing the same memory to a DIFFERENT state must be a new job;
        replaying the same edit coalesces."""
        tm.pool.enqueue_job.return_value = None
        a = await tm.enqueue_graph_enrichment("m1", "c", "e", "p1", "shared")
        replay = await tm.enqueue_graph_enrichment("m1", "c", "e", "p1", "shared")
        diff_project = await tm.enqueue_graph_enrichment("m1", "c", "e", "p2", "shared")
        diff_content = await tm.enqueue_graph_enrichment("m1", "c2", "e", "p1", "shared")
        assert a == replay
        assert len({a, diff_project, diff_content}) == 3


class TestEnqueueStore:
    @pytest.mark.asyncio
    async def test_dispatches_with_messages(self, tm):
        mock_job = MagicMock()
        mock_job.job_id = "store-1"
        tm.pool.enqueue_job.return_value = mock_job

        result = await tm.enqueue_store(
            messages=[{"role": "user", "content": "hello"}],
            user_id="ehfaz",
            project_id="proj",
        )
        assert result == "store-1"
        positional = tm.pool.enqueue_job.call_args[0]
        assert positional[0] == "process_memory_store"

    @pytest.mark.asyncio
    async def test_duplicate_store_returns_existing_id(self, tm):
        tm.pool.enqueue_job.return_value = None
        result = await tm.enqueue_store(
            messages=[{"role": "user", "content": "hello"}],
            user_id="ehfaz",
        )
        assert result.startswith("ns-")


# ──────────────────────────────────────────────
# Status / wait helpers
# ──────────────────────────────────────────────


class TestGetStatus:
    @pytest.mark.asyncio
    async def test_completed_with_success_result(self, tm):
        from arq.jobs import JobStatus

        mock_info = MagicMock()
        mock_info.success = True
        mock_info.result = {"memories": [{"id": "m1"}]}

        with patch("task_manager.Job") as MockJob:
            instance = MockJob.return_value
            instance.status = AsyncMock(return_value=JobStatus.complete)
            instance.result_info = AsyncMock(return_value=mock_info)
            result = await tm.get_status("task-1")

        assert result["status"] == "completed"
        assert result["result"] == {"memories": [{"id": "m1"}]}
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_completed_with_failure_result(self, tm):
        from arq.jobs import JobStatus

        mock_info = MagicMock()
        mock_info.success = False
        mock_info.result = "Something blew up"

        with patch("task_manager.Job") as MockJob:
            instance = MockJob.return_value
            instance.status = AsyncMock(return_value=JobStatus.complete)
            instance.result_info = AsyncMock(return_value=mock_info)
            result = await tm.get_status("task-2")

        assert result["status"] == "failed"
        assert "Something blew up" in result["error"]

    @pytest.mark.asyncio
    async def test_in_progress(self, tm):
        from arq.jobs import JobStatus

        with patch("task_manager.Job") as MockJob:
            instance = MockJob.return_value
            instance.status = AsyncMock(return_value=JobStatus.in_progress)
            result = await tm.get_status("task-3")

        assert result["status"] == "processing"
        assert result["result"] is None

    @pytest.mark.asyncio
    async def test_not_found(self, tm):
        from arq.jobs import JobStatus

        with patch("task_manager.Job") as MockJob:
            instance = MockJob.return_value
            instance.status = AsyncMock(return_value=JobStatus.not_found)
            result = await tm.get_status("missing")

        assert result["status"] == "not_found"


class TestWaitForResult:
    @pytest.mark.asyncio
    async def test_returns_result_on_success(self, tm):
        from arq.jobs import JobStatus
        with patch("task_manager.Job") as MockJob:
            instance = MockJob.return_value
            # _find_job probes status across candidate queues before awaiting result.
            instance.status = AsyncMock(return_value=JobStatus.complete)
            instance.result = AsyncMock(return_value={"memories": []})
            result = await tm.wait_for_result("task-x", timeout=5)
        assert result["status"] == "completed"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_returns_failed_on_exception(self, tm):
        from arq.jobs import JobStatus
        with patch("task_manager.Job") as MockJob:
            instance = MockJob.return_value
            instance.status = AsyncMock(return_value=JobStatus.complete)
            instance.result = AsyncMock(side_effect=Exception("Worker crashed"))
            result = await tm.wait_for_result("task-y", timeout=5)
        assert result["status"] == "failed"
        assert "Worker crashed" in result["error"]


class TestConnectClose:
    @pytest.mark.asyncio
    async def test_close_with_pool(self, tm):
        await tm.close()
        tm.pool.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_without_pool_is_safe(self):
        manager = TaskManager()
        # No pool set; close should not raise
        await manager.close()

    @pytest.mark.asyncio
    async def test_connect_creates_pool(self):
        """connect() calls create_pool with the configured Redis settings."""
        manager = TaskManager()
        with patch("task_manager.create_pool", new=AsyncMock(return_value="fake-pool")) as mock_create:
            await manager.connect()
        mock_create.assert_awaited_once()
        assert manager.pool == "fake-pool"
