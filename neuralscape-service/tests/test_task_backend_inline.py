"""Inline task backend interim behavior (solo engine, pulled forward from
unit 4): with ``task_backend != "redis"`` the TaskManager never touches Redis,
its pool stays a falsy sentinel, and any enqueue raises ConnectionError — the
exact exception type every API/MCP write path's synchronous fallback catches.

Found by the solo e2e boot: previously connect() raised on a missing Redis
(startup crash), and a None pool raised AttributeError from enqueues, which
the fallbacks do NOT catch.
"""

import asyncio

import pytest

from task_manager import TaskManager, _DISABLED_POOL


class TestDisabledPool:
    def test_is_falsy(self):
        assert not _DISABLED_POOL

    def test_any_use_raises_connection_error(self):
        with pytest.raises(ConnectionError, match="task queue unavailable"):
            _DISABLED_POOL.enqueue_job

    def test_fallback_catch_tuple_matches(self):
        """The write paths catch (ConnectionError, OSError) — the sentinel's
        error must land inside that tuple."""
        try:
            _DISABLED_POOL.zadd
        except (ConnectionError, OSError):
            pass


class TestInlineConnect:
    def test_connect_skips_redis_in_inline_mode(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "task_backend", "inline")
        called = {"create_pool": False}

        async def fake_create_pool(*a, **k):
            called["create_pool"] = True

        monkeypatch.setattr("task_manager.create_pool", fake_create_pool)
        tm = TaskManager()
        asyncio.run(tm.connect())
        assert called["create_pool"] is False
        assert not tm.pool

    def test_connect_uses_redis_in_team_mode(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "task_backend", "redis")
        sentinel_pool = object()

        async def fake_create_pool(*a, **k):
            return sentinel_pool

        monkeypatch.setattr("task_manager.create_pool", fake_create_pool)
        tm = TaskManager()
        asyncio.run(tm.connect())
        assert tm.pool is sentinel_pool

    def test_enqueue_raises_connection_error_when_disabled(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "task_backend", "inline")
        tm = TaskManager()
        asyncio.run(tm.connect())
        with pytest.raises(ConnectionError):
            asyncio.run(tm.enqueue_raw("x", "u", "preference"))


# ── Unit 4: the real inline backend (runner + manager) ──────────────


from types import SimpleNamespace


def _stub_worker(runner, **fns):
    """Swap the runner's worker module for stubs (post-construction)."""
    runner._worker = SimpleNamespace(**fns)


def _make_runner():
    import inline_tasks

    async def _build():
        return inline_tasks.InlineTaskRunner(service=object())

    return inline_tasks, _build


class TestInlineTaskRunner:
    def test_lifecycle_and_result(self):
        inline_tasks, build = _make_runner()

        async def scenario():
            runner = await build()

            async def ok(ctx, value):
                return {"echo": value}

            _stub_worker(runner, process_memory_raw=ok)
            assert runner.submit("process_memory_raw", ("hi",), {}, "t1") is True
            got = await runner.wait("t1", timeout=5)
            return got

        got = asyncio.run(scenario())
        assert got == {"task_id": "t1", "status": "completed", "result": {"echo": "hi"}, "error": None}

    def test_failure_reports_error(self):
        inline_tasks, build = _make_runner()

        async def scenario():
            runner = await build()

            async def boom(ctx):
                raise RuntimeError("kaput")

            _stub_worker(runner, process_memory_raw=boom)
            runner.submit("process_memory_raw", (), {}, "t2")
            return await runner.wait("t2", timeout=5)

        got = asyncio.run(scenario())
        assert got["status"] == "failed" and "kaput" in got["error"]

    def test_live_job_id_dedup_mirrors_arq(self):
        inline_tasks, build = _make_runner()

        async def scenario():
            runner = await build()
            gate = asyncio.Event()

            async def waits(ctx):
                await gate.wait()
                return {}

            _stub_worker(runner, process_memory_raw=waits)
            first = runner.submit("process_memory_raw", (), {}, "dup")
            second = runner.submit("process_memory_raw", (), {}, "dup")  # live → refused
            gate.set()
            await runner.wait("dup", timeout=5)
            third = runner.submit("process_memory_raw", (), {}, "dup")  # finished → allowed
            gate.set()
            await runner.wait("dup", timeout=5)
            return first, second, third

        assert asyncio.run(scenario()) == (True, False, True)

    def test_slow_lane_concurrency_cap(self):
        inline_tasks, build = _make_runner()

        async def scenario():
            runner = await build()
            release = asyncio.Event()
            started: list[str] = []

            async def slow(ctx, tag):
                started.append(tag)
                await release.wait()
                return {}

            _stub_worker(runner, process_graph_enrichment=slow)
            for i in range(3):
                runner.submit("process_graph_enrichment", (f"g{i}",), {}, f"g{i}")
            await asyncio.sleep(0.05)
            concurrent = len(started)  # cap = 2
            release.set()
            for i in range(3):
                await runner.wait(f"g{i}", timeout=5)
            return concurrent, len(started)

        concurrent, total = asyncio.run(scenario())
        assert concurrent == 2 and total == 3

    def test_deferred_enqueue_routes_to_slow_lane(self):
        """A fast task's ctx['redis'].enqueue_job (worker.py's deferred graph
        enrichment) must land on the runner — the in-process equivalent of
        the dedicated graph queue."""
        inline_tasks, build = _make_runner()

        async def scenario():
            runner = await build()
            ran: list[str] = []

            async def fast(ctx):
                job = await ctx["redis"].enqueue_job(
                    "process_graph_enrichment", "mem-1", _job_id="deferred-1"
                )
                return {"deferred": job.job_id}

            async def enrich(ctx, memory_id):
                ran.append(memory_id)
                return {"enriched": True}

            _stub_worker(runner, process_memory_raw=fast, process_graph_enrichment=enrich)
            runner.submit("process_memory_raw", (), {}, "f1")
            await runner.wait("f1", timeout=5)
            deferred = await runner.wait("deferred-1", timeout=5)
            return ran, deferred["status"]

        ran, status = asyncio.run(scenario())
        assert ran == ["mem-1"] and status == "completed"

    def test_eviction_never_drops_live_tasks(self):
        import inline_tasks

        async def scenario():
            runner = inline_tasks.InlineTaskRunner(service=object())
            gate = asyncio.Event()

            async def waits(ctx):
                await gate.wait()
                return {}

            async def ok(ctx):
                return {}

            _stub_worker(runner, process_memory_raw=waits, process_memory_retag=ok)
            monkey_cap = 5
            orig = inline_tasks._MAX_FINISHED_TASKS
            inline_tasks._MAX_FINISHED_TASKS = monkey_cap
            try:
                runner.submit("process_memory_raw", (), {}, "live")
                for i in range(monkey_cap + 3):
                    runner.submit("process_memory_retag", (), {}, f"r{i}")
                    await runner.wait(f"r{i}", timeout=5)
                assert "live" in runner._tasks  # never evicted while queued
                gate.set()
                got = await runner.wait("live", timeout=5)
                return got["status"], len(runner._tasks) <= monkey_cap + 2
            finally:
                inline_tasks._MAX_FINISHED_TASKS = orig

        status, bounded = asyncio.run(scenario())
        assert status == "completed" and bounded


class TestInlineTaskManagerE2E:
    def test_enqueue_poll_roundtrip_with_user_tracking(self, monkeypatch):
        from config import settings as cfg

        monkeypatch.setattr(cfg, "task_backend", "inline")
        from task_manager import create_task_manager

        async def scenario():
            tm = create_task_manager()
            tm.bind(service=object())

            async def raw(ctx, content, user_id, category, *a, **kw):
                return {"memories": [{"memory": content, "category": category}]}

            _stub_worker(tm._runner, process_memory_raw=raw)
            task_id = await tm.enqueue_raw("solo fact", "solo-user", "preference")
            got = await tm.wait_for_result(task_id, timeout=5)
            queue = await tm.get_queue_status("solo-user")
            missing = await tm.get_status("nope")
            return got, queue, missing

        got, queue, missing = asyncio.run(scenario())
        assert got["status"] == "completed"
        assert got["result"]["memories"][0]["memory"] == "solo fact"
        assert queue["tracked"] == 1 and queue["caught_up"] is True
        assert missing["status"] == "not_found"

    def test_unbound_manager_keeps_sync_fallback_contract(self, monkeypatch):
        from config import settings as cfg

        monkeypatch.setattr(cfg, "task_backend", "inline")
        from task_manager import create_task_manager

        async def scenario():
            tm = create_task_manager()
            await tm.connect()  # no bind — pool stays disabled
            await tm.enqueue_raw("x", "u", "preference")

        with pytest.raises(ConnectionError):
            asyncio.run(scenario())
