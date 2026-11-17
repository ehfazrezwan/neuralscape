"""A raw batch holding several users' items is not billed to any one of them.

The enqueue-time ``task -> user`` record keeps the last writer only, so a mixed-user batch used to be
attributed to the alphabetically last user.
"""

import asyncio
from types import SimpleNamespace

import pytest

import usage_capture as uc
import worker


@pytest.fixture()
def recorder():
    rec = uc.InMemoryUsageRecorder()
    previous = uc.set_usage_recorder(rec)
    yield rec
    uc.set_usage_recorder(previous)


class Redis:
    def __init__(self, mapping):
        self.mapping = mapping

    async def get(self, key):
        return self.mapping.get(key)


def _bound_tenant(items, *, function="process_memory_raw_batch"):
    ctx = {"job_id": "job-1", "job_try": 1, "redis": Redis({"ns:task-user:job-1": b"zed"})}
    info = SimpleNamespace(function=function, args=(items,))

    async def run():
        await worker._bind_usage_attempt(ctx, info)
        return uc.current_attempt_context()

    return asyncio.run(run()).tenant_id


def test_a_mixed_user_raw_batch_is_recorded_as_unattributed(recorder):
    items = [{"user_id": "alice", "content": "a"}, {"user_id": "zed", "content": "z"}]
    assert _bound_tenant(items) == uc.UNATTRIBUTED_TENANT


def test_a_single_user_raw_batch_is_billed_to_that_user(recorder):
    items = [{"user_id": "zed", "content": "a"}, {"user_id": "zed", "content": "b"}]
    assert _bound_tenant(items) == "zed"


def test_another_job_keeps_the_enqueue_time_owner_whatever_its_args(recorder):
    items = [{"user_id": "alice"}, {"user_id": "bob"}]
    assert _bound_tenant(items, function="process_memory_store") == "zed"


def test_the_start_hook_hands_the_job_record_to_the_attribution(recorder, monkeypatch):
    seen = {}

    async def fake_bind(ctx, info=None):
        seen["info"] = info

    info = SimpleNamespace(function="process_memory_raw_batch", args=([],))

    class FakeJob:
        def __init__(self, *a, **kw):
            pass

        async def info(self):
            return info

    monkeypatch.setattr(worker, "Job", FakeJob)
    monkeypatch.setattr(worker, "_bind_usage_attempt", fake_bind)
    asyncio.run(worker._make_on_job_start("q")({"job_id": "j", "redis": object()}))
    assert seen["info"] is info
