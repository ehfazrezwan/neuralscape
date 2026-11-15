"""``_AsyncBridge.run`` is bounded (fork patch in mem0 graphiti_memory.py).

Upstream waited on ``future.result()`` forever, so a stalled Neo4j/LLM call
parked the calling worker slot for good.  Every test here guards itself with a
hard wall of its own: a regression must fail, never hang the suite.
"""

import asyncio
import threading
import time

import pytest

from mem0.memory import graphiti_memory
from mem0.memory.graphiti_memory import DEFAULT_BRIDGE_TIMEOUT_S, _AsyncBridge

GUARD_S = 20  # how long the test itself waits before declaring "hung"


def _run_guarded(bridge, coro, **kwargs):
    """Run ``bridge.run`` in a thread; return (outcome, elapsed) or raise AssertionError if it hangs."""
    box = {}

    def target():
        started = time.monotonic()
        try:
            box["value"] = bridge.run(coro, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc
        box["elapsed"] = time.monotonic() - started

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(GUARD_S)
    assert not t.is_alive(), f"bridge.run still blocked after {GUARD_S}s: the wait is unbounded"
    return box


@pytest.fixture
def bridge():
    b = _AsyncBridge()
    yield b
    b._loop.call_soon_threadsafe(b._loop.stop)


def test_a_coroutine_that_never_finishes_raises_timeout_error_within_the_bound(bridge):
    cancelled = threading.Event()

    async def never():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    box = _run_guarded(bridge, never(), timeout=0.3)
    assert isinstance(box.get("error"), TimeoutError), box
    assert 0.25 <= box["elapsed"] < 5
    assert cancelled.wait(5), "the stuck coroutine was not cancelled on the loop"


def test_the_loop_stays_usable_after_a_timeout(bridge):
    async def never():
        await asyncio.sleep(3600)

    async def fine():
        return 42

    assert isinstance(_run_guarded(bridge, never(), timeout=0.2).get("error"), TimeoutError)
    assert _run_guarded(bridge, fine(), timeout=5).get("value") == 42


def test_a_finishing_coroutine_returns_its_value_and_errors_pass_through(bridge):
    async def ok():
        return "done"

    async def boom():
        raise ValueError("graph down")

    assert _run_guarded(bridge, ok()).get("value") == "done"
    assert isinstance(_run_guarded(bridge, boom()).get("error"), ValueError)


def test_default_bound_comes_from_env_and_is_finite(bridge, monkeypatch):
    assert 0 < DEFAULT_BRIDGE_TIMEOUT_S < float("inf")
    monkeypatch.setenv("GRAPHITI_BRIDGE_TIMEOUT_S", "0.3")

    async def never():
        await asyncio.sleep(3600)

    box = _run_guarded(bridge, never())  # no explicit timeout: env bound applies
    assert isinstance(box.get("error"), TimeoutError)
    assert box["elapsed"] < 5


@pytest.mark.parametrize("raw", ["", "abc", "0", "-5"])
def test_bad_env_values_fall_back_to_the_default(monkeypatch, raw):
    monkeypatch.setenv("GRAPHITI_BRIDGE_TIMEOUT_S", raw)
    assert graphiti_memory._bridge_timeout_from_env() == DEFAULT_BRIDGE_TIMEOUT_S


def test_timeout_message_does_not_trigger_the_service_transient_retry():
    """The service's retry_transient re-runs on 'timeout'/'timed out'; re-running a hung call is wrong."""
    from memory.retry import _is_transient

    b = _AsyncBridge()
    try:
        async def never():
            await asyncio.sleep(3600)

        err = _run_guarded(b, never(), timeout=0.1)["error"]
    finally:
        b._loop.call_soon_threadsafe(b._loop.stop)
    assert not _is_transient(err)


def test_a_timeout_raised_inside_the_coroutine_propagates_unchanged(bridge):
    """Only the outer wait expiring is a bridge timeout; an inner TimeoutError keeps its message."""
    from memory.retry import _is_transient

    async def inner():
        raise TimeoutError("neo4j read timed out after 5s")

    err = _run_guarded(bridge, inner(), timeout=5)["error"]
    assert isinstance(err, TimeoutError)
    assert str(err) == "neo4j read timed out after 5s"
    assert _is_transient(err)  # the service still retries it as before


def test_default_bound_is_not_shorter_than_the_longest_worker_job_timeout():
    import re
    from pathlib import Path

    worker = (Path(__file__).resolve().parents[1] / "worker.py").read_text()
    longest = max(int(n) for n in re.findall(r"job_timeout = (\d+)", worker))
    assert DEFAULT_BRIDGE_TIMEOUT_S > longest
