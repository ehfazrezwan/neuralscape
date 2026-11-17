"""The dreaming integration fixtures name the call that timed out (unit, no stores)."""

from __future__ import annotations

import pytest

from types import SimpleNamespace

from integration_tests import _dreaming_support as sup
from integration_tests._dreaming_support import DreamEnv, _is_timeout, timed


def _env():
    return DreamEnv(service=None, collection="c", user="u", vault=None, redis=None)


class ReadTimeout(Exception):
    pass


def test_timeout_names_the_call_and_the_elapsed_time():
    env = _env()
    env.timings.append(("earlier ok", 0.4))
    with pytest.raises(pytest.fail.Exception) as exc:
        with timed(env, "store_raw #3 (workflow)"):
            raise RuntimeError("wrapped") from ReadTimeout("timed out")
    msg = str(exc.value)
    assert "store_raw #3 (workflow) timed out after" in msg and "earlier ok 0.40s" in msg


def test_other_errors_pass_through_untouched_and_are_timed():
    env = _env()
    with pytest.raises(ValueError):
        with timed(env, "x"):
            raise ValueError("boom")
    assert env.timings and env.timings[-1][0] == "x"


def test_timeout_found_through_context_and_cause_chains():
    try:
        try:
            raise ReadTimeout("x")
        except ReadTimeout:
            raise RuntimeError("during handling")  # sets __context__ only
    except RuntimeError as exc:
        assert exc.__cause__ is None and _is_timeout(exc)
    assert not _is_timeout(ValueError("no"))


def test_failure_keeps_the_original_exception():
    env = _env()
    with pytest.raises(pytest.fail.Exception) as exc:
        with timed(env, "c"):
            raise ReadTimeout("timed out")
    assert isinstance(exc.value.__cause__, ReadTimeout)


# ── _wait_collection_ready: retry on timeout, honour the deadline ─────


def _ready_env(get_collection):
    env = _env()
    env.collection = "c"
    env.service = SimpleNamespace(
        _memory=SimpleNamespace(vector_store=SimpleNamespace(client=SimpleNamespace(get_collection=get_collection)))
    )
    return env


def _fast_clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(sup.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(sup.time, "sleep", lambda d: now.__setitem__(0, now[0] + d))
    return now


def test_wait_ready_retries_a_timed_out_poll_until_green(monkeypatch):
    _fast_clock(monkeypatch)
    calls = []

    def get_collection(name):
        calls.append(name)
        if len(calls) < 3:
            raise ReadTimeout("timed out")
        return SimpleNamespace(status="green")

    sup._wait_collection_ready(_ready_env(get_collection), deadline_s=30.0)
    assert len(calls) == 3


def test_wait_ready_keeps_polling_a_non_green_status(monkeypatch):
    _fast_clock(monkeypatch)
    seq = iter(["yellow", "yellow", "green"])
    sup._wait_collection_ready(_ready_env(lambda n: SimpleNamespace(status=next(seq))), deadline_s=30.0)


def test_wait_ready_gives_up_at_the_deadline_and_says_why(monkeypatch):
    now = _fast_clock(monkeypatch)
    calls = []

    def get_collection(name):
        calls.append(1)
        raise ReadTimeout("timed out")

    with pytest.raises(pytest.fail.Exception) as exc:
        sup._wait_collection_ready(_ready_env(get_collection), deadline_s=5.0)
    assert "not green after 5s" in str(exc.value) and "ReadTimeout" in str(exc.value)
    assert 4 <= len(calls) <= 6 and now[0] < 8.0  # bounded by the deadline, not unbounded
