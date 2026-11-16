"""The dreaming integration fixtures name the call that timed out (unit, no stores)."""

from __future__ import annotations

import pytest

from integration_tests._dreaming_support import DreamEnv, timed


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
