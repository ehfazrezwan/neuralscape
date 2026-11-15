"""Pin the cold-start backoff schedule of ``memory.core._memory_from_config``.

The comment in core.py claims 8 attempts and 2.8 s of total backoff (sleeps of
0.1 * attempt after attempts 1..7).  Nothing pinned that, so a change to the
constants would pass the suite while the comment lied.
"""

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from memory import core


def _conflict():
    return UnexpectedResponse(
        status_code=409,
        reason_phrase="Conflict",
        content=b"Collection `x` already exists!",
        headers=httpx.Headers(),
    )


def test_backoff_schedule_is_eight_attempts_with_linear_sleeps(monkeypatch):
    sleeps, calls = [], []
    monkeypatch.setattr(core.time, "sleep", sleeps.append)

    class AlwaysConflicts:
        @staticmethod
        def from_config(config):
            calls.append(1)
            raise _conflict()

    with pytest.raises(UnexpectedResponse):
        core._memory_from_config(AlwaysConflicts, {})

    assert core._COLD_START_ATTEMPTS == 8
    assert len(calls) == 8
    assert sleeps == pytest.approx([0.1 * n for n in range(1, 8)])  # no sleep after the last attempt
    assert sum(sleeps) == pytest.approx(2.8)


def test_success_on_a_later_attempt_stops_sleeping(monkeypatch):
    sleeps = []
    monkeypatch.setattr(core.time, "sleep", sleeps.append)
    attempts = []

    class ClearsOnThird:
        @staticmethod
        def from_config(config):
            attempts.append(1)
            if len(attempts) < 3:
                raise _conflict()
            return "memory"

    assert core._memory_from_config(ClearsOnThird, {}) == "memory"
    assert sleeps == pytest.approx([0.1, 0.2])
