"""Retry and negative-control helpers."""

from __future__ import annotations

import time
from typing import Any, Callable


def eventually(
    fn: Callable[[], Any],
    timeout: float = 90.0,
    interval: float = 2.0,
    what: str = "condition",
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Any:
    """Call ``fn`` until it returns without raising AssertionError.

    Use for async effects (queue -> worker -> index). ``fn`` signals "not yet"
    by raising AssertionError; any return value (even a falsy one) counts as
    success and is returned. On timeout raises AssertionError carrying the last failure
    message. Other exceptions (connection errors, bugs) propagate immediately.
    """
    deadline = clock() + timeout
    last: AssertionError | None = None
    attempts = 0
    while True:
        attempts += 1
        try:
            return fn()
        except AssertionError as exc:  # not there yet
            last = exc
        if clock() >= deadline:
            break
        sleep(interval)
    raise AssertionError(
        f"{what} not satisfied after {timeout:.0f}s ({attempts} attempts): {last}"
    )


def expect_check_fails(check: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Negative control: pass only if ``check(*args, **kwargs)`` raises AssertionError.

    Run the family's REAL check with a deliberately wrong input (another
    user's token, a wrong-secret token, ...). If the check still passes, the
    check cannot detect the failure it claims to detect, and this raises.
    Non-assertion errors (connection refused, bugs) propagate: they say
    nothing about whether the check can fail.
    """
    try:
        check(*args, **kwargs)
    except AssertionError:
        return
    raise AssertionError(
        "negative control: check passed with a deliberately wrong input, "
        "so it cannot fail and proves nothing"
    )
