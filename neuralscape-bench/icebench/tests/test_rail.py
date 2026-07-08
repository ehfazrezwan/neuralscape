"""Tests for safety rail."""

import time
import pytest

from icebench.rail import (
    RailConfig,
    run_with_rail,
    run_with_rail_fn,
    _has_systemd_run,
)


def test_rail_success():
    """Test successful command execution."""
    config = RailConfig(memory_limit_mb=1024, timeout_seconds=10)

    result = run_with_rail(["echo", "hello"], config)

    assert result.returncode == 0
    assert "hello" in result.stdout
    assert not result.timed_out
    assert not result.oom_killed
    assert not result.dnf


def test_rail_timeout():
    """Test timeout detection."""
    config = RailConfig(memory_limit_mb=1024, timeout_seconds=1)

    result = run_with_rail(["sleep", "10"], config)

    assert result.timed_out
    assert result.dnf
    assert result.dnf_reason == "timeout"


def test_rail_fn_success():
    """Test function wrapper success."""
    config = RailConfig(timeout_seconds=10)

    def success_fn():
        return True, "success"

    result = run_with_rail_fn(success_fn, config)

    assert result.returncode == 0
    assert result.stdout == "success"
    assert not result.timed_out


def test_rail_fn_timeout():
    """Test function wrapper timeout."""
    config = RailConfig(timeout_seconds=1)

    def slow_fn():
        time.sleep(10)
        return True, "done"

    result = run_with_rail_fn(slow_fn, config)

    assert result.timed_out
    assert result.dnf


def test_rail_fn_exception():
    """Test function wrapper exception handling."""
    config = RailConfig(timeout_seconds=10)

    def error_fn():
        raise ValueError("test error")

    result = run_with_rail_fn(error_fn, config)

    assert result.returncode == -1
    assert "test error" in result.stderr
    assert not result.timed_out


def test_systemd_detection():
    """Test systemd-run detection."""
    # Just verify it doesn't crash
    has_systemd = _has_systemd_run()
    assert isinstance(has_systemd, bool)
