"""Tests for the safety rail: enforcement, DNF classification, resource parsing.

These tests never actually OOM the box — the subprocess layer is mocked for the
breach paths, and only lightweight real commands are exercised.
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from icebench.rail import (
    RailConfig,
    run_with_rail,
    _parse_time_file,
    _has_systemd_run,
    _has_gnu_time,
    _systemd_wrap,
    _ulimit_wrap,
)


SAMPLE_TIME_V = """\tCommand being timed: "python -m foo"
\tUser time (seconds): 3.50
\tSystem time (seconds): 1.25
\tPercent of CPU this job got: 95%
\tMaximum resident set size (kbytes): 2097152
\tExit status: 0
"""

SAMPLE_TIME_V_KILLED = """\tCommand being timed: "python -m hog"
\tUser time (seconds): 0.10
\tSystem time (seconds): 0.20
\tMaximum resident set size (kbytes): 12582912
\tCommand terminated by signal 9
"""


def test_parse_time_file():
    """GNU time -v output parses into peak RSS (MB) + CPU seconds."""
    with tempfile.NamedTemporaryFile("w", suffix=".time", delete=False) as f:
        f.write(SAMPLE_TIME_V)
        path = f.name

    peak_rss_mb, cpu_s, killing_signal = _parse_time_file(path)
    # 2097152 kbytes / 1024 = 2048 MB
    assert peak_rss_mb == pytest.approx(2048.0)
    # 3.50 + 1.25 = 4.75
    assert cpu_s == pytest.approx(4.75)
    assert killing_signal is None
    Path(path).unlink()


def test_parse_time_file_killed():
    """time -v records the killing signal for an OOM-killed child."""
    with tempfile.NamedTemporaryFile("w", suffix=".time", delete=False) as f:
        f.write(SAMPLE_TIME_V_KILLED)
        path = f.name

    peak_rss_mb, cpu_s, killing_signal = _parse_time_file(path)
    assert killing_signal == 9
    assert peak_rss_mb == pytest.approx(12288.0)
    Path(path).unlink()


def test_command_wrappers():
    """Wrapper builders produce correctly separated argv tokens."""
    systemd = _systemd_wrap(["echo", "hi"], 1024 * 1024 * 1024)
    # MemoryMax must be its OWN token (the original bug was "-p MemoryMax=..").
    assert "-p" in systemd
    assert "MemoryMax=1073741824" in systemd
    assert systemd[-2:] == ["echo", "hi"]

    ulimit = _ulimit_wrap(["echo", "hi"], 1024)
    assert ulimit[0] == "bash"
    assert "ulimit -v 1024" in ulimit[2]


def _fake_completed(returncode=0, stdout="ok", stderr=""):
    return subprocess.CompletedProcess(
        args=["x"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_rail_success_mocked():
    """A clean run reports ok + the mechanism, no DNF."""
    cfg = RailConfig(memory_limit_mb=1024, timeout_seconds=10)

    with patch("icebench.rail._has_systemd_run", return_value=False), patch(
        "icebench.rail._has_gnu_time", return_value=False
    ), patch("icebench.rail._has", return_value=True), patch(
        "icebench.rail.subprocess.run", return_value=_fake_completed(0, "done", "")
    ):
        res = run_with_rail(["echo", "done"], cfg)

    assert res.ok
    assert res.returncode == 0
    assert not res.dnf
    assert res.mechanism == "ulimit"
    assert res.memory_cap_mb == 1024


def test_rail_timeout_dnf_mocked():
    """A wall-timeout is DNF with a timeout reason and the cap — never raises."""
    cfg = RailConfig(memory_limit_mb=1024, timeout_seconds=1)

    with patch("icebench.rail._has_systemd_run", return_value=False), patch(
        "icebench.rail._has_gnu_time", return_value=False
    ), patch("icebench.rail._has", return_value=True), patch(
        "icebench.rail.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1),
    ):
        res = run_with_rail(["sleep", "10"], cfg)

    assert res.timed_out
    assert res.dnf
    assert res.dnf_reason == "timeout>1s"


def test_rail_oom_dnf_mocked():
    """A SIGKILL (137) under the cap is classified as OOM DNF with the cap."""
    cfg = RailConfig(memory_limit_mb=12288, timeout_seconds=60)

    with patch("icebench.rail._has_systemd_run", return_value=True), patch(
        "icebench.rail._has_gnu_time", return_value=False
    ), patch("icebench.rail._has", return_value=True), patch(
        "icebench.rail.subprocess.run",
        return_value=_fake_completed(137, "", ""),
    ):
        res = run_with_rail(["python", "-m", "hog"], cfg)

    assert res.oom_killed
    assert res.dnf
    assert res.dnf_reason == "oom>MemoryMax=12288MB"


def test_rail_resource_metrics_from_time_file():
    """peak_rss_mb + cpu_s come from the /usr/bin/time -v -o file."""
    cfg = RailConfig(memory_limit_mb=4096, timeout_seconds=60)

    def _run_side_effect(cmd, **kwargs):
        # The command is wrapped as [.., "-o", <file>, ..]; write a time file.
        assert "-o" in cmd
        time_file = cmd[cmd.index("-o") + 1]
        Path(time_file).write_text(SAMPLE_TIME_V)
        return _fake_completed(0, '{"symbols": 1}', "")

    # No systemd + no bash => mechanism "none", so time's "-o" stays a distinct
    # argv token (the ulimit path would embed it in a bash -c string).
    with patch("icebench.rail._has_systemd_run", return_value=False), patch(
        "icebench.rail._has_gnu_time", return_value=True
    ), patch("icebench.rail._has", return_value=False), patch(
        "icebench.rail.subprocess.run", side_effect=_run_side_effect
    ):
        res = run_with_rail(["python", "-m", "foo"], cfg)

    assert res.ok
    assert res.mechanism == "none"
    assert res.peak_rss_mb == pytest.approx(2048.0)
    assert res.cpu_s == pytest.approx(4.75)


def test_rail_systemd_self_failure_falls_back():
    """If systemd-run itself fails to launch, the rail retries under ulimit."""
    cfg = RailConfig(memory_limit_mb=2048, timeout_seconds=30)

    calls = {"n": 0}

    def _run_side_effect(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # systemd-run self-failure signature.
            return _fake_completed(1, "", "Failed to connect to bus")
        return _fake_completed(0, "ok", "")

    with patch("icebench.rail._has_systemd_run", return_value=True), patch(
        "icebench.rail._has_gnu_time", return_value=False
    ), patch("icebench.rail._has", return_value=True), patch(
        "icebench.rail.subprocess.run", side_effect=_run_side_effect
    ):
        res = run_with_rail(["python", "-m", "foo"], cfg)

    assert calls["n"] == 2  # retried
    assert res.ok
    assert res.mechanism == "ulimit"


def test_detection_helpers_no_crash():
    """Availability probes return bools without raising."""
    assert isinstance(_has_systemd_run(), bool)
    assert isinstance(_has_gnu_time(), bool)
