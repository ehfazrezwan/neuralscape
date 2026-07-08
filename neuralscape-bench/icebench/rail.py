"""
Safety rail for running indexing operations under resource limits.

Runs every indexing step under a hard memory cap + wall-clock timeout.
Breach (OOM-kill or timeout) => DNF, not a crash.
"""

import subprocess
import shutil
import time
import signal
from pathlib import Path
from dataclasses import dataclass
from typing import Callable


@dataclass
class RailConfig:
    """Configuration for the safety rail."""

    memory_limit_mb: int = 12 * 1024  # 12 GB
    timeout_seconds: int = 3600  # 1 hour


@dataclass
class RailResult:
    """Result from a rail-protected execution."""

    returncode: int
    stdout: str
    stderr: str
    wall_s: float
    timed_out: bool
    oom_killed: bool

    @property
    def dnf(self) -> bool:
        """Did Not Finish (timeout or OOM)."""
        return self.timed_out or self.oom_killed

    @property
    def dnf_reason(self) -> str | None:
        """Reason for DNF."""
        if self.timed_out:
            return "timeout"
        if self.oom_killed:
            return "oom"
        return None


def _has_systemd_run() -> bool:
    """Check if systemd-run is available."""
    return shutil.which("systemd-run") is not None


def run_with_rail(
    cmd: list[str],
    config: RailConfig,
    cwd: Path | None = None,
    env: dict | None = None,
) -> RailResult:
    """
    Run a command under the safety rail.

    Prefers systemd-run if available, otherwise uses subprocess with timeout.

    Args:
        cmd: Command to run.
        config: Rail configuration.
        cwd: Working directory.
        env: Environment variables.

    Returns:
        RailResult with timing and status.
    """
    if _has_systemd_run():
        return _run_with_systemd(cmd, config, cwd, env)
    else:
        return _run_with_subprocess(cmd, config, cwd, env)


def _run_with_systemd(
    cmd: list[str],
    config: RailConfig,
    cwd: Path | None = None,
    env: dict | None = None,
) -> RailResult:
    """
    Run with systemd-run for memory limiting.

    Args:
        cmd: Command to run.
        config: Rail configuration.
        cwd: Working directory.
        env: Environment variables.

    Returns:
        RailResult.
    """
    # Build systemd-run command with memory limit
    memory_bytes = config.memory_limit_mb * 1024 * 1024
    systemd_cmd = [
        "systemd-run",
        "--scope",
        "--user",
        f"-p MemoryMax={memory_bytes}",
        "--quiet",
        "--",
    ] + cmd

    start = time.time()
    try:
        result = subprocess.run(
            systemd_cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
        )
        wall_s = time.time() - start

        # If systemd-run itself failed (e.g., unsupported flags), fall back
        if result.returncode != 0 and "Unknown assignment" in result.stderr:
            # Fallback to subprocess without systemd
            return _run_with_subprocess(cmd, config, cwd, env)

        # Check if OOM killed (exit code 137 or SIGKILL)
        oom_killed = result.returncode in (137, -9)

        return RailResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            wall_s=wall_s,
            timed_out=False,
            oom_killed=oom_killed,
        )

    except subprocess.TimeoutExpired as e:
        wall_s = time.time() - start
        return RailResult(
            returncode=-1,
            stdout=e.stdout or "",
            stderr=e.stderr or "",
            wall_s=wall_s,
            timed_out=True,
            oom_killed=False,
        )


def _run_with_subprocess(
    cmd: list[str],
    config: RailConfig,
    cwd: Path | None = None,
    env: dict | None = None,
) -> RailResult:
    """
    Fallback: run with subprocess timeout (no memory limiting).

    Args:
        cmd: Command to run.
        config: Rail configuration.
        cwd: Working directory.
        env: Environment variables.

    Returns:
        RailResult.
    """
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
        )
        wall_s = time.time() - start

        return RailResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            wall_s=wall_s,
            timed_out=False,
            oom_killed=False,
        )

    except subprocess.TimeoutExpired as e:
        wall_s = time.time() - start
        return RailResult(
            returncode=-1,
            stdout=e.stdout or "",
            stderr=e.stderr or "",
            wall_s=wall_s,
            timed_out=True,
            oom_killed=False,
        )


def run_with_rail_fn(
    fn: Callable[[], tuple[bool, str]],
    config: RailConfig,
) -> RailResult:
    """
    Run a Python function under the safety rail (timeout only).

    Note: Memory limiting requires running in a subprocess, so this
    only provides timeout protection.

    Args:
        fn: Function to run. Should return (success, message).
        config: Rail configuration.

    Returns:
        RailResult.
    """

    def _timeout_handler(signum, frame):
        raise TimeoutError("Function timed out")

    start = time.time()
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(config.timeout_seconds)

    try:
        ok, msg = fn()
        wall_s = time.time() - start
        signal.alarm(0)  # Cancel alarm

        return RailResult(
            returncode=0 if ok else 1,
            stdout=msg,
            stderr="",
            wall_s=wall_s,
            timed_out=False,
            oom_killed=False,
        )

    except TimeoutError:
        wall_s = time.time() - start
        return RailResult(
            returncode=-1,
            stdout="",
            stderr="Function timed out",
            wall_s=wall_s,
            timed_out=True,
            oom_killed=False,
        )

    except Exception as e:
        wall_s = time.time() - start
        return RailResult(
            returncode=-1,
            stdout="",
            stderr=str(e),
            wall_s=wall_s,
            timed_out=False,
            oom_killed=False,
        )

    finally:
        signal.signal(signal.SIGALRM, old_handler)
