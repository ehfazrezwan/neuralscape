"""
Safety rail for running indexing operations under resource limits.

Runs every indexing step under a hard memory cap + wall-clock timeout AND
measures resources (peak RSS + CPU-seconds) via GNU `/usr/bin/time -v`.

Memory cap mechanism, in preference order:
  1. `systemd-run --scope -p MemoryMax=<cap>` (a true cgroup cap → OOM-kill).
  2. `ulimit -v <cap>` in a child shell (address-space cap → alloc failure).

Breach (OOM-kill or wall-timeout) => RailResult.dnf is True with a reason and
the cap. NEVER raises — a competitor blowup becomes DATA, not a VM outage.
"""

import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass


# GNU time binary (busybox `time` lacks -v, so we require the coreutils one)
TIME_BIN = "/usr/bin/time"

# Substrings that indicate systemd-run ITSELF failed to launch (vs the child
# process failing) — used to fall back to the ulimit mechanism.
_SYSTEMD_SELF_FAILURE = (
    "Unknown assignment",
    "Failed to connect to bus",
    "Interactive authentication required",
    "Failed to start transient scope unit",
    "Failed to create bus connection",
)


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
    peak_rss_mb: float
    cpu_s: float
    timed_out: bool
    oom_killed: bool
    memory_cap_mb: int
    timeout_s: int
    mechanism: str  # "systemd" | "ulimit" | "none"

    @property
    def ok(self) -> bool:
        """True if the command finished successfully and did not breach a rail."""
        return self.returncode == 0 and not self.dnf

    @property
    def dnf(self) -> bool:
        """Did Not Finish (timeout or OOM)."""
        return self.timed_out or self.oom_killed

    @property
    def dnf_reason(self) -> str | None:
        """Reason for DNF, including the cap that was breached."""
        if self.timed_out:
            return f"timeout>{self.timeout_s}s"
        if self.oom_killed:
            return f"oom>MemoryMax={self.memory_cap_mb}MB"
        return None


def _has(binary: str) -> bool:
    """Check if a binary is on PATH."""
    return shutil.which(binary) is not None


def _has_systemd_run() -> bool:
    """Check if systemd-run is available."""
    return _has("systemd-run")


def _has_gnu_time() -> bool:
    """Check if GNU /usr/bin/time is available."""
    return Path(TIME_BIN).exists()


def _parse_time_file(path: str) -> tuple[float, float, int | None]:
    """
    Parse a GNU `time -v` output file.

    Args:
        path: Path to the time output file.

    Returns:
        (peak_rss_mb, cpu_s, killing_signal_or_None)
    """
    peak_rss_mb = 0.0
    cpu_s = 0.0
    killing_signal: int | None = None

    try:
        text = Path(path).read_text()
    except OSError:
        return peak_rss_mb, cpu_s, killing_signal

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Maximum resident set size"):
            m = re.search(r":\s*(\d+)", line)
            if m:
                # GNU time reports kbytes
                peak_rss_mb = int(m.group(1)) / 1024.0
        elif line.startswith("User time"):
            m = re.search(r":\s*([\d.]+)", line)
            if m:
                cpu_s += float(m.group(1))
        elif line.startswith("System time"):
            m = re.search(r":\s*([\d.]+)", line)
            if m:
                cpu_s += float(m.group(1))
        elif "terminated by signal" in line:
            m = re.search(r"signal (\d+)", line)
            if m:
                killing_signal = int(m.group(1))

    return peak_rss_mb, cpu_s, killing_signal


def _wrap_with_time(cmd: list[str], time_file: str | None) -> list[str]:
    """Wrap a command with `/usr/bin/time -v -o <file>` if GNU time is present."""
    if time_file is not None and _has_gnu_time():
        return [TIME_BIN, "-v", "-o", time_file] + cmd
    return list(cmd)


def _systemd_wrap(inner: list[str], memory_bytes: int) -> list[str]:
    """Wrap a command in a memory-capped transient systemd scope."""
    return [
        "systemd-run",
        "--scope",
        "--user",
        "--quiet",
        "-p",
        f"MemoryMax={memory_bytes}",
        "--",
    ] + inner


def _ulimit_wrap(inner: list[str], memory_kb: int) -> list[str]:
    """Wrap a command in a shell that caps address space via ulimit -v."""
    quoted = " ".join(shlex.quote(a) for a in inner)
    return ["bash", "-c", f"ulimit -v {memory_kb}; exec {quoted}"]


def _looks_like_systemd_self_failure(returncode: int, stderr: str) -> bool:
    """Detect systemd-run's own launch failure (not the child failing)."""
    if returncode == 0:
        return False
    return any(marker in stderr for marker in _SYSTEMD_SELF_FAILURE)


def run_with_rail(
    cmd: list[str],
    config: RailConfig,
    cwd: Path | None = None,
    env: dict | None = None,
) -> RailResult:
    """
    Run a command under the safety rail: hard memory cap + wall timeout, with
    resource measurement.

    Prefers a systemd cgroup cap; falls back to `ulimit -v`. Resource metrics
    (peak RSS, CPU-seconds) come from a `/usr/bin/time -v` wrapper. Never raises
    — a breach is reported via RailResult.dnf / dnf_reason.

    Args:
        cmd: Command to run (argv list).
        config: Rail configuration (cap + timeout).
        cwd: Working directory.
        env: Environment variables.

    Returns:
        RailResult with timing, resources, and breach status.
    """
    memory_bytes = config.memory_limit_mb * 1024 * 1024
    memory_kb = config.memory_limit_mb * 1024

    # A temp file for GNU time -v output (keeps the child's own stderr clean).
    time_file: str | None = None
    if _has_gnu_time():
        tf = tempfile.NamedTemporaryFile(
            mode="r", suffix=".ice-time", delete=False
        )
        tf.close()
        time_file = tf.name

    try:
        inner = _wrap_with_time(cmd, time_file)

        if _has_systemd_run():
            mechanism = "systemd"
            full = _systemd_wrap(inner, memory_bytes)
        elif _has("bash"):
            mechanism = "ulimit"
            full = _ulimit_wrap(inner, memory_kb)
        else:
            mechanism = "none"
            full = inner

        result = _execute(
            full, config, cwd, env, memory_bytes, memory_kb, time_file, mechanism
        )

        # If systemd-run itself failed to launch, retry under ulimit.
        if (
            mechanism == "systemd"
            and _looks_like_systemd_self_failure(result.returncode, result.stderr)
            and _has("bash")
        ):
            fallback = _ulimit_wrap(inner, memory_kb)
            result = _execute(
                fallback, config, cwd, env, memory_bytes, memory_kb, time_file, "ulimit"
            )

        return result
    finally:
        if time_file is not None:
            try:
                Path(time_file).unlink()
            except OSError:
                pass


def _execute(
    full_cmd: list[str],
    config: RailConfig,
    cwd: Path | None,
    env: dict | None,
    memory_bytes: int,
    memory_kb: int,
    time_file: str | None,
    mechanism: str,
) -> RailResult:
    """Execute a fully-wrapped command, measure resources, classify breaches."""
    start = time.monotonic()
    try:
        proc = subprocess.run(
            full_cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
        )
        wall_s = time.monotonic() - start

        peak_rss_mb, cpu_s, killing_signal = (0.0, 0.0, None)
        if time_file is not None:
            peak_rss_mb, cpu_s, killing_signal = _parse_time_file(time_file)

        # OOM classification: SIGKILL (137 / -9) or time reported signal 9.
        oom_killed = proc.returncode in (137, -9) or killing_signal == 9

        return RailResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            wall_s=wall_s,
            peak_rss_mb=peak_rss_mb,
            cpu_s=cpu_s,
            timed_out=False,
            oom_killed=oom_killed,
            memory_cap_mb=config.memory_limit_mb,
            timeout_s=config.timeout_seconds,
            mechanism=mechanism,
        )

    except subprocess.TimeoutExpired as e:
        wall_s = time.monotonic() - start
        peak_rss_mb, cpu_s, _ = (0.0, 0.0, None)
        if time_file is not None:
            peak_rss_mb, cpu_s, _ = _parse_time_file(time_file)

        return RailResult(
            returncode=-1,
            stdout=e.stdout or "" if isinstance(e.stdout, str) else "",
            stderr=e.stderr or "" if isinstance(e.stderr, str) else "",
            wall_s=wall_s,
            peak_rss_mb=peak_rss_mb,
            cpu_s=cpu_s,
            timed_out=True,
            oom_killed=False,
            memory_cap_mb=config.memory_limit_mb,
            timeout_s=config.timeout_seconds,
            mechanism=mechanism,
        )
