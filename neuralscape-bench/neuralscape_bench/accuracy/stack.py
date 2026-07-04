"""Isolated NS stack management for accuracy runs.

Reuses the latency harness's isolation pattern: a dedicated docker compose
project (own network + volumes → its OWN Qdrant collection, Neo4j, Redis)
with ``docker-compose.bench.yml`` layered on top so only the API is
published, on an offset port. NEVER the livestack or the default
collection.

All Docker operations run under the ``/tmp/ns-e2e.lock`` mkdir-lock protocol
shared by the repo's e2e scripts: acquire with ``mkdir`` (atomic), steal if
the holder is older than 45 minutes, release with ``rmdir``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent.parent
OVERRIDE = BENCH_DIR / "docker-compose.bench.yml"
LOCK_DIR = Path("/tmp/ns-e2e.lock")
LOCK_STALE_S = 45 * 60
DEFAULT_PROJECT = "nsbench-accuracy"
DEFAULT_PORT = 8398


def _log(msg: str) -> None:
    print(f"[stack] {msg}", flush=True)


class E2ELock:
    """``/tmp/ns-e2e.lock`` mkdir-lock (steal if >45 min old; rmdir to release)."""

    def __init__(self, path: Path = LOCK_DIR, stale_s: float = LOCK_STALE_S):
        self.path = path
        self.stale_s = stale_s
        self._held = False

    def acquire(self, *, timeout_s: float = 900.0, poll_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self.path.mkdir()
                self._held = True
                return
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue  # released between mkdir and stat — retry now
                if age > self.stale_s:
                    _log(f"stealing stale lock (age {age / 60:.0f} min)")
                    try:
                        self.path.rmdir()
                    except OSError:
                        pass
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"{self.path} held for {age / 60:.0f} min; timed out waiting"
                    ) from None
                time.sleep(poll_s)

    def release(self) -> None:
        if self._held:
            try:
                self.path.rmdir()
            except OSError:
                pass
            self._held = False

    def __enter__(self) -> "E2ELock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _repo_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def _compose(project: str, repo_root: Path) -> list[str]:
    return ["docker", "compose", "-p", project,
            "-f", str(repo_root / "docker-compose.yml"), "-f", str(OVERRIDE)]


def read_env_token(env_file: Path) -> str | None:
    """The stack's ``NEURALSCAPE_API_KEY`` (legacy shared bearer key), if set.

    With the shared key the body ``user_id`` stays authoritative — exactly
    what the per-conversation bench users need. Falls back to None (auth
    disabled) when the key is absent.
    """
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("NEURALSCAPE_API_KEY="):
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            return val or None
    return None


def stack_up(*, project: str = DEFAULT_PROJECT, api_port: int = DEFAULT_PORT,
             env_file: Path | None = None, build: bool = True) -> None:
    repo_root = _repo_root()
    target_env = repo_root / ".env"
    if env_file and env_file != target_env and not target_env.exists():
        shutil.copyfile(env_file, target_env)
    if not target_env.exists():
        raise FileNotFoundError(
            f"No .env at {target_env}; the stack needs GOOGLE_API_KEY / NEO4J_PASSWORD."
        )
    env = {**os.environ, "BENCH_API_PORT": str(api_port)}
    cmd = _compose(project, repo_root) + ["up", "-d"] + (["--build"] if build else [])
    with E2ELock():
        _log("$ " + " ".join(cmd))
        subprocess.run(cmd, env=env, check=True, timeout=1800)
    wait_health(api_port)


def stack_down(*, project: str = DEFAULT_PROJECT, volumes: bool = True) -> None:
    repo_root = _repo_root()
    cmd = _compose(project, repo_root) + ["down"] + (["-v"] if volumes else [])
    with E2ELock():
        _log("$ " + " ".join(cmd))
        subprocess.run(cmd, check=True, timeout=600)


def wait_health(api_port: int, timeout_s: float = 300.0) -> None:
    url = f"http://localhost:{api_port}/health"
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 — localhost
                if r.status == 200:
                    _log(f"healthy: {url}")
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    raise TimeoutError(f"{url} not healthy within {timeout_s}s")
