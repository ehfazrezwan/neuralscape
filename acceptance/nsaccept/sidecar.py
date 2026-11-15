"""Throwaway sidecar containers for scenarios that need something to talk TO.

The one sidecar today is the **sink**: a tiny HTTP server on the stack's own
compose network that (a) receives webhooks the stack fires (``queue.empty``)
and (b) serves fixture JSON a REST connector can pull. It logs every request
(method, path, headers of interest, body) as one JSON line on stdout, and the
harness reads those lines back with ``docker logs`` to assert what arrived.

Safety, enforced here and covered by self-tests:

- only names starting ``nsc-acc-`` are ever created, inspected or removed;
- at most ``MAX_SIDECARS`` live at once (a leaked loop cannot flood a host);
- fixed small limits (64 MB, 0.25 CPU, 32 pids, read-only root, no caps);
- refuses the production compose project (and any project without a network);
- only ``docker run|logs|rm|network ls`` are ever issued, and ``logs``/``rm``
  only against a name this module started;
- always removed: ``Sink.stop()`` in the session finalizer, plus ``atexit``.
"""

from __future__ import annotations

import atexit
import json
import re
import subprocess
import time
from typing import Any, Callable

from . import guard

SIDECAR_PREFIX = "nsc-acc-"
NAME_RE = re.compile(r"^nsc-acc-[a-z0-9][a-z0-9-]{0,40}$")
MAX_SIDECARS = 2
IMAGE = "python:3.12-alpine"
PORT = 8080
SINK_NAME = "nsc-acc-sink"
CONNECTOR_ITEMS_PATH = "/items"
QUEUE_EMPTY_PATH = "/queue-empty"

#: Names this process started (and so may log/remove).
_LIVE: set[str] = set()

Runner = Callable[..., subprocess.CompletedProcess]


class SidecarError(RuntimeError):
    pass


def check_name(name: str) -> str:
    if not NAME_RE.match(name or ""):
        raise SidecarError(f"refusing sidecar name {name!r}: must match {NAME_RE.pattern}")
    return name


# The inline server. Reads fixtures from NSC_FIXTURES (JSON: {path: body}).
# GET <fixture path> -> 200 with the fixture; any other GET -> 404; every other
# method -> 204. Each request is one JSON line on stdout. Port from NSC_PORT.
SERVER_SCRIPT = r"""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
FIX = json.loads(os.environ.get("NSC_FIXTURES") or "{}")
class H(BaseHTTPRequestHandler):
    def _log(self, body):
        rec = {"ts": time.time(), "method": self.command, "path": self.path.split("?", 1)[0],
               "query": self.path.split("?", 1)[1] if "?" in self.path else "",
               "authorization": self.headers.get("Authorization", ""), "body": body}
        sys.stdout.write(json.dumps(rec) + "\n"); sys.stdout.flush()
    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n).decode("utf-8", "replace") if n else ""
    def do_GET(self):
        self._log("")
        data = FIX.get(self.path.split("?", 1)[0])
        if data is None:
            self.send_response(404); self.end_headers(); return
        raw = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers(); self.wfile.write(raw)
    def _other(self):
        self._log(self._body())
        self.send_response(204); self.end_headers()
    do_POST = do_PUT = do_DELETE = do_PATCH = _other
    def log_message(self, *a): pass
port = int(os.environ.get("NSC_PORT") or 8080)
sys.stdout.write(json.dumps({"ready": True, "port": port}) + "\n"); sys.stdout.flush()
ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
"""


def connector_marker(run_id: str, n: int) -> str:
    return f"zq-{run_id}-conn{n}"


def sink_fixtures(run_id: str) -> dict[str, Any]:
    """The JSON the sink serves for this run: two items a REST connector can import."""
    items = [
        {
            "id": f"item-{n}",
            "title": f"Harbour note {n}",
            "rev": "1",
            "body": (
                f"# Harbour note {n}\n\n{connector_marker(run_id, n)} The tide table for the "
                f"{'north' if n == 1 else 'south'} quay is posted every morning at the pilot hut."
            ),
        }
        for n in (1, 2)
    ]
    return {CONNECTOR_ITEMS_PATH: {"items": items}}


def _is_prod(project: str) -> bool:
    return project.strip().lower() == guard.PROD_COMPOSE_PROJECT


class Sink:
    """One ``nsc-acc-*`` sink container on a compose project's network."""

    def __init__(
        self,
        project: str,
        fixtures: dict[str, Any] | None = None,
        name: str = SINK_NAME,
        run: Runner = subprocess.run,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if _is_prod(project):
            raise guard.GuardError(f"refusing compose project '{guard.PROD_COMPOSE_PROJECT}' (production)")
        if not project.strip():
            raise SidecarError("a sink needs the compose project it joins")
        self.project = project.strip()
        self.name = check_name(name)
        self.fixtures = fixtures or {}
        self._run, self._clock, self._sleep = run, clock, sleep
        self.started = False

    # -- docker, narrowly ------------------------------------------------
    def _docker(self, *args: str, check: bool = True, timeout: float = 60.0) -> subprocess.CompletedProcess:
        verb = args[0] if args else ""
        if verb == "network":
            ok = args[:2] == ("network", "ls")
        elif verb in ("logs", "rm"):
            ok = args[-1] == self.name  # the name was validated by check_name() at construction
        else:
            ok = verb == "run"
        if not ok:
            raise SidecarError(f"docker {' '.join(args[:2])} is not something a sidecar may run")
        proc = self._run(["docker", *args], capture_output=True, text=True, timeout=timeout)
        if check and proc.returncode != 0:
            raise SidecarError(f"docker {verb} failed ({proc.returncode}): {(proc.stderr or '').strip()[:300]}")
        return proc

    def network(self) -> str:
        """The stack's compose network, discovered (never assumed)."""
        proc = self._docker(
            "network", "ls", "--filter", f"label=com.docker.compose.project={self.project}",
            "--format", "{{.Name}}",
        )
        names = [n.strip() for n in proc.stdout.splitlines() if n.strip()]
        if f"{self.project}_default" in names:
            return f"{self.project}_default"
        if len(names) == 1:
            return names[0]
        raise SidecarError(f"cannot pick the network of compose project {self.project!r} from {names}")

    # -- lifecycle --------------------------------------------------------
    def start(self, ready_timeout: float = 60.0) -> "Sink":
        if self.name in _LIVE:
            raise SidecarError(f"{self.name} is already running")
        if len(_LIVE) >= MAX_SIDECARS:
            raise SidecarError(f"refusing a third sidecar (cap {MAX_SIDECARS}): {sorted(_LIVE)}")
        net = self.network()
        # A leftover from a crashed earlier run would hold the name.
        self._docker("rm", "-f", self.name, check=False)
        self._docker(
            "run", "-d", "--rm", "--name", self.name, "--network", net,
            "--memory", "64m", "--cpus", "0.25", "--pids-limit", "32",
            "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "-e", f"NSC_FIXTURES={json.dumps(self.fixtures)}", "-e", f"NSC_PORT={PORT}",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            IMAGE, "python", "-c", SERVER_SCRIPT,
            timeout=300.0,  # first use may pull the image
        )
        _LIVE.add(self.name)
        self.started = True
        atexit.register(self.stop)
        deadline = self._clock() + ready_timeout
        while self._clock() < deadline:
            if any(r.get("ready") for r in self.log_records()):
                return self
            self._sleep(1.0)
        self.stop()
        raise SidecarError(f"{self.name} did not report ready within {ready_timeout:.0f}s")

    def stop(self) -> None:
        if self.name in _LIVE or self.started:
            try:
                self._docker("rm", "-f", self.name, check=False)
            finally:
                _LIVE.discard(self.name)
                self.started = False

    # -- what arrived -----------------------------------------------------
    def log_records(self) -> list[dict]:
        proc = self._docker("logs", self.name, check=False)
        out: list[dict] = []
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
        return out

    def requests(self, method: str | None = None, path: str | None = None) -> list[dict]:
        recs = [r for r in self.log_records() if "method" in r]
        if method:
            recs = [r for r in recs if r["method"] == method]
        if path:
            recs = [r for r in recs if r["path"] == path]
        return recs

    def json_bodies(self, path: str) -> list[dict]:
        bodies = []
        for r in self.requests("POST", path):
            try:
                body = json.loads(r.get("body") or "")
            except ValueError:
                continue
            if isinstance(body, dict):
                bodies.append(body)
        return bodies
