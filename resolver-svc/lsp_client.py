"""Minimal LSP JSON-RPC stdio client for pyright-langserver.

Why a hand-rolled client and not multilspy: upstream multilspy only wraps
``jedi-language-server`` for Python — the *same* Jedi fidelity NS already ships
in-process (``adapters/code_graph/code_resolve.py``). The whole point of this
service is **pyright-grade** resolution (a real type checker: cross-file,
inheritance, dynamic dispatch — where Jedi falls short). Upstream multilspy has
no pyright backend, so we drive ``pyright-langserver --stdio`` directly with a
compact LSP client. This mirrors the CBM-bridge pattern: the shim owns the
language-server subprocess lifecycle and exposes a clean HTTP surface.

Scope: exactly the LSP surface ``/resolve_calls`` needs — ``initialize`` /
``initialized``, ``textDocument/didOpen``, and ``textDocument/definition``. One
warm server per workspace root (repo), reused across many resolve calls.

LSP wire facts we rely on:
  - Framing: ``Content-Length: N\\r\\n\\r\\n<json>`` per message.
  - Positions are 0-based (line, character). Character offsets are UTF-16 code
    units per spec; on our ASCII-dominant corpus that equals byte/char offsets
    (same caveat Jedi carries — documented in config.py).
  - ``textDocument/definition`` returns a Location, a Location[], or a
    LocationLink[]; we normalize all three.
"""

from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import pathname2url

logger = logging.getLogger(__name__)

# pyright analyzes the workspace in the background after `initialized`; a
# definition request issued before the first analysis pass can miss cross-file
# targets. We give each freshly-warmed server a short settle window.
_SETTLE_SECONDS = float(os.getenv("PYRIGHT_SETTLE_SECONDS", "2.0"))
_REQUEST_TIMEOUT = float(os.getenv("PYRIGHT_REQUEST_TIMEOUT", "30.0"))


def path_to_uri(path: str) -> str:
    """Absolute filesystem path → ``file://`` URI."""
    return "file://" + pathname2url(str(Path(path).resolve()))


def uri_to_path(uri: str) -> str | None:
    """``file://`` URI → absolute filesystem path (None if not a file URI)."""
    try:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            return None
        return unquote(parsed.path)
    except Exception:
        return None


class LSPError(RuntimeError):
    """Raised on a protocol/transport fault talking to the language server."""


class PyrightServer:
    """One warm ``pyright-langserver --stdio`` process rooted at a repo.

    Thread-safe: a single lock serializes JSON-RPC turns (the resolver shim may
    receive concurrent requests, but one stdio pipe is one conversation).
    """

    def __init__(self, root_path: str, langserver_cmd: list[str] | None = None):
        self.root_path = str(Path(root_path).resolve())
        self.root_uri = path_to_uri(self.root_path)
        self._cmd = langserver_cmd or self._default_cmd()
        self._proc: subprocess.Popen | None = None
        self._next_id = 1
        self._opened: set[str] = set()  # file URIs already didOpen'd
        self._lock = threading.Lock()
        self._started_at = 0.0
        self._stdout_fd: int | None = None  # for select-bounded reads
        self._rbuf = b""  # unparsed bytes carried across _read_message calls

    @staticmethod
    def _default_cmd() -> list[str]:
        # pyright npm package installs the `pyright-langserver` binary. Overridable
        # for tests / alternative installs (e.g. basedpyright-langserver).
        bin_name = os.getenv("PYRIGHT_LANGSERVER", "pyright-langserver")
        return [bin_name, "--stdio"]

    # ── process lifecycle ────────────────────────────────────────────

    def start(self) -> None:
        if self._proc is not None:
            return
        logger.info("starting pyright-langserver for root=%s", self.root_path)
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._stdout_fd = self._proc.stdout.fileno()
        self._initialize()
        self._started_at = time.monotonic()

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ── JSON-RPC framing ─────────────────────────────────────────────

    def _write(self, payload: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise LSPError("language server not started")
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        try:
            self._proc.stdin.write(header + body)
            self._proc.stdin.flush()
        except BrokenPipeError as e:
            raise LSPError(f"language server pipe broken: {e}") from e

    def _recv(self, n: int, deadline: float) -> bytes:
        """Read up to ``n`` bytes from the server's stdout, bounded by ``deadline``.

        Uses ``select`` on the raw fd + ``os.read`` so a stalled language server
        cannot block past the deadline (a plain ``readline``/``read(n)`` on a
        blocking pipe would ignore the deadline entirely — the defect this fixes).
        """
        if self._stdout_fd is None:
            raise LSPError("language server not started")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LSPError("timed out waiting for LSP output")
        r, _, _ = select.select([self._stdout_fd], [], [], remaining)
        if not r:
            raise LSPError("timed out waiting for LSP output")
        chunk = os.read(self._stdout_fd, n)
        if chunk == b"":
            raise LSPError("language server closed stdout (EOF)")
        return chunk

    def _read_message(self, deadline: float) -> dict:
        """Read one framed LSP message from stdout, bounded by ``deadline``.

        Buffers across calls (``self._rbuf``): a single ``os.read`` may span
        message boundaries, and a message may arrive in several reads.
        """
        # Accumulate until the header terminator is present.
        while b"\r\n\r\n" not in self._rbuf:
            self._rbuf += self._recv(4096, deadline)

        header_blob, _, rest = self._rbuf.partition(b"\r\n\r\n")
        length = 0
        for line in header_blob.split(b"\r\n"):
            if b":" in line:
                k, _, v = line.partition(b":")
                if k.strip().lower() == b"content-length":
                    try:
                        length = int(v.strip())
                    except ValueError:
                        raise LSPError(f"bad Content-Length: {v!r}")
        self._rbuf = rest

        while len(self._rbuf) < length:
            self._rbuf += self._recv(4096, deadline)
        body = self._rbuf[:length]
        self._rbuf = self._rbuf[length:]
        return json.loads(body.decode("utf-8"))

    def _request(self, method: str, params: dict, timeout: float = _REQUEST_TIMEOUT) -> dict:
        """Send a request and pump messages until its matching response arrives.

        Server-initiated requests (e.g. ``client/registerCapability``,
        ``window/workDoneProgress/create``) are answered with a null result so
        pyright doesn't stall; notifications are ignored.
        """
        req_id = self._next_id
        self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            msg = self._read_message(deadline)
            if msg.get("id") == req_id and ("result" in msg or "error" in msg):
                if "error" in msg:
                    raise LSPError(f"{method} failed: {msg['error']}")
                return msg.get("result")
            # Server → client request: acknowledge so it can proceed.
            if "method" in msg and "id" in msg:
                self._write({"jsonrpc": "2.0", "id": msg["id"], "result": None})
            # else: a notification (logMessage, publishDiagnostics, ...) — ignore.

    def _notify(self, method: str, params: dict) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    # ── LSP handshake + document ops ─────────────────────────────────

    def _initialize(self) -> None:
        self._request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": self.root_uri,
                "rootPath": self.root_path,
                "workspaceFolders": [{"uri": self.root_uri, "name": "repo"}],
                "capabilities": {
                    "textDocument": {
                        "definition": {"linkSupport": True},
                        "synchronization": {"didSave": False},
                    }
                },
            },
        )
        self._notify("initialized", {})

    def _ensure_open(self, file_path: str) -> None:
        uri = path_to_uri(file_path)
        if uri in self._opened:
            return
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            raise LSPError(f"cannot read {file_path}: {e}") from e
        self._notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "python",
                    "version": 1,
                    "text": text,
                }
            },
        )
        self._opened.add(uri)

    def refresh_document(self, file_path: str) -> None:
        """Re-open ``file_path`` from disk so a WARM server sees on-disk edits.

        MF-2: an opened document's sent text shadows the file on disk for the life
        of the server, and the pool keeps one warm server per repo indefinitely. On
        an incremental reindex after edits, resolving against reindex #1's text
        yields shifted line numbers → wrong FQN mapping. Sending didClose + a fresh
        didOpen makes pyright re-read current contents. Called once per file per
        resolve request (before its sites), so per-site request_definition then
        finds it already open."""
        with self._lock:
            if not self.is_alive():
                raise LSPError("language server not alive")
            uri = path_to_uri(file_path)
            if uri in self._opened:
                self._notify("textDocument/didClose", {"textDocument": {"uri": uri}})
                self._opened.discard(uri)
            self._ensure_open(file_path)

    def request_definition(
        self, file_path: str, line0: int, char0: int, timeout: float = _REQUEST_TIMEOUT
    ) -> list[tuple[str, int]]:
        """Resolve the definition at 0-based ``(line0, char0)`` in ``file_path``.

        Returns a list of ``(abs_def_path, def_line_1based)``. Empty when pyright
        resolves nothing in-repo (stdlib/external/dynamic → dropped by the caller).
        ``timeout`` bounds the round-trip so a stalled server can't hang the request.
        """
        with self._lock:
            if not self.is_alive():
                raise LSPError("language server not alive")
            self._ensure_open(file_path)
            result = self._request(
                "textDocument/definition",
                {
                    "textDocument": {"uri": path_to_uri(file_path)},
                    "position": {"line": line0, "character": char0},
                },
                timeout=timeout,
            )
        return _normalize_definition(result)

    def settle(self) -> None:
        """Block briefly so pyright's first background analysis pass completes
        before the first definition request (cross-file targets otherwise miss)."""
        elapsed = time.monotonic() - self._started_at
        remaining = _SETTLE_SECONDS - elapsed
        if remaining > 0:
            time.sleep(remaining)


def _normalize_definition(result) -> list[tuple[str, int]]:
    """Normalize a Location | Location[] | LocationLink[] into (path, line1based)."""
    if not result:
        return []
    if isinstance(result, dict):
        result = [result]
    out: list[tuple[str, int]] = []
    for loc in result:
        if not isinstance(loc, dict):
            continue
        # LocationLink uses targetUri/targetRange; Location uses uri/range.
        uri = loc.get("uri") or loc.get("targetUri")
        rng = loc.get("range") or loc.get("targetSelectionRange") or loc.get("targetRange")
        if not uri or not rng:
            continue
        path = uri_to_path(uri)
        if path is None:
            continue
        try:
            line0 = int(rng["start"]["line"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append((path, line0 + 1))  # → 1-based to match Jedi / stored spans
    return out
