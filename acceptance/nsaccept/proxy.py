"""A recording reverse proxy between a plugin hook and the stack.

The hooks are black boxes that talk HTTP; most of what they promise is only
visible on that wire (what was sent, how often, with which identity, and
what was NOT sent). Point a hook's URL option at this proxy and every request
is recorded, then either forwarded to the real stack or answered with a canned
reply:

- ``upstream=<stack url>``: unmatched requests are forwarded and the stack's
  answer is returned unchanged (task ids, status codes and all).
- ``stub``: request paths answered locally without touching the stack. The
  default stubs the two LLM-backed conversation-compiler routes (flush and
  compile), so a scenario that only needs to see what the hook SENDS costs no
  extraction call. ``forward_llm=True`` lets them through.
- ``fail``: ``{path: status}`` answers that path with that HTTP status (fault
  injection); ``delay``: ``{path: seconds}`` slows an answer so that two hooks
  really overlap.

Loopback only. Request bodies are kept (they hold the user's own synthetic
test text); the Authorization header is kept for identity checks but is never
rendered into a message: use ``Seen.bearer`` to compare it.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import httpx

FLUSH = "/v1/extensions/conversation-compiler/flush"
COMPILE = "/v1/extensions/conversation-compiler/compile"
LLM_PATHS = (FLUSH, COMPILE)


@dataclass
class Seen:
    method: str
    path: str  # without the query string
    query: dict[str, list[str]]
    raw_body: str
    authorization: str
    status: int = 0
    response: dict | None = None  # parsed JSON answer, when there was one
    stubbed: bool = False
    at: float = field(default_factory=time.monotonic)

    @property
    def body(self) -> dict:
        try:
            parsed = json.loads(self.raw_body) if self.raw_body else {}
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @property
    def bearer(self) -> str:
        return self.authorization[len("Bearer "):] if self.authorization.startswith("Bearer ") else ""

    @property
    def task_id(self) -> str:
        return (self.response or {}).get("task_id") or ""


class RecordingProxy:
    def __init__(
        self,
        upstream: str | None,
        *,
        stub: tuple[str, ...] = LLM_PATHS,
        forward_llm: bool = False,
    ) -> None:
        self.upstream = upstream.rstrip("/") if upstream else None
        self.stub = {p for p in stub if not (forward_llm and p in LLM_PATHS)}
        self.fail: dict[str, int] = {}
        self.delay: dict[str, float] = {}
        self._lock = threading.Lock()
        self._seen: list[Seen] = []
        self._stub_n = 0
        self._http = httpx.Client(timeout=60.0)
        self._server: ThreadingHTTPServer | None = None
        self.url = ""

    # ── lifecycle ─────────────────────────────────────────────
    def start(self) -> "RecordingProxy":
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _handle(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n).decode("utf-8", "replace") if n else ""
                parts = urlsplit(self.path)
                seen = Seen(
                    method=self.command,
                    path=parts.path,
                    query=parse_qs(parts.query),
                    raw_body=raw,
                    authorization=self.headers.get("Authorization", ""),
                )
                status, body, ctype = proxy._answer(seen, parts, raw)
                seen.status = status
                try:
                    parsed = json.loads(body) if body else None
                    seen.response = parsed if isinstance(parsed, dict) else None
                except ValueError:
                    seen.response = None
                with proxy._lock:
                    proxy._seen.append(seen)
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._http.close()

    def __enter__(self) -> "RecordingProxy":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ── answering ─────────────────────────────────────────────
    def _answer(self, seen: Seen, parts, raw: str) -> tuple[int, bytes, str]:
        delay = self.delay.get(seen.path, 0.0)
        if delay:
            time.sleep(delay)
        if seen.path in self.fail:
            seen.stubbed = True
            return self.fail[seen.path], b'{"detail":"injected failure"}', "application/json"
        if seen.path in self.stub or self.upstream is None:
            seen.stubbed = True
            with self._lock:
                self._stub_n += 1
                n = self._stub_n
            if seen.path in LLM_PATHS:
                return 202, json.dumps({"status": "accepted", "task_id": f"stub-{n}"}).encode(), "application/json"
            return 200, json.dumps({"status": "ok", "task_id": None}).encode(), "application/json"
        headers = {"Content-Type": "application/json"}
        if seen.authorization:
            headers["Authorization"] = seen.authorization
        try:
            resp = self._http.request(
                seen.method,
                f"{self.upstream}{parts.path}" + (f"?{parts.query}" if parts.query else ""),
                content=raw.encode() if raw else None,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            return 502, json.dumps({"detail": f"proxy could not reach the stack: {type(exc).__name__}"}).encode(), "application/json"
        return resp.status_code, resp.content, resp.headers.get("content-type", "application/json")

    # ── reading what was seen ─────────────────────────────────
    @property
    def requests(self) -> list[Seen]:
        with self._lock:
            return list(self._seen)

    def calls(self, path: str | None = None, method: str | None = None, *, startswith: str | None = None) -> list[Seen]:
        out = []
        for s in self.requests:
            if path is not None and s.path != path:
                continue
            if startswith is not None and not s.path.startswith(startswith):
                continue
            if method is not None and s.method != method:
                continue
            out.append(s)
        return out

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()
