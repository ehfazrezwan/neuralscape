"""A tiny local HTTP stub standing in for the stack in self-tests."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class State:
    def __init__(self) -> None:
        self.mcp_mode = "json"  # json | sse | session
        self.require_token = "good-token"
        self.status_sequence = ["queued", "processing", "completed"]
        self.status_calls = 0
        self.seen_headers: list[dict] = []


def make_server(state: State) -> tuple[ThreadingHTTPServer, str]:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _send(self, code, body=b"", ctype="application/json", headers=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _authed(self) -> bool:
            return self.headers.get("Authorization") == f"Bearer {state.require_token}"

        def do_GET(self):
            if self.path == "/health/live":
                return self._send(200, b'{"status":"alive"}')
            if not self._authed():
                return self._send(401, b'{"detail":"nope"}')
            if self.path.startswith("/v1/memories/status/"):
                i = min(state.status_calls, len(state.status_sequence) - 1)
                state.status_calls += 1
                return self._send(200, json.dumps({"task_id": "t1", "status": state.status_sequence[i]}).encode())
            self._send(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            if self.path.startswith("/mcp"):
                if not self._authed():
                    return self._send(401, b"{}")
                state.seen_headers.append({k.lower(): v for k, v in self.headers.items()})
                msg = json.loads(raw)
                if "id" not in msg:
                    return self._send(202)
                if msg["method"] == "initialize":
                    result = {"serverInfo": {"name": "stub"}, "protocolVersion": "2025-03-26"}
                elif msg["method"] == "tools/list":
                    result = {"tools": [{"name": "remember", "inputSchema": {"type": "object"}}]}
                else:
                    result = {"content": [{"type": "text", "text": "[]"}], "isError": False}
                body = {"jsonrpc": "2.0", "id": msg["id"], "result": result}
                extra = {"mcp-session-id": "sess-1"} if state.mcp_mode == "session" else {}
                if state.mcp_mode == "sse":
                    payload = (
                        "event: message\n"
                        + 'data: {"jsonrpc":"2.0","method":"notifications/message","params":{}}\n\n'
                        + "event: message\ndata: " + json.dumps(body) + "\n\n"
                    ).encode()
                    return self._send(200, payload, "text/event-stream")
                return self._send(200, json.dumps(body).encode(), "application/json", extra)
            if not self._authed():
                return self._send(401, b"{}")
            if self.path == "/v1/memories/raw":
                return self._send(202, b'{"status":"accepted","task_id":"t1","poll_url":"/x"}')
            if self.path == "/v1/search":
                return self._send(200, json.dumps({"results": [{"id": "1", "memory": "hello zq-x-marker"}]}).encode())
            self._send(404)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
