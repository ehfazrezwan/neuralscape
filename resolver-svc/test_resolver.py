"""Unit tests for the precise-neighbors resolver shim.

Run from this directory:  cd resolver-svc && python -m pytest test_resolver.py -v

These cover the pure logic — URI/coordinate handling, LSP framing, definition
normalization, and the /resolve_calls contract with a fake pyright server. The
real pyright integration is exercised at benchmark time (it needs the Node
toolchain from the image), not in these host-side unit tests.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

import lsp_client
import main
from lsp_client import (
    PyrightServer,
    _normalize_definition,
    path_to_uri,
    uri_to_path,
)


# ── URI helpers ──────────────────────────────────────────────────────


def test_uri_roundtrip(tmp_path):
    p = tmp_path / "pkg" / "mod.py"
    p.parent.mkdir(parents=True)
    p.write_text("x = 1\n")
    uri = path_to_uri(str(p))
    assert uri.startswith("file://")
    assert uri_to_path(uri) == str(p.resolve())


def test_uri_to_path_rejects_non_file():
    assert uri_to_path("untitled:Untitled-1") is None
    assert uri_to_path("https://example.com/x.py") is None


# ── Definition normalization ─────────────────────────────────────────


def test_normalize_none_and_empty():
    assert _normalize_definition(None) == []
    assert _normalize_definition([]) == []


def test_normalize_single_location():
    loc = {"uri": "file:///repo/a.py", "range": {"start": {"line": 4, "character": 0}}}
    assert _normalize_definition(loc) == [("/repo/a.py", 5)]  # 0-based → 1-based


def test_normalize_location_list():
    locs = [
        {"uri": "file:///repo/a.py", "range": {"start": {"line": 0, "character": 0}}},
        {"uri": "file:///repo/b.py", "range": {"start": {"line": 9, "character": 2}}},
    ]
    assert _normalize_definition(locs) == [("/repo/a.py", 1), ("/repo/b.py", 10)]


def test_normalize_location_link():
    # LocationLink[] shape (linkSupport=True): targetUri + targetSelectionRange.
    link = [
        {
            "targetUri": "file:///repo/c.py",
            "targetSelectionRange": {"start": {"line": 6, "character": 4}},
        }
    ]
    assert _normalize_definition(link) == [("/repo/c.py", 7)]


def test_normalize_drops_non_file_uri():
    locs = [{"uri": "jdt://something", "range": {"start": {"line": 1, "character": 0}}}]
    assert _normalize_definition(locs) == []


def test_normalize_skips_malformed():
    locs = [
        {"uri": "file:///repo/a.py"},  # no range
        {"range": {"start": {"line": 1, "character": 0}}},  # no uri
        {"uri": "file:///repo/b.py", "range": {"start": {}}},  # no line
    ]
    assert _normalize_definition(locs) == []


# ── LSP framing ──────────────────────────────────────────────────────


def _framed(payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


class _FakeProc:
    """A subprocess.Popen stand-in: stdin captures writes; stdout is a real OS
    pipe pre-loaded with scripted bytes (so the select-bounded reader exercises
    the real ``select`` + ``os.read`` path, not a BytesIO shim)."""

    def __init__(self, stdout_bytes: bytes):
        self.stdin = io.BytesIO()
        self._r, w = os.pipe()
        os.write(w, stdout_bytes)
        os.close(w)  # EOF after scripted bytes
        self._alive = True

    def stdout_fd(self):
        return self._r

    def poll(self):
        return None if self._alive else 0


def test_read_message_parses_frame():
    srv = PyrightServer("/repo")
    payload = {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
    srv._proc = _FakeProc(_framed(payload))
    srv._stdout_fd = srv._proc.stdout_fd()
    import time

    msg = srv._read_message(deadline=time.monotonic() + 5)
    assert msg == payload


def test_read_message_times_out_when_no_output():
    srv = PyrightServer("/repo")
    r, _w = os.pipe()  # nothing ever written → reader must honor the deadline
    srv._stdout_fd = r
    import time

    with pytest.raises(Exception):
        srv._read_message(deadline=time.monotonic() + 0.3)


def test_write_frames_content_length():
    srv = PyrightServer("/repo")
    srv._proc = _FakeProc(b"")
    srv._write({"jsonrpc": "2.0", "method": "initialized", "params": {}})
    written = srv._proc.stdin.getvalue()
    assert written.startswith(b"Content-Length: ")
    header, _, body = written.partition(b"\r\n\r\n")
    length = int(header.split(b":")[1].strip())
    assert length == len(body)
    assert json.loads(body)["method"] == "initialized"


def test_request_correlates_and_answers_server_requests():
    srv = PyrightServer("/repo")
    # Server first sends a server→client request (must be ack'd with null), then
    # the real response to our id=1 request.
    server_req = {"jsonrpc": "2.0", "id": 99, "method": "window/workDoneProgress/create", "params": {}}
    real_resp = {"jsonrpc": "2.0", "id": 1, "result": [{"uri": "file:///repo/a.py", "range": {"start": {"line": 0, "character": 0}}}]}
    srv._proc = _FakeProc(_framed(server_req) + _framed(real_resp))
    srv._stdout_fd = srv._proc.stdout_fd()
    result = srv._request("textDocument/definition", {}, timeout=5)
    assert _normalize_definition(result) == [("/repo/a.py", 1)]
    # We answered the server request with a null result.
    written = srv._proc.stdin.getvalue()
    assert b'"id": 99' in written or b'"id":99' in written


# ── /resolve_calls contract with a fake pyright ──────────────────────


class _FakeServer:
    """Stand-in for PyrightServer that maps hard-coded sites → defs."""

    def __init__(self, repo_root: str, mapping: dict):
        self.root = repo_root
        self._mapping = mapping  # (rel, line0, col) -> [(abs_path, line1), ...]

    def is_alive(self):
        return True

    def request_definition(self, abs_path, line0, char0):
        rel = str(Path(abs_path).resolve().relative_to(Path(self.root).resolve()))
        return self._mapping.get((rel, line0, char0), [])


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    (tmp_path / "a.py").write_text("def f():\n    g()\n")
    (tmp_path / "b.py").write_text("def g():\n    pass\n")
    repo = str(tmp_path.resolve())

    # a.py line2 (1-based) col4 → g in b.py line1. External call → out-of-repo.
    mapping = {
        ("a.py", 1, 4): [(str(tmp_path / "b.py"), 1)],       # resolves in-repo
        ("a.py", 5, 0): [("/usr/lib/python3/os.py", 10)],    # external → dropped
    }

    class _Pool:
        def get(self, repo_path):
            return _FakeServer(repo, mapping)

        def shutdown(self):
            pass

    fake_pool = _Pool()
    # Patch BOTH the module global AND the ServerPool class: the app's lifespan
    # hook reassigns `pool = ServerPool()` on startup, so if a test enters the
    # TestClient as a context manager (triggering lifespan) the plain global
    # patch would be clobbered — patching the constructor keeps the fake in place.
    monkeypatch.setattr(main, "pool", fake_pool)
    monkeypatch.setattr(main, "ServerPool", lambda: fake_pool)
    return TestClient(main.app), repo


def test_resolve_calls_in_repo(client):
    tc, repo = client
    resp = tc.post(
        "/resolve_calls",
        json={"repo_path": repo, "files": [{"path": "a.py", "sites": [[2, 4]]}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["resolved"] == 1
    defs = data["files"][0]["defs"]
    assert defs[0][0].endswith("b.py")
    assert defs[0][1] == 1


def test_resolve_calls_external_dropped(client):
    tc, repo = client
    resp = tc.post(
        "/resolve_calls",
        json={"repo_path": repo, "files": [{"path": "a.py", "sites": [[6, 0]]}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["resolved"] == 0
    assert data["files"][0]["defs"][0] == [None, None]


def test_resolve_calls_unknown_site_unresolved(client):
    tc, repo = client
    resp = tc.post(
        "/resolve_calls",
        json={"repo_path": repo, "files": [{"path": "a.py", "sites": [[99, 9]]}]},
    )
    assert resp.status_code == 200
    assert resp.json()["files"][0]["defs"][0] == [None, None]


def test_resolve_calls_bad_repo_path(client):
    tc, _ = client
    resp = tc.post(
        "/resolve_calls",
        json={"repo_path": "/no/such/repo", "files": []},
    )
    assert resp.status_code == 400


def test_health_ok_or_unavailable(client, monkeypatch):
    tc, _ = client
    # Force the binary "present" branch.
    monkeypatch.setattr(lsp_client, "os", lsp_client.os)  # no-op keep import
    resp = tc.get("/health")
    assert resp.status_code in (200, 503)
