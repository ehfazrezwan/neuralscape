"""Self-tests for the helpers the core scenarios use (no stack needed)."""

import io
import subprocess
import threading
import time
import zipfile

import httpx
import pytest

from nsaccept import caps, guard, probe, runctx
from nsaccept.client import NSClient
from nsaccept.helpers import Written, blob_has, memory_ids_in, rows_with
from nsaccept.pdfgen import make_pdf
from nsaccept.sse import SSEReader, SSEFrame, parse_sse_lines
from nsaccept.upload import build_zip, upload_files


# ── ctx.compose ──────────────────────────────────────────────────


def _ctx(project):
    return runctx.Ctx(
        base_url="http://x", profile="default", run_id="r", user_a="a", user_b="b",
        token_a="ta", token_b="tb", bad_token="bad", project_id="p", token_source="s",
        compose_project=project,
    )


def _fake_run(calls, rc=0):
    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, rc, "out", "boom")
    return run


def test_compose_runs_docker_compose_with_the_project():
    calls = []
    _ctx("nscampaign").compose("pause", "qdrant", run=_fake_run(calls))
    assert calls == [["docker", "compose", "-p", "nscampaign", "pause", "qdrant"]]


def test_compose_needs_a_project():
    with pytest.raises(runctx.ConfigError):
        _ctx(None).compose("ps", run=_fake_run([]))


@pytest.mark.parametrize("project", ["neuralscape", " Neuralscape "])
def test_compose_refuses_the_production_project(project):
    calls = []
    with pytest.raises(guard.GuardError):
        _ctx(project).compose("stop", "qdrant", run=_fake_run(calls))
    assert calls == []


@pytest.mark.parametrize("verb", ["down", "rm", "kill", "up", "build", ""])
def test_compose_only_lets_lifecycle_verbs_through(verb):
    calls = []
    with pytest.raises(runctx.ConfigError):
        _ctx("nscampaign").compose(verb, run=_fake_run(calls))
    assert calls == []


def test_compose_raises_on_failure_unless_unchecked():
    with pytest.raises(RuntimeError, match="exited 1"):
        _ctx("nscampaign").compose("start", "qdrant", run=_fake_run([], rc=1))
    assert _ctx("nscampaign").compose("start", "qdrant", check=False, run=_fake_run([], rc=1)).returncode == 1


def test_compose_capability_follows_the_flag(monkeypatch):
    monkeypatch.delenv("NS_HAVE_COMPOSE", raising=False)
    assert "--compose-project" in caps.missing_reason("compose")
    monkeypatch.setenv("NS_HAVE_COMPOSE", "1")
    monkeypatch.setattr(caps.shutil, "which", lambda name: "/usr/bin/docker")
    assert caps.missing_reason("compose") is None
    monkeypatch.setattr(caps.shutil, "which", lambda name: None)
    assert "docker" in caps.missing_reason("compose")


# ── SSE ──────────────────────────────────────────────────────────


def test_sse_parser_handles_comments_events_and_multiline_data():
    lines = [": connected", "", "event: memory_stored", 'data: {"memory_id":"m1"}', "", ": keep-alive", "",
             "data: a", "data: b", ""]
    frames = list(parse_sse_lines(lines))
    assert [f.comment for f in frames] == ["connected", None, "keep-alive", None]
    assert frames[1].event == "memory_stored" and frames[1].json() == {"memory_id": "m1"}
    assert frames[3].event == "message" and frames[3].data == "a\nb"


def test_sse_frame_json_is_none_for_non_json():
    assert SSEFrame(data="not json").json() is None
    assert SSEFrame(data="[1]").json() is None


def _sse_server(body: bytes, status=200):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer good":
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            time.sleep(0.3)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_sse_reader_collects_frames_from_a_live_stream():
    srv, url = _sse_server(b': connected\n\nevent: memory_stored\ndata: {"memory_id":"m1","content":"zq-x"}\n\n')
    r = SSEReader(url, "good").start()
    try:
        assert r.wait_connected(5)
        frame = r.wait_for(lambda f: "zq-x" in f.data, 5)
        assert frame is not None and frame.event == "memory_stored"
        assert r.events() == [{"memory_id": "m1", "content": "zq-x"}] and r.status_code == 200
    finally:
        r.close()
        srv.shutdown()
        srv.server_close()


def test_sse_reader_reports_a_refused_stream():
    srv, url = _sse_server(b"")
    r = SSEReader(url, "wrong").start()
    try:
        assert r.wait_connected(2) is False
        assert r.status_code == 401 and r.error == "HTTP 401"
    finally:
        r.close()
        srv.shutdown()
        srv.server_close()


def test_sse_wait_for_times_out_with_none():
    srv, url = _sse_server(b": connected\n\n")
    r = SSEReader(url, "good").start()
    try:
        assert r.wait_for(lambda f: "never" in f.data, 0.5) is None
    finally:
        r.close()
        srv.shutdown()
        srv.server_close()


# ── PDF / zip / upload ───────────────────────────────────────────


def test_pdf_builder_makes_a_structurally_valid_pdf():
    pdf = make_pdf(["Hello (world)", "second \\ line"])
    assert pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF")
    assert b"(Hello \\(world\\)) Tj" in pdf and b"T* (second \\\\ line) Tj" in pdf
    # every xref offset points at "<n> 0 obj"
    start = int(pdf.rsplit(b"startxref\n", 1)[1].split(b"\n", 1)[0])
    table = pdf[start:].split(b"\n")
    count = int(table[1].split()[1])
    for n in range(1, count):
        offset = int(table[2 + n].split()[0])
        assert pdf[offset:].startswith(f"{n} 0 obj".encode()), f"xref entry {n} is off"
    stream_len = int(pdf.split(b"/Length ", 1)[1].split(b" ", 1)[0])
    body = pdf.split(b"stream\n", 1)[1].split(b"\nendstream", 1)[0]
    assert len(body) == stream_len


def test_zip_builder_round_trips():
    data = build_zip({"a/b.md": "hello", "c.bin": b"\x00\x01"})
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert sorted(zf.namelist()) == ["a/b.md", "c.bin"] and zf.read("a/b.md") == b"hello"


def test_upload_files_sends_multipart_with_stringified_form_fields():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ctype"] = request.headers["content-type"]
        seen["body"] = request.content
        return httpx.Response(202, json={"files": [], "count": 0})

    c = NSClient("http://x", "t")
    c.http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    r = upload_files(c, {"n.md": b"payload"}, mime="text/markdown", extract_facts=False, project_id=None, scope="global")
    assert r.status_code == 202 and seen["ctype"].startswith("multipart/form-data")
    body = seen["body"]
    assert b'name="files"; filename="n.md"' in body and b"payload" in body
    assert b'name="extract_facts"' in body and b"\r\n\r\nfalse\r\n" in body
    assert b"project_id" not in body  # None fields are dropped


def test_upload_files_asserts_the_expected_status():
    c = NSClient("http://x", "t")
    c.http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(lambda r: httpx.Response(400, text="no")))
    with pytest.raises(AssertionError, match="400"):
        upload_files(c, {"n.md": b"x"})
    assert upload_files(c, {"n.md": b"x"}, expect=None).status_code == 400


# ── route probe ──────────────────────────────────────────────────


def _resp(status, body):
    return httpx.Response(status, json=body)


@pytest.mark.parametrize(
    "status,body,present",
    [
        (200, {}, True),
        (422, {"detail": []}, True),
        (404, {"detail": "Memory x not found"}, True),
        (404, {"detail": "Not Found"}, False),
        (405, {"detail": "Method Not Allowed"}, False),
        (500, {"detail": "boom"}, False),
        (501, {"detail": "extra missing"}, True),
        (503, {"detail": "Connectors are disabled."}, True),
        (503, {"detail": "redis down"}, False),
    ],
)
def test_probe_classification(status, body, present):
    assert probe.classify("GET", "/x", _resp(status, body)).present is present


def test_probe_route_sends_through_the_client():
    c = NSClient("http://x", "t")
    c.http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(lambda r: _resp(404, {"detail": "Not Found"})))
    result = probe.probe_route(c, "GET", "/v1/nope")
    assert result.present is False and result.status == 404 and "route not mounted" in result.line()


# ── Written / helpers ────────────────────────────────────────────


def test_memory_ids_in_reads_the_task_result():
    assert memory_ids_in({"result": {"memories": [{"id": "a"}, {"x": 1}, {"id": "b"}]}}) == ["a", "b"]
    assert memory_ids_in({"result": None}) == [] and memory_ids_in({}) == []


def test_written_require_explains_each_failure():
    with pytest.raises(AssertionError, match="did not start"):
        Written("m", "c", error="boom").require()
    with pytest.raises(AssertionError, match="ended as"):
        Written("m", "c", status={"status": "failed"}).require()
    with pytest.raises(AssertionError, match="without a memory id"):
        Written("m", "c", status={"status": "completed"}).require()
    assert Written("m", "c", status={"status": "completed"}, memory_id="x").require().memory_id == "x"


def test_rows_with_and_blob_has():
    rows = [{"memory": "has zq-1"}, {"memory": "no"}]
    assert rows_with(rows, "zq-1") == [rows[0]]
    assert blob_has({"a": ["Hello ZqAbc"]}, "zqabc") and not blob_has({"a": 1}, "zz")


# ── compose `up`, ps parsing, restart count, in-container token minting ──


def test_compose_up_only_starts_existing_containers():
    calls = []
    ctx = _ctx("nscampaign")
    for bad in (["up", "-d", "svc"], ["up", "svc"], ["up", "-d", "--no-build", "svc"]):
        with pytest.raises(runctx.ConfigError):
            ctx.compose(*bad, run=_fake_run(calls))
    ctx.compose("up", "-d", "--no-build", "--no-recreate", "neuralscape", run=_fake_run(calls))
    assert calls[-1][:4] == ["docker", "compose", "-p", "nscampaign"] and len(calls) == 1


def test_parse_ps_json_accepts_array_and_json_lines():
    assert runctx.parse_ps_json("") == []
    assert runctx.parse_ps_json('[{"Service":"a"},{"Service":"b"}]') == [{"Service": "a"}, {"Service": "b"}]
    assert runctx.parse_ps_json('{"Service":"a"}\n{"Service":"b"}\n') == [{"Service": "a"}, {"Service": "b"}]


def test_restart_count_reads_docker_inspect_and_rejects_odd_ids():
    def run(cmd, **kw):
        assert cmd[:3] == ["docker", "inspect", "--format"]
        return subprocess.CompletedProcess(cmd, 0, "2\n", "")

    assert runctx.container_restart_count("abcdef123456", run=run) == 2
    with pytest.raises(runctx.ConfigError):
        runctx.container_restart_count("x; rm -rf /", run=run)


def test_in_container_minting_never_passes_a_secret_and_refuses_production():
    from nsaccept import tokens

    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "aaaaaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbbbbbb\n", "")

    tok = tokens.mint_via_compose_python("nscampaign", "u1", -60, "api", run=run)
    assert tok == "aaaaaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbbbbbb"
    cmd = seen[0]
    assert cmd[:7] == ["docker", "compose", "-p", "nscampaign", "exec", "-T", "api"] and cmd[-2:] == ["u1", "-60"]
    assert "NEURALSCAPE_USER_TOKEN_SECRET" in cmd[9] and "print(issue_user_token" in cmd[9]
    with pytest.raises(tokens.TokenError):
        tokens.mint_via_compose_python("neuralscape", "u1", -60, run=run)


def test_compose_exec_and_up_are_restricted_to_what_the_scenarios_use():
    ctx, calls = _ctx("nscampaign"), []
    ok = _fake_run(calls)
    ctx.compose("exec", "-T", "qdrant", "sh", "-c", "curl x", run=ok)
    for bad in (("exec", "-T", "neuralscape", "sh", "-c", "x"), ("exec", "-T", "qdrant", "bash"),
                ("exec", "-T", "qdrant", "sh", "-c", "x", "y")):
        with pytest.raises(runctx.ConfigError):
            ctx.compose(*bad, run=ok)
    ctx.compose("up", "-d", "--no-build", "--no-recreate", "--no-deps", "neuralscape", "neuralscape-worker", run=ok)
    for bad in (("up", "-d", "--no-build", "--no-recreate", "qdrant"),
                ("up", "-d", "--no-build", "--no-recreate", "--force-recreate", "neuralscape"),
                ("up", "-d", "--no-build", "--no-recreate")):
        with pytest.raises(runctx.ConfigError):
            ctx.compose(*bad, run=ok)
