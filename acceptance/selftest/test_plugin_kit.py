"""Self-tests for the plugin-scenario kit: the recording proxy, the transcript writer, and the
Session driver against the REAL built hook scripts (no stack: the proxy answers everything).

The node-backed tests are what keep the scenarios honest: the hook behaviours the scenarios assert
(buffer rows, delivered turns, cursor, note, snapshot, compile prompt, outage notice) are checked
here against the actual bundles with the same helpers, so a scenario that could never pass fails
here first.
"""

import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenarios"))

from nsaccept.proxy import COMPILE, FLUSH, RecordingProxy  # noqa: E402
from nsaccept.runctx import Ctx  # noqa: E402
from nsaccept.transcript import Transcript  # noqa: E402
from plugin_kit import (  # noqa: E402
    Session,
    check_sent_exactly_once,
    context_of,
    everything_sent,
    flushes_with,
    option,
)

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


@pytest.fixture
def upstream():
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _do(self):
            n = int(self.headers.get("Content-Length") or 0)
            seen.append((self.command, self.path, self.rfile.read(n).decode() if n else ""))
            body = json.dumps({"from": "upstream", "path": self.path}).encode()
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _do

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    srv.shutdown()
    srv.server_close()


def ctx_for(url: str = "http://127.0.0.1:1") -> Ctx:
    return Ctx(
        base_url=url, profile="default", run_id="selftest", user_a="acc-selftest-a", user_b="acc-selftest-b",
        token_a="ta.sig", token_b="tb.sig", bad_token="bad.sig", project_id="acc-selftest-proj", token_source="test",
    )


# ── the proxy ─────────────────────────────────────────────────────


def test_proxy_forwards_records_and_keeps_the_bearer_out_of_band(upstream):
    url, seen = upstream
    with RecordingProxy(url) as proxy:
        r = httpx.get(f"{proxy.url}/v1/context/p?user_id=u&limit=5", headers={"Authorization": "Bearer tok"})
        assert r.status_code == 201 and r.json()["from"] == "upstream"
    (call,) = proxy.requests
    assert (call.method, call.path, call.query) == ("GET", "/v1/context/p", {"user_id": ["u"], "limit": ["5"]})
    assert call.bearer == "tok" and call.status == 201 and not call.stubbed
    assert seen == [("GET", "/v1/context/p?user_id=u&limit=5", "")]


def test_proxy_answers_the_llm_routes_itself_unless_told_to_forward(upstream):
    url, seen = upstream
    with RecordingProxy(url) as proxy:
        r = httpx.post(f"{proxy.url}{FLUSH}", json={"user_message": "a"})
        assert r.status_code == 202 and r.json()["task_id"].startswith("stub-")
        assert httpx.post(f"{proxy.url}{COMPILE}", json={}).status_code == 202
    assert seen == [] and all(c.stubbed for c in proxy.requests)
    with RecordingProxy(url, forward_llm=True) as proxy:
        assert httpx.post(f"{proxy.url}{FLUSH}", json={"user_message": "a"}).status_code == 201
    assert len(seen) == 1 and proxy.requests[0].task_id == ""


def test_proxy_injects_failures_and_delays_and_works_without_an_upstream():
    with RecordingProxy(None) as proxy:
        proxy.fail[FLUSH] = 500
        assert httpx.post(f"{proxy.url}{FLUSH}", json={}).status_code == 500
        del proxy.fail[FLUSH]
        assert httpx.post(f"{proxy.url}{FLUSH}", json={}).status_code == 202
        assert httpx.get(f"{proxy.url}/anything").json() == {"status": "ok", "task_id": None}
        assert [c.status for c in proxy.calls(FLUSH)] == [500, 202]


def test_proxy_reports_an_unreachable_upstream_as_502():
    with RecordingProxy("http://127.0.0.1:1") as proxy:
        assert httpx.get(f"{proxy.url}/x").status_code == 502


# ── the transcript ────────────────────────────────────────────────


def test_transcript_entries_have_the_real_shape(tmp_path):
    t = Transcript(tmp_path / "t.jsonl", "sess-1", "/work")
    t.turn("a prompt", "an answer", tool=("Bash", {"command": "make"}, "out"))
    t.user("meta", isMeta=True)
    entries = t.lines()
    assert [e["type"] for e in entries] == ["user", "assistant", "user", "assistant", "user"]
    assert entries[0]["message"] == {"role": "user", "content": "a prompt"}
    kinds = [b["type"] for b in entries[1]["message"]["content"]]
    assert kinds == ["thinking", "tool_use"]
    assert entries[2]["message"]["content"][0]["type"] == "tool_result"
    assert entries[3]["message"]["content"] == [{"type": "text", "text": "an answer"}]
    assert entries[4]["isMeta"] is True
    assert all(e["sessionId"] == "sess-1" and e["cwd"] == "/work" and e["isSidechain"] is False for e in entries[:4])
    assert entries[1]["parentUuid"] == entries[0]["uuid"]


def test_a_second_transcript_over_the_same_file_does_not_wipe_it(tmp_path):
    Transcript(tmp_path / "t.jsonl", "s", "/w").user("keep me")
    Transcript(tmp_path / "t.jsonl", "s", "/w")
    assert len(Transcript(tmp_path / "t.jsonl", "s", "/w").lines()) == 1


# ── Session against the real built hooks ──────────────────────────


@pytest.fixture
def lane():
    proxy = RecordingProxy(None).start()
    s = Session(ctx_for(), project_id="acc-selftest-lane", proxy=proxy)
    yield s, proxy
    proxy.stop()
    s.close()


@needs_node
def test_post_tool_use_buffers_and_the_compile_prompt_fires_once(lane):
    s, _ = lane
    s.extra_env.update(option("COMPILE_THRESHOLD", "2"))
    assert context_of(s.user_prompt("hello")) == ""
    s.bash("make build", "built zq-1")
    s.post_tool("Read", {"file_path": "/x"}, "zq-skip")
    s.bash("ls -la", "zq-skip")
    s.bash("make test", "tested")
    assert [r["tool"] for r in s.buffer_rows()] == ["Bash", "Bash"]
    assert "zq-skip" not in s.buffer_path.read_text()
    text = context_of(s.user_prompt("go on"))
    assert str(s.buffer_path) in text and "compile-observations" in text and "2 tool observations" in text
    assert context_of(s.user_prompt("again")) == ""


@needs_node
def test_stop_delivers_the_real_turn_once_and_only_the_turn(lane):
    s, proxy = lane
    t = s.transcript
    t.user("For the record: the codename is zq-code. <private>zq-secret</private> Remember.")
    t.assistant({"type": "thinking", "thinking": "x"}, {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "make"}})
    t.tool_result("toolu_1", "zq-toolres output")
    t.assistant("zq-side a sub-agent", isSidechain=True)
    t.user("zq-meta injected", isMeta=True)
    final = "Noted, the queue is zq-queue, with enough words to be a real answer."
    t.assistant(final)
    s.stop(final)
    check_sent_exactly_once(proxy, "zq-code")
    (body,) = flushes_with(proxy, "zq-code")
    assert "zq-queue" in body["assistant_response"] and body["channel"] == "claude-code"
    assert body["project_id"] == "acc-selftest-lane" and body["user_id"] == "acc-selftest-a"
    sent = proxy.calls(FLUSH)[0].raw_body
    assert all(x not in sent for x in ("zq-toolres", "zq-side", "zq-meta", "zq-secret")) and "[redacted]" in sent
    assert proxy.calls(FLUSH)[0].bearer == "ta.sig"
    assert s.offset() == 6
    s.stop(final)
    check_sent_exactly_once(proxy, "zq-code")  # nothing new: nothing sent
    # Compile is windowed per session (plugin 2.9.8, #514): the first Stop delivered turns and is due, so it
    # compiles once; the second Stop delivers nothing, so it sends no compile. Exactly one, deterministically.
    assert len(proxy.calls(COMPILE)) == 1


@needs_node
def test_session_end_sends_the_note_and_the_ended_marker(lane):
    s, proxy = lane
    s.bash("make build", "ok")
    s.transcript.turn(
        "Please fix zq-req.",
        "All done. Turns out the bug was in the cache layer.\n\nNext steps:\n- zq-next verify",
    )
    s.session_end()
    (post,) = proxy.calls("/v1/checkpoint", "POST")
    note = post.body["session_note"]
    assert post.body["memories"] == [] and "zq-req" in note["request"] and "zq-next" in note["next_steps"]
    assert "cache layer" in note["learned"] and "1 command(s)" in note["investigated"]
    assert (s.sandbox.observations / f"{s.session_id}.jsonl.ended").is_file()


@needs_node
def test_pre_compact_posts_the_snapshot_and_keeps_a_local_copy_that_resumes(lane):
    s, proxy = lane
    for i in range(4):
        s.transcript.turn(f"prompt zq-p{i}", "answered, with enough words to count as one")
    s.pre_compact("manual")
    (post,) = proxy.calls("/v1/memories/raw", "POST")
    assert post.body["category"] == "task_context" and f"session:{s.session_id}" in post.body["tags"]
    content = post.body["content"]
    assert content.startswith("Compact snapshot:") and "zq-p3" in content and "zq-p0" not in content
    assert proxy.calls(FLUSH) == []  # PreCompact does not deliver turns for Claude Code
    down = Session(ctx_for(), project_id="acc-selftest-lane", url="http://127.0.0.1:1", session_id=s.session_id, sandbox=s.sandbox)
    text = context_of(down.session_start("compact"))
    assert "## Resuming after compaction" in text and "zq-p3" in text and "unreachable" in text


@needs_node
def test_outage_counter_goes_loud_and_the_session_never_blocks():
    s = Session(ctx_for(), project_id="acc-selftest-down", proxy=None, url="http://127.0.0.1:1", extra_env=option("FAIL_LOUD_THRESHOLD", "2"))
    try:
        first = s.session_start()
        assert first.returncode == 0 and "consecutive" not in context_of(first)
        assert "unreachable for 2 consecutive events" in context_of(s.session_start())
        assert s.hook("session-start", {}, raw_stdin="{nope").returncode == 2
        assert s.hook("pre-tool-use", {}, raw_stdin="{nope").returncode == 0
    finally:
        s.close()


@needs_node
def test_excluded_project_sends_nothing():
    proxy = RecordingProxy(None).start()
    s = Session(ctx_for(), project_id="acc-selftest-secret", proxy=proxy, extra_env=option("EXCLUDED_PROJECTS", "ACC-*-SECRET"))
    try:
        s.session_start()
        s.bash("make build", "ok")
        s.transcript.turn("hello there", "an answer with enough words to be delivered")
        s.stop("an answer with enough words to be delivered")
        s.pre_compact()
        s.session_end()
        assert proxy.requests == [] and not s.buffer_path.exists() and s.offset() is None
        assert everything_sent(proxy) == ""
    finally:
        proxy.stop()
        s.close()
