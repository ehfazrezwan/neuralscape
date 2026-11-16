"""The queue.empty webhook always refuses link-local and cloud-metadata targets, at send time.

Private and loopback targets stay allowed (local consumers, the acceptance sink). The
connection goes to the address that was validated, so a second DNS answer cannot rebind
the name onto a metadata address. Sinks here are real loopback HTTP servers.
"""

from __future__ import annotations

import http.server
import logging
import threading

import pytest

import webhooks

PAYLOAD = {"event": "queue.empty", "queue": "q"}


class _Sink:
    def __init__(self, status=204, location=None):
        sink = self
        self.requests: list[dict] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                sink.requests.append({"host": self.headers.get("Host"), "body": self.rfile.read(length)})
                self.send_response(status)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def sink():
    s = _Sink()
    yield s
    s.close()


@pytest.fixture()
def no_connection(monkeypatch):
    """Fail the test if a send is attempted at all."""
    calls = []

    def boom(addresses):
        calls.append(addresses)
        raise AssertionError("a request was attempted")

    monkeypatch.setattr(webhooks, "_pinned_transport", boom)
    return calls


NEVER = [
    "169.254.169.254",
    "[fe80::1]",
    "168.63.129.16",
    "[fd00:ec2::254]",
    "[::ffff:169.254.169.254]",
    "[64:ff9b::a9fe:a9fe]",
    "2852039166",  # decimal spelling of 169.254.169.254
]


@pytest.mark.parametrize("host", NEVER)
def test_literal_never_allowed_hosts_are_refused_at_dispatch(host, monkeypatch, caplog, no_connection):
    started = []
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self))
    with caplog.at_level(logging.WARNING, logger="webhooks"):
        assert webhooks.fire_queue_empty(f"http://{host}/latest?token=abc", PAYLOAD) is False
    assert started == []
    assert "token=abc" not in caplog.text
    assert "rejected" in caplog.text


@pytest.mark.parametrize("host", NEVER)
def test_literal_never_allowed_hosts_are_refused_when_sent(host, caplog, no_connection):
    with caplog.at_level(logging.WARNING, logger="webhooks"):
        webhooks._post(f"http://{host}/latest?token=abc", PAYLOAD)
    assert no_connection == []
    assert "refused" in caplog.text
    assert "token=abc" not in caplog.text


@pytest.mark.parametrize(
    "answers",
    [["169.254.169.254"], ["203.0.113.9", "169.254.169.254"], ["fe80::1%eth0"], ["::ffff:169.254.169.254"], ["fd00:ec2::254"]],
)
def test_name_resolving_to_a_never_allowed_address_is_refused(answers, monkeypatch, caplog, no_connection):
    monkeypatch.setattr(webhooks, "_resolve", lambda host, port: answers)
    with caplog.at_level(logging.WARNING, logger="webhooks"):
        webhooks._post("http://hooks.example.test/x?secret=1", PAYLOAD)
    assert no_connection == []
    assert "hooks.example.test" in caplog.text
    assert "secret=1" not in caplog.text


def test_a_name_that_does_not_resolve_sends_nothing(monkeypatch, no_connection):
    def fail(host, port):
        raise OSError("no such host")

    monkeypatch.setattr(webhooks, "_resolve", fail)
    webhooks._post("http://nowhere.test/x", PAYLOAD)
    assert no_connection == []


@pytest.mark.parametrize("host", ["127.0.0.1", "[::1]", "10.1.2.3", "192.168.1.5", "203.0.113.9"])
def test_loopback_private_and_public_literals_are_not_refused(host, monkeypatch):
    sent = []
    monkeypatch.setattr(webhooks, "_pinned_transport", lambda addresses: sent.append(addresses) or (_ for _ in ()).throw(RuntimeError("stop")))
    assert webhooks._validated_addresses(f"http://{host}:8/x") is not None
    webhooks._post(f"http://{host}:8/x", PAYLOAD)  # reaches the send step
    assert len(sent) == 1


def test_loopback_sink_receives_the_event(sink):
    webhooks._post(f"http://127.0.0.1:{sink.port}/hook", PAYLOAD)
    assert len(sink.requests) == 1
    assert b'"queue.empty"' in sink.requests[0]["body"]


def test_name_resolving_to_loopback_is_delivered(sink, monkeypatch):
    monkeypatch.setattr(webhooks, "_resolve", lambda host, port: ["127.0.0.1"])
    webhooks._post(f"http://consumer.test:{sink.port}/hook", PAYLOAD)
    assert len(sink.requests) == 1
    assert sink.requests[0]["host"] == f"consumer.test:{sink.port}"


def test_connection_goes_to_the_validated_address_not_a_second_lookup(sink, monkeypatch):
    lookups = []

    def rebinding(host, port):
        lookups.append(host)
        return ["127.0.0.1"] if len(lookups) == 1 else ["169.254.169.254"]

    monkeypatch.setattr(webhooks, "_resolve", rebinding)
    webhooks._post(f"http://rebind.test:{sink.port}/hook", PAYLOAD)
    assert lookups == ["rebind.test"]  # the connect did not resolve the name again
    assert len(sink.requests) == 1  # and reached the validated address


def test_redirects_are_not_followed(monkeypatch):
    target = _Sink()
    redirecting = _Sink(status=302, location=f"http://127.0.0.1:{target.port}/elsewhere")
    try:
        webhooks._post(f"http://127.0.0.1:{redirecting.port}/hook", PAYLOAD)
        assert len(redirecting.requests) == 1
        assert target.requests == []
    finally:
        redirecting.close()
        target.close()


def test_proxy_environment_is_ignored(sink, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    webhooks._post(f"http://127.0.0.1:{sink.port}/hook", PAYLOAD)
    assert len(sink.requests) == 1


# --- review findings: invalid port, log redaction, one parser, bounded resolver ---------------


@pytest.mark.parametrize("url", ["http://hooks.example.test:abc/x?token=zzz", "http://[::1]:99999999999/x?token=zzz"])
def test_invalid_port_is_refused_loudly_at_dispatch(url, monkeypatch, caplog):
    started = []
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self))
    with caplog.at_level(logging.WARNING, logger="webhooks"):
        assert webhooks.fire_queue_empty(url, PAYLOAD) is False
    assert started == []
    assert "rejected" in caplog.text
    assert "token=zzz" not in caplog.text


@pytest.mark.parametrize("url", ["ftp://hooks.example.test/x?token=zzz", "gopher://u:pw@hooks.example.test/x?token=zzz", "http:///x?token=zzz"])
def test_rejection_log_never_carries_the_query_or_userinfo(url, caplog):
    with caplog.at_level(logging.WARNING, logger="webhooks"):
        assert webhooks.fire_queue_empty(url, PAYLOAD) is False
    assert "rejected" in caplog.text
    assert "token=zzz" not in caplog.text
    assert "pw" not in caplog.text.replace("rejected", "")


@pytest.mark.parametrize(
    "url",
    [
        "http://169\u3002254\u3002169\u3002254/latest",  # ideographic dots: urlparse sees a name, httpx sends to the IP
        "http://169.254.169.254\t/latest",  # urlparse accepts the tab, httpx refuses the URL
    ],
)
def test_parser_divergences_cannot_hide_a_metadata_host(url, monkeypatch):
    started = []
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self))
    assert webhooks.fire_queue_empty(url, PAYLOAD) is False
    assert started == []


@pytest.mark.parametrize(
    "url, host",
    [
        ("http://example.com\\@169.254.169.254/x", "169.254.169.254"),
        ("http://169.254.169.254\\@example.com/x", "example.com"),
        ("http://example.com@169.254.169.254/x", "169.254.169.254"),
    ],
)
def test_host_judged_is_the_host_httpx_sends_to(url, host):
    import httpx

    assert webhooks._host_and_port(url)[0] == host == httpx.URL(url).raw_host.decode()


def test_backslash_userinfo_to_metadata_is_refused(no_connection):
    webhooks._post("http://example.com\\@169.254.169.254/x", PAYLOAD)
    assert no_connection == []


def test_a_lookup_that_never_answers_times_out_and_sends_nothing(monkeypatch, no_connection, caplog):
    release = threading.Event()
    monkeypatch.setattr(webhooks, "WEBHOOK_TIMEOUT_S", 0.2)
    monkeypatch.setattr(webhooks.socket, "getaddrinfo", lambda *a, **k: release.wait(5) or [])
    try:
        with caplog.at_level(logging.WARNING, logger="webhooks"):
            webhooks._post("http://slow.example.test/x", PAYLOAD)
    finally:
        release.set()
    assert no_connection == []
    assert "slow.example.test" in caplog.text and "TimeoutError" in caplog.text


def test_concurrent_lookups_are_bounded_and_the_overflow_is_refused(monkeypatch):
    release = threading.Event()
    calls = []

    def blocked(*args, **kwargs):
        calls.append(args[0])
        release.wait(5)
        return []

    monkeypatch.setattr(webhooks, "WEBHOOK_TIMEOUT_S", 0.05)
    monkeypatch.setattr(webhooks.socket, "getaddrinfo", blocked)
    try:
        for i in range(webhooks.MAX_RESOLVER_THREADS):
            with pytest.raises(TimeoutError):
                webhooks._resolve(f"slow{i}.example.test", 80)
        with pytest.raises(OSError, match="busy"):
            webhooks._resolve("overflow.example.test", 80)
        assert len(calls) == webhooks.MAX_RESOLVER_THREADS  # no extra thread was started
    finally:
        release.set()
    # slots come back once the stuck lookups finish
    deadline = __import__("time").monotonic() + 3
    while __import__("time").monotonic() < deadline:
        monkeypatch.setattr(webhooks.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 80))])
        try:
            assert webhooks._resolve("ok.example.test", 80) == ["127.0.0.1"]
            break
        except OSError:
            __import__("time").sleep(0.05)
    else:
        pytest.fail("resolver slots were never released")
