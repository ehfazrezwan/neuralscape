import pytest

from nsaccept import eventually, expect_check_fails
from nsaccept.client import NSClient


def test_expect_check_fails_passes_when_check_raises_assertion():
    def check(x):
        assert x == 1, "wrong"
    expect_check_fails(check, 2)


def test_expect_check_fails_raises_when_check_passes():
    with pytest.raises(AssertionError, match="cannot fail"):
        expect_check_fails(lambda: None)


def test_expect_check_fails_propagates_other_errors():
    def check():
        raise ConnectionError("down")
    with pytest.raises(ConnectionError):
        expect_check_fails(check)


def test_eventually_retries_then_succeeds():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        assert calls["n"] >= 3, "not yet"
        return "ok"

    assert eventually(fn, timeout=100, interval=0, sleep=lambda s: None) == "ok"
    assert calls["n"] == 3


def test_eventually_times_out_with_last_message():
    t = {"now": 0.0}

    def fn():
        raise AssertionError("still empty")

    with pytest.raises(AssertionError, match="still empty"):
        eventually(
            fn, timeout=5, interval=1, what="thing",
            sleep=lambda s: t.__setitem__("now", t["now"] + s), clock=lambda: t["now"],
        )


def test_eventually_does_not_swallow_other_errors():
    def fn():
        raise ValueError("bug")
    with pytest.raises(ValueError):
        eventually(fn, timeout=5, interval=0, sleep=lambda s: None)


def test_client_store_poll_search_against_stub(stub):
    state, url = stub
    c = NSClient(url, "good-token", "u")
    task = c.store_raw("hello", "preference")
    assert task == "t1"
    final = c.poll_task(task, timeout=10, interval=0, sleep=lambda s: None)
    assert final["status"] == "completed" and state.status_calls == 3
    assert "marker" in c.search("q")[0]["memory"]
    c.close()


def test_poll_task_reports_timeout_instead_of_raising(stub):
    state, url = stub
    state.status_sequence = ["processing"]
    c = NSClient(url, "good-token")
    t = {"now": 0.0}
    final = c.poll_task(
        "t1", timeout=3, interval=1,
        sleep=lambda s: t.__setitem__("now", t["now"] + s), clock=lambda: t["now"],
    )
    assert final["status"] == "timeout" and final["last_status"] == "processing"


def test_client_without_token_gets_401(stub):
    _, url = stub
    c = NSClient(url)
    assert c.get("/v1/memories/status/t1").status_code == 401
    with pytest.raises(AssertionError, match="401"):
        c.store_raw("x", "preference")
