"""Bearer middleware: no token and wrong-secret tokens are refused, a real one works."""

from __future__ import annotations

from nsaccept import expect_check_fails, negative_control, proves

READ = "/v1/memories?limit=1"


def check_read_ok(client) -> None:
    r = client.get(READ)
    assert r.status_code == 200, f"GET {READ} -> {r.status_code}: {r.text[:160]}"


def check_read_refused(client) -> None:
    r = client.get(READ)
    assert r.status_code == 401, f"GET {READ} -> {r.status_code} (want 401)"


@proves("auth.bearer-middleware")
def test_no_token_is_401(anon):
    check_read_refused(anon)


@proves("auth.bearer-middleware")
def test_wrong_secret_token_is_401(bad):
    check_read_refused(bad)


@proves("auth.bearer-middleware")
def test_valid_token_is_200(a):
    check_read_ok(a)


@proves("auth.bearer-middleware")
@negative_control
def test_valid_token_check_fails_with_wrong_secret_token(bad):
    expect_check_fails(check_read_ok, bad)
