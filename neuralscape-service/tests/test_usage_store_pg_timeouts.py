"""The PostgreSQL usage store bounds every statement and lock wait (no live server needed)."""

import sys
import types

import usage_store


class _Conn:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        raise _Stop


class _Stop(Exception):
    pass


def test_session_timeouts_are_set_before_anything_else_runs(monkeypatch):
    conn = _Conn()
    fake = types.SimpleNamespace(connect=lambda dsn, **kw: conn)
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    try:
        usage_store.PostgresUsageStore("dbname=x")
    except _Stop:
        pass
    assert conn.calls and conn.calls[0][1][0] in {"statement_timeout", "lock_timeout"}


def test_both_timeouts_are_applied(monkeypatch):
    seen = []

    class Conn(_Conn):
        def execute(self, sql, params=None):
            seen.append(params)
            if len(seen) == 2:
                raise _Stop
            return types.SimpleNamespace(fetchone=lambda: ("on",))

    fake = types.SimpleNamespace(connect=lambda dsn, **kw: Conn())
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    try:
        usage_store.PostgresUsageStore("dbname=x", statement_timeout_ms=5000, lock_timeout_ms=2000)
    except _Stop:
        pass
    assert seen == [("statement_timeout", "5000ms"), ("lock_timeout", "2000ms")]
