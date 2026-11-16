"""With USAGE_STORE_ENABLED and an unreachable store the API and every worker refuse to start (fail closed).

Capture the operator asked for must not silently turn into "records nothing": the process fails at start-up
with the store's own error, before it builds any provider client.  Documented in ``.env.example``.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

import usage_capture as uc
import usage_store
from config import settings


class StoreDown(RuntimeError):
    pass


@pytest.fixture()
def store_unreachable(monkeypatch):
    monkeypatch.setattr(settings, "usage_store_enabled", True)

    def boom(_settings=None):
        raise StoreDown("usage store unreachable")

    monkeypatch.setattr(usage_store, "install_usage_capture", boom)
    yield
    uc.set_usage_recorder(None)


def test_install_with_an_unusable_store_raises_and_installs_nothing(tmp_path):
    from config import Settings

    broken = Settings(usage_store_enabled=True, usage_store_sqlite_path=str(tmp_path / "missing-dir" / "u.sqlite3"))
    with pytest.raises(Exception):
        usage_store.install_usage_capture(broken)
    assert uc.get_usage_recorder() is None


def test_an_unreachable_postgres_store_raises_at_install(monkeypatch):
    import sys
    import types

    from config import Settings

    def refuse(dsn, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setitem(sys.modules, "psycopg", types.SimpleNamespace(connect=refuse))
    broken = Settings(
        usage_store_enabled=True, usage_store_backend="postgres", usage_store_postgres_dsn="host=127.0.0.1 port=1",
    )
    with pytest.raises(OSError):
        usage_store.install_usage_capture(broken)
    assert uc.get_usage_recorder() is None


def test_worker_startup_refuses_to_start_and_builds_no_service(store_unreachable, monkeypatch):
    import worker

    monkeypatch.setattr("worker_liveness.start_heartbeat", lambda ctx: None)
    built = []
    monkeypatch.setattr(worker, "MemoryService", lambda: built.append(1) or MagicMock())
    ctx: dict = {}
    with pytest.raises(StoreDown):
        asyncio.run(worker.startup(ctx))
    assert built == [] and "service" not in ctx


def test_api_lifespan_refuses_to_start_before_the_service_builds_its_clients(store_unreachable, monkeypatch):
    import main

    monkeypatch.setattr(type(settings), "validate_required", lambda self: None)

    async def no_warm():
        return None

    monkeypatch.setattr("tokens.warm_revocations", no_warm)
    service = MagicMock()
    monkeypatch.setattr(main, "_service", service)

    async def enter():
        async with main.lifespan(MagicMock()):
            pass

    with pytest.raises(StoreDown):
        asyncio.run(enter())
    service._get_memory.assert_not_called()
