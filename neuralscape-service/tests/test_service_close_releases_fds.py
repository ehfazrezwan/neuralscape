"""MemoryService.close() must release every descriptor the service holds, and the
integration conftest must close services a test forgot to close.

The gate crashed with "Too many open files" because integration tests build a
MemoryService (bridge loop thread, driver pool, Qdrant and embedder HTTP
clients) and never close it. macOS ``/dev/fd`` (Linux ``/proc/self/fd``) gives the
open-descriptor count without any store.
"""
import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest

from mem0.memory.graphiti_memory import _AsyncBridge

FD_DIR = "/dev/fd" if os.path.isdir("/dev/fd") else "/proc/self/fd"
SLACK = 4
CYCLES = 20


def open_fds() -> int:
    return len(os.listdir(FD_DIR))


class SocketClient:
    """Stands in for an HTTP client holding a pooled connection until closed."""

    def __init__(self):
        self.a, self.b = socket.socketpair()
        self.closed = 0

    def close(self):
        self.closed += 1
        self.a.close()
        self.b.close()


class FakeGraphiti:
    async def close(self):
        return None


def build_service():
    from memory_service import MemoryService

    svc = MemoryService()
    svc._bridge = _AsyncBridge()
    svc._graphiti = FakeGraphiti()
    svc._memory = Mock()
    svc._memory.close = Mock()
    svc._memory.vector_store.client = SocketClient()
    svc._memory.embedding_model.client = SocketClient()
    return svc


def test_build_and_close_cycles_leave_the_open_fd_count_flat():
    before = open_fds()
    for _ in range(CYCLES):
        build_service().close()
    assert open_fds() - before <= SLACK


def test_close_closes_the_qdrant_and_embedder_clients_once_and_is_idempotent():
    svc = build_service()
    qdrant, embedder = svc._memory.vector_store.client, svc._memory.embedding_model.client
    svc.close()
    svc.close()
    assert (qdrant.closed, embedder.closed) == (1, 1)
    assert not svc._bridge._thread.is_alive()


def test_close_calls_mem0_memory_close_to_release_the_history_connection():
    svc = build_service()
    memory = svc._memory
    svc.close()
    svc.close()
    memory.close.assert_called_once_with()


def test_a_failing_client_close_does_not_raise_or_skip_the_other_client():
    svc = build_service()
    svc._memory.vector_store.client.close = Mock(side_effect=RuntimeError("boom"))
    embedder = svc._memory.embedding_model.client
    svc.close()
    assert embedder.closed == 1


INNER_TEST = textwrap.dedent(
    '''
    import os
    import third_party_telemetry  # noqa: F401
    import pytest
    from mem0.memory.graphiti_memory import _AsyncBridge

    FD_DIR = "/dev/fd" if os.path.isdir("/dev/fd") else "/proc/self/fd"
    built = []
    BASE = []

    class FakeGraphiti:
        async def close(self):
            return None

    @pytest.mark.parametrize("n", range(%(cycles)d))
    def test_forgets_to_close(n):
        if not BASE:
            BASE.append(len(os.listdir(FD_DIR)))
        from memory_service import MemoryService
        svc = MemoryService()
        svc._bridge = _AsyncBridge()
        svc._graphiti = FakeGraphiti()
        built.append(svc)

    def test_every_forgotten_service_was_closed_and_fds_are_flat():
        assert len(built) == %(cycles)d
        assert all(not s._bridge._thread.is_alive() for s in built)
        assert len(os.listdir(FD_DIR)) - BASE[0] <= %(slack)d
    '''
)


def test_integration_conftest_resets_copies_of_a_closed_shared_service(tmp_path):
    service_root = Path(__file__).resolve().parent.parent
    (tmp_path / "conftest.py").write_text((service_root / "integration_tests" / "conftest.py").read_text())
    (tmp_path / "test_inner2.py").write_text(textwrap.dedent(
        '''
        import third_party_telemetry  # noqa: F401
        import sys, types
        seen = []

        def test_first():
            import memory_service
            from knowledge.ns_memory import NSMemorySystem
            svc = memory_service.get_shared_service()
            from mem0.memory.graphiti_memory import _AsyncBridge
            svc._bridge = _AsyncBridge()  # something to release, so close() marks it closed
            fake_main = types.ModuleType("main")
            fake_main._service = svc
            sys.modules["main"] = fake_main
            fake_mcp = types.ModuleType("mcp_server")
            fake_mcp._service = svc
            sys.modules["mcp_server"] = fake_mcp
            ns = NSMemorySystem()
            ns._get_service()
            seen.extend([svc, fake_main, ns, fake_mcp])

        def test_second_sees_fresh_copies():
            import memory_service
            svc, fake_main, ns, fake_mcp = seen
            assert svc._closed
            assert memory_service._shared_service is not svc
            assert fake_main._service is not svc and not getattr(fake_main._service, "_closed", False)
            assert ns._service is None
            assert fake_mcp._service is not svc and not getattr(fake_mcp._service, "_closed", False)
        '''))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(service_root), *sys.path]))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_inner2.py"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout[-2500:] + result.stderr[-1500:]


def test_integration_conftest_closes_services_a_test_forgot_to_close(tmp_path):
    service_root = Path(__file__).resolve().parent.parent
    (tmp_path / "conftest.py").write_text((service_root / "integration_tests" / "conftest.py").read_text())
    (tmp_path / "test_inner.py").write_text(INNER_TEST % {"cycles": CYCLES, "slack": SLACK})
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(service_root), *sys.path]))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_inner.py"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout[-2500:] + result.stderr[-1500:]


def test_close_on_a_stopped_bridge_loop_releases_the_clients_but_abandons_the_driver_sockets():
    """Spells out the one path where sockets are NOT released: the Graphiti driver, whose
    close needs the bridge loop. The Qdrant and embedder clients and the loop itself are."""
    from unittest.mock import patch

    class DriverGraphiti:
        def __init__(self):
            self.sockets = SocketClient()

        async def close(self):
            self.sockets.close()

    svc = build_service()
    svc._graphiti = driver = DriverGraphiti()
    qdrant, embedder = svc._memory.vector_store.client, svc._memory.embedding_model.client
    svc._bridge._loop.call_soon_threadsafe(svc._bridge._loop.stop)
    svc._bridge._thread.join(5)
    try:
        with patch("memory.core._CLOSE_DRIVER_TIMEOUT_S", 0.3):
            svc.close()
        assert (qdrant.closed, embedder.closed) == (1, 1)  # released
        assert svc._bridge._loop.is_closed() and not svc._bridge._thread.is_alive()  # released
        assert driver.sockets.closed == 0  # NOT released: abandoned to the OS (bounded by the process)
    finally:
        driver.sockets.close()
