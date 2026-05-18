"""Tests for the NeuralScape extension registry.

Covers discovery, lifecycle, event dispatch, route mounting,
graceful failure handling, and the /v1/extensions listing endpoint.
"""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from extensions import EmitResult, ExtensionRegistry
from extensions.base import ExtensionManifest, NeuralscapeExtension
from extensions.events import EventType


# ── Helpers ──────────────────────────────────────


class DummyExtension:
    """Minimal working extension for tests."""

    def __init__(
        self,
        name: str = "dummy",
        version: str = "1.0.0",
        hooks: list[str] | None = None,
        routes: bool = False,
        fail_startup: bool = False,
        fail_shutdown: bool = False,
        fail_event: bool = False,
    ):
        self.manifest = ExtensionManifest(
            name=name,
            version=version,
            description=f"Test extension: {name}",
            hooks=hooks or [],
        )
        self._routes = routes
        self._fail_startup = fail_startup
        self._fail_shutdown = fail_shutdown
        self._fail_event = fail_event
        self.started = False
        self.stopped = False
        self.events_received: list[tuple[str, dict]] = []

    async def startup(self) -> None:
        if self._fail_startup:
            raise RuntimeError("startup boom")
        self.started = True

    async def shutdown(self) -> None:
        if self._fail_shutdown:
            raise RuntimeError("shutdown boom")
        self.stopped = True

    async def on_event(self, event_type: str, payload: dict) -> Optional[dict]:
        if self._fail_event:
            raise RuntimeError("event boom")
        self.events_received.append((event_type, payload))
        return {"handled_by": self.manifest.name}

    def get_routes(self) -> Optional[APIRouter]:
        if not self._routes:
            return None
        router = APIRouter()

        @router.get("/ping")
        async def ping():
            return {"pong": self.manifest.name}

        return router


# ── Registration ─────────────────────────────────


class TestRegistration:
    def test_register_extension(self):
        registry = ExtensionRegistry()
        ext = DummyExtension(name="test-ext")
        registry.register(ext)
        assert "test-ext" in registry.extensions
        assert registry.extensions["test-ext"].status == "registered"

    def test_register_duplicate_skipped(self):
        registry = ExtensionRegistry()
        ext1 = DummyExtension(name="dup")
        ext2 = DummyExtension(name="dup", version="2.0.0")
        registry.register(ext1)
        registry.register(ext2)
        # Only the first one should be registered
        assert registry.extensions["dup"].manifest.version == "1.0.0"

    def test_list_extensions(self):
        registry = ExtensionRegistry()
        registry.register(DummyExtension(name="ext-a", hooks=["memory_stored"]))
        registry.register(DummyExtension(name="ext-b"))
        listing = registry.list_extensions()
        assert len(listing) == 2
        names = {e["name"] for e in listing}
        assert names == {"ext-a", "ext-b"}
        # Check structure
        ext_a = next(e for e in listing if e["name"] == "ext-a")
        assert ext_a["version"] == "1.0.0"
        assert ext_a["status"] == "registered"
        assert ext_a["hooks"] == ["memory_stored"]

    def test_list_extensions_empty(self):
        registry = ExtensionRegistry()
        assert registry.list_extensions() == []


# ── Lifecycle ────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_startup_all(self):
        registry = ExtensionRegistry()
        ext = DummyExtension(name="life")
        registry.register(ext)
        await registry.startup_all()
        assert ext.started is True
        assert registry.extensions["life"].status == "started"

    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        registry = ExtensionRegistry()
        ext = DummyExtension(name="life")
        registry.register(ext)
        await registry.startup_all()
        await registry.shutdown_all()
        assert ext.stopped is True
        assert registry.extensions["life"].status == "stopped"

    @pytest.mark.asyncio
    async def test_shutdown_skips_not_started(self):
        registry = ExtensionRegistry()
        ext = DummyExtension(name="never-started")
        registry.register(ext)
        # Don't call startup_all
        await registry.shutdown_all()
        assert ext.stopped is False

    @pytest.mark.asyncio
    async def test_startup_failure_marks_failed(self):
        registry = ExtensionRegistry()
        ext = DummyExtension(name="bad", fail_startup=True)
        registry.register(ext)
        await registry.startup_all()
        assert ext.started is False
        assert registry.extensions["bad"].status == "failed"

    @pytest.mark.asyncio
    async def test_startup_failure_doesnt_block_others(self):
        registry = ExtensionRegistry()
        bad = DummyExtension(name="bad", fail_startup=True)
        good = DummyExtension(name="good")
        registry.register(bad)
        registry.register(good)
        await registry.startup_all()
        assert good.started is True
        assert registry.extensions["good"].status == "started"
        assert registry.extensions["bad"].status == "failed"

    @pytest.mark.asyncio
    async def test_shutdown_failure_doesnt_block_others(self):
        registry = ExtensionRegistry()
        bad = DummyExtension(name="bad-shutdown", fail_shutdown=True)
        good = DummyExtension(name="good-shutdown")
        registry.register(bad)
        registry.register(good)
        await registry.startup_all()
        await registry.shutdown_all()
        assert good.stopped is True


# ── Event Dispatch ───────────────────────────────


class TestEventDispatch:
    @pytest.mark.asyncio
    async def test_event_dispatched_to_matching_hooks(self):
        registry = ExtensionRegistry()
        ext = DummyExtension(name="listener", hooks=["memory_stored"])
        registry.register(ext)
        await registry.startup_all()

        result = await registry.emit_event("memory_stored", {"memory_id": "abc"})
        assert isinstance(result, EmitResult)
        assert len(result.responses) == 1
        assert result.responses[0] == {"handled_by": "listener"}
        assert result.notified_count == 1
        assert ext.events_received == [("memory_stored", {"memory_id": "abc"})]

    @pytest.mark.asyncio
    async def test_event_not_dispatched_to_non_matching(self):
        registry = ExtensionRegistry()
        ext = DummyExtension(name="listener", hooks=["session_start"])
        registry.register(ext)
        await registry.startup_all()

        result = await registry.emit_event("memory_stored", {"memory_id": "abc"})
        assert len(result.responses) == 0
        assert result.notified_count == 0
        assert ext.events_received == []

    @pytest.mark.asyncio
    async def test_event_skips_failed_extensions(self):
        registry = ExtensionRegistry()
        ext = DummyExtension(name="failed-ext", hooks=["memory_stored"], fail_startup=True)
        registry.register(ext)
        await registry.startup_all()

        result = await registry.emit_event("memory_stored", {"memory_id": "abc"})
        assert len(result.responses) == 0
        assert result.notified_count == 0

    @pytest.mark.asyncio
    async def test_event_failure_doesnt_block_others(self):
        registry = ExtensionRegistry()
        bad = DummyExtension(name="bad-handler", hooks=["memory_stored"], fail_event=True)
        good = DummyExtension(name="good-handler", hooks=["memory_stored"])
        registry.register(bad)
        registry.register(good)
        await registry.startup_all()

        result = await registry.emit_event("memory_stored", {"memory_id": "abc"})
        # Only good handler's response, but both were notified
        assert len(result.responses) == 1
        assert result.responses[0] == {"handled_by": "good-handler"}
        assert result.notified_count == 2

    @pytest.mark.asyncio
    async def test_multiple_extensions_receive_event(self):
        registry = ExtensionRegistry()
        ext1 = DummyExtension(name="ext-1", hooks=["session_start"])
        ext2 = DummyExtension(name="ext-2", hooks=["session_start"])
        registry.register(ext1)
        registry.register(ext2)
        await registry.startup_all()

        result = await registry.emit_event("session_start", {"session_id": "s1"})
        assert len(result.responses) == 2
        assert result.notified_count == 2
        assert ext1.events_received == [("session_start", {"session_id": "s1"})]
        assert ext2.events_received == [("session_start", {"session_id": "s1"})]

    @pytest.mark.asyncio
    async def test_notified_count_includes_none_responders(self):
        """Extension returning None is still counted as notified."""

        class NoneResponder:
            def __init__(self):
                self.manifest = ExtensionManifest(
                    name="none-responder",
                    version="1.0.0",
                    description="Returns None",
                    hooks=["memory_stored"],
                )

            async def startup(self) -> None:
                pass

            async def shutdown(self) -> None:
                pass

            async def on_event(self, event_type: str, payload: dict) -> Optional[dict]:
                return None

            def get_routes(self) -> Optional[APIRouter]:
                return None

        registry = ExtensionRegistry()
        responder = DummyExtension(name="responder", hooks=["memory_stored"])
        none_ext = NoneResponder()
        registry.register(responder)
        registry.register(none_ext)
        await registry.startup_all()

        result = await registry.emit_event("memory_stored", {"memory_id": "abc"})
        assert result.notified_count == 2
        assert len(result.responses) == 1  # Only DummyExtension returns non-None


# ── Route Mounting ───────────────────────────────


class TestRouteMounting:
    def test_mount_routes(self):
        registry = ExtensionRegistry()
        ext = DummyExtension(name="routed", routes=True)
        registry.register(ext)

        test_app = FastAPI()
        registry.mount_routes(test_app)

        client = TestClient(test_app)
        resp = client.get("/v1/extensions/routed/ping")
        assert resp.status_code == 200
        assert resp.json() == {"pong": "routed"}

    def test_mount_routes_no_routes(self):
        registry = ExtensionRegistry()
        ext = DummyExtension(name="no-routes", routes=False)
        registry.register(ext)

        test_app = FastAPI()
        registry.mount_routes(test_app)

        client = TestClient(test_app)
        resp = client.get("/v1/extensions/no-routes/ping")
        assert resp.status_code == 404

    def test_mount_multiple_extensions(self):
        registry = ExtensionRegistry()
        ext1 = DummyExtension(name="ext-a", routes=True)
        ext2 = DummyExtension(name="ext-b", routes=True)
        registry.register(ext1)
        registry.register(ext2)

        test_app = FastAPI()
        registry.mount_routes(test_app)

        client = TestClient(test_app)
        assert client.get("/v1/extensions/ext-a/ping").json() == {"pong": "ext-a"}
        assert client.get("/v1/extensions/ext-b/ping").json() == {"pong": "ext-b"}


# ── Discovery ────────────────────────────────────


class TestDiscovery:
    @pytest.mark.asyncio
    async def test_discover_empty(self):
        """Discovery with no local extensions and no env var should succeed."""
        registry = ExtensionRegistry()
        with patch.dict("os.environ", {"NEURALSCAPE_EXTENSIONS": ""}, clear=False):
            # Patch _discover_local to avoid scanning real filesystem
            with patch.object(registry, "_discover_local"):
                await registry.discover()
        assert len(registry.extensions) == 0

    @pytest.mark.asyncio
    async def test_discover_env_var(self):
        """Extensions listed in NEURALSCAPE_EXTENSIONS env var are loaded."""
        registry = ExtensionRegistry()

        # Create a mock module with a valid extension class
        mock_module = MagicMock()
        mock_module.extension = DummyExtension(name="env-ext")

        with patch.dict("os.environ", {"NEURALSCAPE_EXTENSIONS": "fake.module"}):
            with patch.object(registry, "_discover_local"):
                with patch("extensions.importlib.import_module", return_value=mock_module):
                    await registry.discover()

        assert "env-ext" in registry.extensions

    @pytest.mark.asyncio
    async def test_discover_env_var_multiple(self):
        """Multiple comma-separated extensions are loaded."""
        registry = ExtensionRegistry()

        def fake_import(name):
            mod = MagicMock()
            mod.extension = DummyExtension(name=f"ext-{name.split('.')[-1]}")
            return mod

        with patch.dict("os.environ", {"NEURALSCAPE_EXTENSIONS": "pkg.one,pkg.two"}):
            with patch.object(registry, "_discover_local"):
                with patch("extensions.importlib.import_module", side_effect=fake_import):
                    await registry.discover()

        assert "ext-one" in registry.extensions
        assert "ext-two" in registry.extensions

    @pytest.mark.asyncio
    async def test_discover_bad_import_handled(self):
        """Failed imports are logged but don't crash discovery."""
        registry = ExtensionRegistry()

        with patch.dict("os.environ", {"NEURALSCAPE_EXTENSIONS": "nonexistent.module"}):
            with patch.object(registry, "_discover_local"):
                with patch(
                    "extensions.importlib.import_module",
                    side_effect=ImportError("no such module"),
                ):
                    await registry.discover()

        assert len(registry.extensions) == 0


# ── /v1/extensions Endpoint ──────────────────────


class TestExtensionsEndpoint:
    @pytest.fixture(autouse=True)
    def mock_globals(self):
        """Patch main module globals to avoid real service initialization."""
        import main

        original_memory = main._memory
        original_graphiti = main._graphiti
        original_bridge = main._bridge
        original_service = main._service
        original_tm = main._task_manager
        original_registry = main._extension_registry

        main._memory = MagicMock()
        main._graphiti = MagicMock()
        main._bridge = MagicMock()
        main._service = MagicMock()
        mock_tm = MagicMock()
        mock_tm.connect = AsyncMock()
        mock_tm.close = AsyncMock()
        mock_tm.pool = None
        main._task_manager = mock_tm

        yield

        main._memory = original_memory
        main._graphiti = original_graphiti
        main._bridge = original_bridge
        main._service = original_service
        main._task_manager = original_tm
        main._extension_registry = original_registry

    @pytest.fixture
    def client(self):
        from main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_list_extensions_empty(self, client):
        import main
        main._extension_registry = ExtensionRegistry()
        resp = client.get("/v1/extensions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["extensions"] == []

    def test_list_extensions_with_entries(self, client):
        import main
        registry = ExtensionRegistry()
        registry.register(DummyExtension(name="ext-a", hooks=["memory_stored"]))
        registry.register(DummyExtension(name="ext-b"))
        main._extension_registry = registry

        resp = client.get("/v1/extensions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["extensions"]) == 2
        names = {e["name"] for e in data["extensions"]}
        assert names == {"ext-a", "ext-b"}

    def test_emit_event_endpoint(self, client):
        import main

        registry = ExtensionRegistry()
        ext = DummyExtension(name="evt-listener", hooks=["memory_stored"])
        registry.register(ext)
        # Manually set status to started so events are dispatched
        registry._extensions["evt-listener"].status = "started"
        main._extension_registry = registry

        resp = client.post(
            "/v1/extensions/events",
            json={
                "event_type": "memory_stored",
                "payload": {"user_id": "u1", "memory_id": "m1", "content": "test fact"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["event_type"] == "memory_stored"
        assert data["extensions_notified"] == 1

    def test_emit_event_no_listeners(self, client):
        import main
        main._extension_registry = ExtensionRegistry()

        resp = client.post(
            "/v1/extensions/events",
            json={
                "event_type": "memory_stored",
                "payload": {"user_id": "u1", "memory_id": "m1", "content": "test"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["extensions_notified"] == 0


# ── Event Schema Models ──────────────────────────


class TestEventSchemas:
    def test_event_type_enum_values(self):
        assert EventType.CONVERSATION_TURN == "conversation_turn"
        assert EventType.SESSION_START == "session_start"
        assert EventType.SESSION_END == "session_end"
        assert EventType.MEMORY_STORED == "memory_stored"
        assert EventType.COMPILE_REQUESTED == "compile_requested"

    def test_event_payload_models(self):
        from extensions.events import (
            CompileRequestedEvent,
            ConversationTurnEvent,
            EVENT_PAYLOAD_MODELS,
            MemoryStoredEvent,
            SessionEndEvent,
            SessionStartEvent,
        )

        assert len(EVENT_PAYLOAD_MODELS) == 5

        # Validate each model can be instantiated
        ConversationTurnEvent(user_id="u1", messages=[{"role": "user", "content": "hi"}])
        SessionStartEvent(user_id="u1", session_id="s1")
        SessionEndEvent(user_id="u1", session_id="s1")
        MemoryStoredEvent(user_id="u1", memory_id="m1", content="fact")
        CompileRequestedEvent(user_id="u1")


# ── Protocol Compliance ──────────────────────────


class TestProtocolCompliance:
    def test_dummy_extension_satisfies_protocol(self):
        ext = DummyExtension(name="proto-test")
        assert isinstance(ext, NeuralscapeExtension)

    def test_manifest_validation(self):
        manifest = ExtensionManifest(
            name="test",
            version="1.0.0",
            description="Test extension",
            author="Tester",
            hooks=["memory_stored", "session_start"],
        )
        assert manifest.name == "test"
        assert manifest.hooks == ["memory_stored", "session_start"]


# ── Manifest Name Validation ───────────────────


class TestManifestNameValidation:
    def test_valid_lowercase(self):
        m = ExtensionManifest(name="my-ext", version="1.0.0", description="test")
        assert m.name == "my-ext"

    def test_valid_with_digits(self):
        m = ExtensionManifest(name="ext-2", version="1.0.0", description="test")
        assert m.name == "ext-2"

    def test_valid_single_char(self):
        m = ExtensionManifest(name="x", version="1.0.0", description="test")
        assert m.name == "x"

    def test_valid_all_digits(self):
        m = ExtensionManifest(name="42", version="1.0.0", description="test")
        assert m.name == "42"

    def test_invalid_uppercase(self):
        with pytest.raises(ValueError):
            ExtensionManifest(name="MyExtension", version="1.0.0", description="test")

    def test_invalid_spaces(self):
        with pytest.raises(ValueError):
            ExtensionManifest(name="my extension", version="1.0.0", description="test")

    def test_invalid_starts_with_hyphen(self):
        with pytest.raises(ValueError):
            ExtensionManifest(name="-bad", version="1.0.0", description="test")

    def test_invalid_special_chars(self):
        with pytest.raises(ValueError):
            ExtensionManifest(name="ext@foo", version="1.0.0", description="test")

    def test_invalid_underscore(self):
        with pytest.raises(ValueError):
            ExtensionManifest(name="my_ext", version="1.0.0", description="test")

    def test_invalid_empty(self):
        with pytest.raises(ValueError):
            ExtensionManifest(name="", version="1.0.0", description="test")
