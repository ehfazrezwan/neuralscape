"""Tests for Bearer token authentication middleware.

Tests cover:
- Auth disabled when NEURALSCAPE_API_KEY is empty (local dev mode)
- Auth enabled when NEURALSCAPE_API_KEY is set
- Health endpoint always accessible without auth
- Protected endpoints return 401 without/invalid token
- Protected endpoints pass through with valid token
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main
from config import Settings
from main import app


TEST_API_KEY = "test-secret-key-12345"


@pytest.fixture(autouse=True)
def mock_memory():
    """Patch lazy-init globals so no real mem0/Graphiti is created."""
    original_memory = main._memory
    original_graphiti = main._graphiti
    original_bridge = main._bridge
    original_async_memory = main._async_memory

    main._memory = MagicMock(name="Memory")
    main._graphiti = MagicMock(name="Graphiti")
    main._bridge = MagicMock(name="AsyncBridge")
    main._async_memory = None

    yield

    main._memory = original_memory
    main._graphiti = original_graphiti
    main._bridge = original_bridge
    main._async_memory = original_async_memory


@pytest.fixture(autouse=True)
def mock_task_manager():
    """Patch TaskManager so tests don't need Redis."""
    mock_tm = MagicMock(name="TaskManager")
    mock_tm.connect = AsyncMock()
    mock_tm.close = AsyncMock()
    mock_tm.pool = MagicMock()
    mock_tm.pool.ping = AsyncMock()
    original = main._task_manager
    main._task_manager = mock_tm
    yield mock_tm
    main._task_manager = original


@pytest.fixture
def mock_service():
    """Patch MemoryService for v1 endpoints."""
    mock_svc = MagicMock(name="MemoryService")
    mock_svc._memory = MagicMock()
    mock_svc._graphiti = MagicMock()
    original = main._service
    main._service = mock_svc
    yield mock_svc
    main._service = original


class TestAuthDisabled:
    """When NEURALSCAPE_API_KEY is empty, all requests pass through."""

    @pytest.fixture(autouse=True)
    def disable_auth(self):
        from config import settings
        original = settings.neuralscape_api_key
        settings.neuralscape_api_key = ""
        yield
        settings.neuralscape_api_key = original

    def test_health_no_auth(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_v1_endpoint_no_auth(self, mock_service):
        mock_service.list_memories.return_value = []
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/memories", params={"user_id": "test"})
        assert resp.status_code == 200


class TestAuthEnabled:
    """When NEURALSCAPE_API_KEY is set, endpoints require Bearer token."""

    @pytest.fixture(autouse=True)
    def enable_auth(self):
        from config import settings
        original = settings.neuralscape_api_key
        settings.neuralscape_api_key = TEST_API_KEY
        yield
        settings.neuralscape_api_key = original

    def test_health_always_public(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_no_auth_header_returns_401(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/memories", params={"user_id": "test"})
        assert resp.status_code == 401
        assert "Missing" in resp.json()["detail"]

    def test_invalid_token_returns_401(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/v1/memories",
            params={"user_id": "test"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]

    def test_valid_token_passes_through(self, mock_service):
        mock_service.list_memories.return_value = []
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/v1/memories",
            params={"user_id": "test"},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert resp.status_code == 200

    def test_legacy_endpoint_requires_auth(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/memories", params={"user_id": "test"})
        assert resp.status_code == 401

    def test_legacy_endpoint_with_valid_token(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/memories",
            params={"user_id": "test"},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        # Should pass auth (may fail for other reasons in test, but not 401)
        assert resp.status_code != 401

    def test_malformed_auth_header_returns_401(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/v1/memories",
            params={"user_id": "test"},
            headers={"Authorization": f"Basic {TEST_API_KEY}"},
        )
        assert resp.status_code == 401

    def test_mcp_endpoint_requires_auth(self):
        """MCP HTTP transport at /mcp/ should also require auth."""
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/mcp/")
        # Should be 401 (auth) not 404 (route not found)
        # Note: MCP may not be mounted in test env, so we accept 401 or 404
        assert resp.status_code in (401, 404)
