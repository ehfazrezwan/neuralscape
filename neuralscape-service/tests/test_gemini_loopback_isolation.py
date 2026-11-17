"""The loopback fixture leaves nothing behind that still points at the stand-in server.

``NS_GEMINI_BASE_URL`` patches the SDK client class for the whole process.  The loopback integration tests
set it, so a test that runs later and builds a Gemini client of its own (a real embedding call) must get
the default endpoint again.  These tests drive that fixture's own generator with fakes, no store needed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import provider_routing
from google import genai

LOOPBACK = Path(__file__).resolve().parent.parent / "integration_tests" / "test_gemini_base_url_loopback.py"
DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/"


@pytest.fixture(autouse=True)
def _clean():
    provider_routing.reset()
    yield
    provider_routing.reset()


@pytest.fixture()
def loopback_module():
    spec = importlib.util.spec_from_file_location("nsit_loopback_under_test", LOOPBACK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_generator(module):
    fn = module.routed_service
    fn = getattr(fn, "__wrapped__", None) or getattr(fn, "_get_wrapped_function", lambda: fn)()
    stand_in = SimpleNamespace(server_address=("127.0.0.1", 9))
    unique_ns = SimpleNamespace(qdrant_collection="c", on_cleanup=lambda f: None)
    stores = SimpleNamespace(qdrant_url="http://127.0.0.1:1")
    return fn(lambda **fields: None, stores, unique_ns, stand_in)


def _base_url(client):
    return client._api_client._http_options.base_url


class _Service:
    """Stands in for ``MemoryService``: builds its client under the override, as the real one does."""

    fail = False

    def __init__(self):
        self._genai_model = None
        self._graphiti = None
        self._memory = None
        provider_routing.apply_gemini_base_url("http://127.0.0.1:9")
        if _Service.fail:
            raise RuntimeError("service did not build")

    def _get_genai_client(self):
        self._genai_model = genai.Client(api_key="k")
        return self._genai_model

    def close(self):
        pass


def _patch_service(monkeypatch, fail=False):
    import memory_service

    monkeypatch.setattr(memory_service, "MemoryService", _Service)
    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=object))
    _Service.fail = fail


def test_after_teardown_new_clients_have_the_default_endpoint_and_timeout(loopback_module, monkeypatch):
    _patch_service(monkeypatch)
    gen = _fixture_generator(loopback_module)
    service = next(gen)
    assert _base_url(service._get_genai_client()) == "http://127.0.0.1:9"
    with pytest.raises(StopIteration):
        next(gen)

    from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

    assert provider_routing._APPLIED == []
    assert _base_url(genai.Client(api_key="k")) == DEFAULT_ENDPOINT
    embedder = GeminiEmbedder(GeminiEmbedderConfig(api_key="k"))  # what the real-Gemini MRL test builds
    assert _base_url(embedder.client) == DEFAULT_ENDPOINT
    assert embedder.client._api_client._http_options.timeout == 60_000


def test_a_service_that_fails_to_build_does_not_leave_the_override_applied(loopback_module, monkeypatch):
    _patch_service(monkeypatch, fail=True)
    gen = _fixture_generator(loopback_module)
    with pytest.raises(RuntimeError, match="did not build"):
        next(gen)
    assert provider_routing._APPLIED == []
    assert _base_url(genai.Client(api_key="k")) == DEFAULT_ENDPOINT
