"""``NS_GEMINI_BASE_URL`` reaches every Gemini client the process builds, in every process.

The service and the vendored memory and graph libraries each build their own clients; unless the
override is applied to the client class they would still call the default endpoint.  A client built
with an address of its own keeps it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import provider_routing
from google import genai
from google.genai import types


@pytest.fixture(autouse=True)
def _restore_the_sdk_constructor():
    provider_routing.reset()
    yield
    provider_routing.reset()


def _base_url(client) -> str | None:
    return client._api_client._http_options.base_url


def test_unset_changes_nothing():
    original = genai.Client.__init__
    assert provider_routing.apply_gemini_base_url("") is False
    assert provider_routing.apply_from_settings(SimpleNamespace(ns_gemini_base_url="")) is False
    assert provider_routing.apply_from_settings(SimpleNamespace(ns_gemini_base_url=None)) is False  # a mocked setting
    assert genai.Client.__init__ is original


def test_every_new_client_is_routed_and_keeps_its_own_options():
    assert provider_routing.apply_gemini_base_url("https://gateway.example.test/g") is True
    plain = genai.Client(api_key="k")
    assert _base_url(plain).rstrip("/") == "https://gateway.example.test/g"
    timed = genai.Client(api_key="k", http_options=types.HttpOptions(timeout=1234))
    assert _base_url(timed).rstrip("/") == "https://gateway.example.test/g"
    assert timed._api_client._http_options.timeout == 1234  # the library's own bounded timeout survives
    as_dict = genai.Client(api_key="k", http_options={"timeout": 99})
    assert _base_url(as_dict).rstrip("/") == "https://gateway.example.test/g"


def test_a_shared_options_object_is_not_changed_behind_its_back():
    options = types.HttpOptions(timeout=5)
    provider_routing.apply_gemini_base_url("https://gateway.example.test/g")
    genai.Client(api_key="k", http_options=options)
    assert options.base_url is None


def test_applying_twice_is_idempotent_and_a_different_url_is_refused():
    assert provider_routing.apply_gemini_base_url("https://a.example.test") is True
    assert provider_routing.apply_gemini_base_url("https://a.example.test") is True
    with pytest.raises(ValueError, match="cannot change"):
        provider_routing.apply_gemini_base_url("https://b.example.test")
    assert _base_url(genai.Client(api_key="k")).rstrip("/") == "https://a.example.test"


def test_a_non_http_url_is_refused():
    with pytest.raises(ValueError, match="http"):
        provider_routing.apply_gemini_base_url("ftp://x/y")


def test_an_explicit_base_url_is_not_overridden():
    provider_routing.apply_gemini_base_url("https://gateway.example.test/g")
    own = genai.Client(api_key="k", http_options=types.HttpOptions(base_url="https://own.example.test/v1"))
    assert _base_url(own).rstrip("/") == "https://own.example.test/v1"
    own_dict = genai.Client(api_key="k", http_options={"base_url": "https://own.example.test/v2"})
    assert _base_url(own_dict).rstrip("/") == "https://own.example.test/v2"
    # an options object without an address still gets the configured one
    plain = genai.Client(api_key="k", http_options=types.HttpOptions(timeout=7))
    assert _base_url(plain).rstrip("/") == "https://gateway.example.test/g"


def test_the_service_applies_it_before_it_builds_any_client(monkeypatch):
    """API, worker queues and a stdio MCP process all build clients through the service."""
    import threading

    import memory.core as core
    from config import settings

    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "ns_gemini_base_url", "https://gateway.example.test/svc")
    svc = core.CoreMixin.__new__(core.CoreMixin)
    svc._genai_model = None
    svc._init_lock = threading.Lock()
    client = svc._get_genai_client()  # the real SDK client class, routed
    assert provider_routing._APPLIED == ["https://gateway.example.test/svc"]
    assert _base_url(client).rstrip("/") == "https://gateway.example.test/svc"


def test_the_service_builds_an_unchanged_client_when_unset(monkeypatch):
    import threading

    import memory.core as core
    from config import settings

    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "ns_gemini_base_url", "")
    svc = core.CoreMixin.__new__(core.CoreMixin)
    svc._genai_model = None
    svc._init_lock = threading.Lock()
    client = svc._get_genai_client()
    assert provider_routing._APPLIED == []
    assert "generativelanguage.googleapis.com" in _base_url(client)


def test_invalid_endpoint_settings_are_rejected():
    import config

    for bad in ("ftp://x/y", "https://user:pw@host/x", "not a url", "https://"):
        with pytest.raises(Exception, match="NS_GEMINI_BASE_URL"):
            config.Settings(
                ns_gemini_base_url=bad, google_api_key="k", neo4j_password="p", neuralscape_auth_disabled=True
            ).validate_required()
    config.Settings(
        ns_gemini_base_url="https://ok.example.test/x", google_api_key="k", neo4j_password="p",
        neuralscape_auth_disabled=True,
    ).validate_required()


def test_the_worker_start_up_routes_before_the_service_exists():
    import inspect

    import worker

    source = inspect.getsource(worker.startup)
    assert source.index("apply_gemini_base_url") < source.index("MemoryService()")
