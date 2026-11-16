"""Every Gemini client the service builds carries ``GEMINI_REQUEST_TIMEOUT_S``.

A Gemini endpoint that accepts the connection and never answers used to block the calling
request or worker job without bound: the service's own client, the conversation compiler's
client and mem0's Gemini LLM/embedder were all built without a request timeout.
``HttpOptions.timeout`` is in MILLISECONDS; the setting is in seconds.
"""

from __future__ import annotations

import socket
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import config
from config import Settings


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, google_api_key="k", neo4j_password="x", **kw)


# ── the setting ──────────────────────────────────────────────────────────────


def test_default_is_120_seconds(monkeypatch):
    monkeypatch.delenv("GEMINI_REQUEST_TIMEOUT_S", raising=False)
    assert _settings().gemini_request_timeout_s == 120.0
    assert _settings().gemini_http_options().timeout == 120_000


def test_env_override_and_fractional_seconds(monkeypatch):
    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT_S", "2.5")
    s = _settings()
    assert s.gemini_request_timeout_s == 2.5
    assert s.gemini_http_options().timeout == 2500


@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5])
def test_rejects_zero_and_negative(bad, monkeypatch):
    monkeypatch.delenv("GEMINI_REQUEST_TIMEOUT_S", raising=False)
    with pytest.raises(ValidationError):
        _settings(gemini_request_timeout_s=bad)
    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT_S", str(bad))
    with pytest.raises(ValidationError):
        _settings()


# ── each client is constructed with the timeout ──────────────────────────────


def _timeout_of(client_cls) -> int:
    client_cls.assert_called_once()
    return client_cls.call_args.kwargs["http_options"].timeout


def test_service_client_gets_timeout(monkeypatch):
    import memory.core as core

    monkeypatch.setattr(core.settings, "gemini_request_timeout_s", 7.0)
    svc = object.__new__(core.CoreMixin)
    svc._genai_model = None
    svc._init_lock = threading.Lock()
    with patch.object(core.genai, "Client") as client_cls:
        svc._get_genai_client()
    assert _timeout_of(client_cls) == 7000


def test_compiler_client_gets_timeout(monkeypatch):
    from extensions.conversation_compiler import compile as comp

    monkeypatch.setattr(comp.core_settings, "gemini_request_timeout_s", 9.0)
    with patch.object(comp.genai, "Client") as client_cls:
        client_cls.return_value.models.generate_content.return_value = MagicMock(text="ok")
        assert comp._call_gemini("hi") == "ok"
    assert _timeout_of(client_cls) == 9000


def test_mem0_embedder_gets_timeout():
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.embeddings import gemini as emb

    cfg = BaseEmbedderConfig(api_key="k", request_timeout_s=11.0)
    with patch.object(emb.genai, "Client") as client_cls:
        emb.GoogleGenAIEmbedding(cfg)
    assert _timeout_of(client_cls) == 11_000


def test_mem0_llm_gets_timeout():
    from mem0.configs.llms.base import BaseLlmConfig
    from mem0.llms import gemini as llm

    cfg = BaseLlmConfig(api_key="k", request_timeout_s=13.0)
    with patch.object(llm.genai, "Client") as client_cls:
        llm.GeminiLLM(cfg)
    assert _timeout_of(client_cls) == 13_000


def test_mem0_clients_without_a_timeout_keep_the_sdk_default():
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.configs.llms.base import BaseLlmConfig
    from mem0.embeddings import gemini as emb
    from mem0.llms import gemini as llm

    with patch.object(emb.genai, "Client") as c1:
        emb.GoogleGenAIEmbedding(BaseEmbedderConfig(api_key="k"))
    with patch.object(llm.genai, "Client") as c2:
        llm.GeminiLLM(BaseLlmConfig(api_key="k"))
    assert c1.call_args.kwargs["http_options"] is None
    assert c2.call_args.kwargs["http_options"] is None


def test_mem0_config_carries_the_timeout_to_both_gemini_blocks():
    cfg = _settings(gemini_request_timeout_s=42.0).get_mem0_config()
    assert cfg["llm"]["provider"] == "gemini" and cfg["llm"]["config"]["request_timeout_s"] == 42.0
    assert cfg["embedder"]["provider"] == "gemini"
    assert cfg["embedder"]["config"]["request_timeout_s"] == 42.0


def test_mem0_factories_pass_the_timeout_through():
    from mem0.utils.factory import EmbedderFactory, LlmFactory

    block = _settings(gemini_request_timeout_s=21.0).get_mem0_config()
    with patch("mem0.llms.gemini.genai.Client") as c1:
        LlmFactory.create("gemini", dict(block["llm"]["config"]))
    with patch("mem0.embeddings.gemini.genai.Client") as c2:
        EmbedderFactory.create("gemini", dict(block["embedder"]["config"]), None)
    assert _timeout_of(c1) == 21_000
    assert _timeout_of(c2) == 21_000


def test_every_runtime_genai_client_construction_is_covered():
    """No service/mem0 module builds a genai.Client without http_options (graphiti has its own)."""
    import re
    from pathlib import Path

    service = Path(__file__).resolve().parent.parent
    roots = [service, service.parent / "mem0" / "mem0"]
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            rel = path.relative_to(root.parent if root.name == "mem0" else root)
            parts = set(path.parts)
            if {"tests", "integration_tests", "scripts", ".venv"} & parts:
                continue
            if path.name == "gcp_auth.py":  # unused helper; builds a Vertex/AI Studio client on demand
                continue
            text = path.read_text(errors="ignore")
            for m in re.finditer(r"genai\.Client\((.*?)\)\s*$", text, re.S | re.M):
                if "http_options" not in m.group(1):
                    offenders.append(str(rel))
    assert not offenders, offenders


# ── a stalled endpoint no longer hangs the caller ────────────────────────────


class _BlackHole:
    """Accepts TCP connections and never answers."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self._held = []
        self._stop = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        self.sock.settimeout(0.2)
        while not self._stop:
            try:
                conn, _ = self.sock.accept()
                self._held.append(conn)
            except OSError:
                continue

    def close(self):
        self._stop = True
        for c in self._held:
            c.close()
        self.sock.close()


def test_stalled_endpoint_fails_fast_through_the_service_client(monkeypatch):
    import memory.core as core
    from memory.retry import _is_transient

    server = _BlackHole()
    try:
        real_client = core.genai.Client
        base_url = f"http://127.0.0.1:{server.port}"

        def _client(*args, **kwargs):  # same redirect trick the harness shim uses
            opts = kwargs["http_options"].model_copy(update={"base_url": base_url})
            return real_client(*args, **{**kwargs, "http_options": opts})

        monkeypatch.setattr(core.genai, "Client", _client)
        monkeypatch.setattr(core.settings, "gemini_request_timeout_s", 1.0)
        monkeypatch.setattr(core.settings, "google_api_key", "test-key")
        svc = object.__new__(core.CoreMixin)
        svc._genai_model = None
        svc._init_lock = threading.Lock()
        client = svc._get_genai_client()

        started = time.monotonic()
        with pytest.raises(Exception) as err:
            client.models.generate_content(model="m", contents="hello")
        elapsed = time.monotonic() - started
    finally:
        server.close()
    assert elapsed < 5, f"call took {elapsed:.1f}s with a 1 s timeout"
    # The same class of error the service's retry helper already treats as a transient provider
    # failure: it is retried with backoff, then falls back to the fallback model.
    assert _is_transient(err.value), repr(err.value)
