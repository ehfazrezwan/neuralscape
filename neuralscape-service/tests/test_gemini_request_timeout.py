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


@pytest.mark.parametrize("bad", ["inf", "-inf", "nan", "Infinity"])
def test_rejects_non_finite_values_at_settings_load(bad, monkeypatch):
    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT_S", bad)
    with pytest.raises(ValidationError):
        _settings()
    monkeypatch.delenv("GEMINI_REQUEST_TIMEOUT_S")
    with pytest.raises(ValidationError):
        _settings(gemini_request_timeout_s=float(bad))


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


def _unbounded_genai_clients(source: str) -> list[int]:
    """Line numbers of google-genai client constructions without an ``http_options`` keyword.

    A construction is a call whose callee resolves, through the module's own imports, to
    ``google.genai.Client`` (``genai.Client``, an aliased module, ``google.genai.Client``, a
    directly imported ``Client`` under any name, or ``google.genai.client.Client``).
    """
    import ast

    tree = ast.parse(source)
    module_names: set[str] = set()  # dotted names that refer to google.genai (or .client)
    class_names: set[str] = set()  # bare names bound to the Client class
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in ("google.genai", "google.genai.client"):
                    module_names.add(a.asname or a.name)
                elif a.name == "google" and not a.asname:
                    module_names.add("google.genai")
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                bound = a.asname or a.name
                if node.module == "google" and a.name == "genai":
                    module_names.add(bound)
                elif node.module == "google.genai" and a.name == "client":
                    module_names.add(bound)
                elif node.module in ("google.genai", "google.genai.client") and a.name == "Client":
                    class_names.add(bound)

    def dotted(n):
        parts = []
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
            return ".".join(reversed(parts))
        return None

    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func)
        if name is None:
            continue
        is_client = name in class_names or (
            name.endswith(".Client") and name.rsplit(".", 1)[0] in module_names
        ) or name == "google.genai.Client" and "google.genai" in module_names
        if is_client and not any(k.arg == "http_options" for k in node.keywords):
            bad.append(node.lineno)
    return bad


@pytest.mark.parametrize(
    "source",
    [
        "from google import genai\nc = genai.Client(api_key=k)",
        "from google import genai\nc = genai.Client(api_key=k)  # http_options=...",
        "from google import genai\nm = genai.Client(api_key=k).models",
        "from google import genai as g\nc = g.Client(api_key=k)",
        "from google.genai import Client\nc = Client(api_key=k)",
        "from google.genai import Client as C\nc = C(api_key=k)",
        "import google.genai\nc = google.genai.Client(api_key=k)",
        "import google.genai as gg\nc = gg.Client(api_key=k)",
        "from google import genai\nc = genai.Client(\n    api_key=k,\n)\n# http_options only in a comment",
        "from google import genai\nc = genai.Client(api_key=k, **opts)",
    ],
)
def test_client_scan_catches_unbounded_constructions(source):
    assert _unbounded_genai_clients(source) == [next(
        i for i, ln in enumerate(source.splitlines(), 1) if "Client(" in ln or "C(" in ln
    )]


@pytest.mark.parametrize(
    "source",
    [
        "from google import genai\nc = genai.Client(api_key=k, http_options=o)",
        "from google import genai as g\nc = g.Client(\n    api_key=k,\n    http_options=o,\n).models",
        "from google.genai import Client\nc = Client(api_key=k, http_options=o)  # bounded",
        "import google.genai\nc = google.genai.Client(http_options=o)",
        "class Client: ...\nc = Client(api_key=k)",  # an unrelated local class
        "import other\nc = other.Client(api_key=k)",
    ],
)
def test_client_scan_accepts_bounded_or_unrelated_constructions(source):
    assert _unbounded_genai_clients(source) == []


def test_every_runtime_genai_client_construction_is_covered():
    """No service/mem0 module builds a genai.Client without http_options (graphiti has its own)."""
    from pathlib import Path

    service = Path(__file__).resolve().parent.parent
    roots = [service, service.parent / "mem0" / "mem0"]
    offenders = []
    scanned = 0
    for root in roots:
        for path in root.rglob("*.py"):
            parts = set(path.parts)
            if {"tests", "integration_tests", "scripts", ".venv"} & parts:
                continue
            if path.name == "gcp_auth.py":  # unused helper; builds a Vertex/AI Studio client on demand
                continue
            scanned += 1
            try:
                lines = _unbounded_genai_clients(path.read_text(errors="ignore"))
            except SyntaxError:
                continue
            offenders += [f"{path.relative_to(root.parent)}:{n}" for n in lines]
    assert scanned > 100  # the scan really walked the runtime packages
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


# ── a raw write retries a transient embed error before anything is stored ─────


def _raw_service(embed_side_effect):
    import httpx

    from memory_service import MemoryService

    svc = MemoryService()
    mem = MagicMock()
    mem.embedding_model.embed.side_effect = embed_side_effect
    prepared = {"content": "x", "response": MagicMock()}
    finalize = MagicMock()
    return svc, mem, prepared, finalize, httpx


def test_raw_write_retries_a_timed_out_embed_and_stores_one_row():
    import httpx

    calls = []

    def embed(text, memory_action=None):
        calls.append(text)
        if len(calls) == 1:
            raise httpx.ReadTimeout("The read operation timed out")
        return [0.1, 0.2, 0.3]

    svc, mem, prepared, finalize, _ = _raw_service(embed)
    with patch.object(svc, "_prepare_raw_store", return_value=(prepared, None)), \
         patch.object(svc, "_get_memory", return_value=mem), \
         patch.object(svc, "_finalize_raw_store", finalize), \
         patch("memory.retry.time.sleep"):
        svc.store_raw(content="x", user_id="u", category="personal_fact")
    assert len(calls) == 2
    finalize.assert_called_once()
    assert finalize.call_args.args[1] == [0.1, 0.2, 0.3]


def test_raw_write_fails_with_the_same_error_class_when_every_attempt_times_out():
    import httpx

    def embed(text, memory_action=None):
        raise httpx.ReadTimeout("The read operation timed out")

    svc, mem, prepared, finalize, _ = _raw_service(embed)
    with patch.object(svc, "_prepare_raw_store", return_value=(prepared, None)), \
         patch.object(svc, "_get_memory", return_value=mem), \
         patch.object(svc, "_finalize_raw_store", finalize), \
         patch("memory.retry.time.sleep"):
        with pytest.raises(httpx.ReadTimeout):
            svc.store_raw(content="x", user_id="u", category="personal_fact")
    finalize.assert_not_called()  # nothing was stored
    assert mem.embedding_model.embed.call_count > 1


def test_raw_write_does_not_retry_a_non_transient_embed_error():
    def embed(text, memory_action=None):
        raise ValueError("bad input")

    svc, mem, prepared, finalize, _ = _raw_service(embed)
    with patch.object(svc, "_prepare_raw_store", return_value=(prepared, None)), \
         patch.object(svc, "_get_memory", return_value=mem), \
         patch.object(svc, "_finalize_raw_store", finalize), \
         patch("memory.retry.time.sleep"):
        with pytest.raises(ValueError):
            svc.store_raw(content="x", user_id="u", category="personal_fact")
    assert mem.embedding_model.embed.call_count == 1
