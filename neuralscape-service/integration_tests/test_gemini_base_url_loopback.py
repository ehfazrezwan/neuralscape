"""``NS_GEMINI_BASE_URL`` end to end: the service's own Gemini clients reach the configured endpoint.

A loopback HTTP stand-in (127.0.0.1, an ephemeral port) plays the Gemini API.  With the setting on, the real
``MemoryService`` is built over real Qdrant and Neo4j (a collection of its own), and three kinds of client are
exercised: the service's own generation client, the embedder mem0 builds, and the embedder Graphiti builds.
Every request must arrive at the stand-in; the default endpoint is never contacted because nothing here holds a
real key.  The stand-in's server socket and every connection are closed in teardown (``HTTP/1.0``: one request
per connection), the SDK client is closed, and the process-wide override is undone.
"""

from __future__ import annotations

import gc
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import provider_routing

EMBED_DIMS = 768


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # the connection closes after each response: no socket outlives the test

    def log_message(self, *args):  # noqa: D401 - silence the default stderr line per request
        return

    def _reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802 - the http.server hook name
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
        self.server.requests.append(self.path)
        path = self.path.split("?")[0]
        if path.endswith(":generateContent"):
            self._reply(200, {
                "candidates": [{"content": {"role": "model", "parts": [{"text": "stand-in answer"}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
            })
        elif path.endswith(":embedContent"):
            dims = int(body.get("outputDimensionality") or EMBED_DIMS)
            self._reply(200, {"embedding": {"values": [0.01] * dims}})
        elif path.endswith(":batchEmbedContents"):
            dims = EMBED_DIMS
            requests = body.get("requests") or [{}]
            self._reply(200, {"embeddings": [
                {"values": [0.01] * int(r.get("outputDimensionality") or dims)} for r in requests
            ]})
        else:
            self._reply(404, {"error": {"code": 404, "message": "not served by the stand-in", "status": "NOT_FOUND"}})

    do_GET = do_POST  # noqa: N815


@pytest.fixture()
def stand_in():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)
        assert not thread.is_alive(), "the loopback stand-in did not stop"


def _close_clients(service) -> None:
    """Close every Gemini client the service built, on the loop that owns it, then collect garbage.

    The vendored clients own async HTTP pools bound to the service's bridge loop.  Left to the garbage
    collector, their finalisers could fire later inside a loop that belongs to another test and schedule
    a close of connections that belong to a loop that is gone.  Closing them here, while their loop still
    runs, and collecting with no loop running, leaves nothing that refers to the stand-in.
    """

    graphiti = getattr(service, "_graphiti", None)
    embedder = getattr(graphiti, "embedder", None)
    aio = getattr(getattr(embedder, "client", None), "aio", None)
    if aio is not None:
        try:
            service._run_on_bridge(aio.aclose(), timeout=10.0)
        except Exception:  # noqa: BLE001 - teardown must go on
            pass
    memory = getattr(service, "_memory", None)
    for client in (
        getattr(getattr(memory, "embedding_model", None), "client", None),
        getattr(embedder, "client", None),
        service._genai_model,
    ):
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
    service.close()
    gc.collect()


@pytest.fixture()
def routed_service(flags, stores, unique_ns, stand_in):
    """The real service, built after the override is set, over a Qdrant collection of its own.

    The override patches the SDK client class for the whole process, so it is undone in a ``finally``
    that covers everything after it can be applied (a service that fails to build included), and the
    settings and environment go back through ``flags`` (monkeypatch).
    """
    from qdrant_client import QdrantClient

    from memory_service import MemoryService

    provider_routing.reset()
    service = None
    try:
        url = f"http://127.0.0.1:{stand_in.server_address[1]}"
        flags(ns_gemini_base_url=url, google_api_key="nsit-not-a-real-key", qdrant_collection=unique_ns.qdrant_collection)
        unique_ns.on_cleanup(lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection))
        service = MemoryService()
        yield service
    finally:
        try:
            if service is not None:
                _close_clients(service)
        finally:
            provider_routing.reset()
            assert not provider_routing._APPLIED


def test_every_gemini_client_the_service_builds_reaches_the_configured_endpoint(routed_service, stand_in):
    service = routed_service

    # 1. the service's own generation client (extraction, Ask's model call)
    reply = service._get_genai_client().models.generate_content(model="nsit-model", contents="hello")
    assert reply.text == "stand-in answer"
    assert any(path.endswith(":generateContent") for path in stand_in.requests), stand_in.requests

    # 2. the embedder mem0 builds (built with real Qdrant and Neo4j behind it)
    memory = service._get_memory()
    stand_in.requests.clear()
    vector = memory.embedding_model.embed("hello", memory_action="search")
    assert len(vector) == EMBED_DIMS
    assert any("mbedContent" in path for path in stand_in.requests), stand_in.requests

    # 3. the embedder Graphiti builds, driven the way the graph leg drives it (on the service's bridge)
    stand_in.requests.clear()
    created = service._run_on_bridge(service._graphiti.embedder.create("hello graph"), timeout=30.0)
    assert len(created) > 0
    assert stand_in.requests, "Graphiti's embedder did not reach the configured endpoint"


def test_the_override_is_the_configured_address_and_nothing_else_is_contacted(routed_service, stand_in):
    routed_service._get_memory()
    port = stand_in.server_address[1]
    assert provider_routing._APPLIED == [f"http://127.0.0.1:{port}"]
