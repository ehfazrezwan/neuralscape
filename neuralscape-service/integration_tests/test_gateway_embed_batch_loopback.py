"""End-to-end proof of the gateway embed-batching patch against a loopback stub gateway.

There is no real gateway on the gate box, so a local OpenAI-compatible stub
(``POST /v1/embeddings``, bound to 127.0.0.1 on an ephemeral port) plays the
part of the single-input-only embedding endpoint, records every request and
returns deterministic vectors of the configured dimension. With the generic
``LLM_GATEWAY_*`` settings pointed at it, the real ``MemoryService`` stores a raw
batch through the real path into real Qdrant (throwaway collection). Raw writes
need no extraction LLM. Proved:

* default (``EMBEDDER_MAX_BATCH_SIZE`` unset): every input is its own request,
  and every stored point holds the stub's vector for its text;
* an explicit batch size of 3 makes 7 inputs arrive as calls of 3, 3 and 1;
* a gateway that rejects batched input with a single-input error triggers the
  per-item fallback (one rejected batched call, then one call per input);
* a 4xx on one input fails the write with an error naming that input's index; a
  5xx propagates as a clear error naming the status;
* with the gateway flag OFF the stub receives nothing.

The gateway base URL is a loopback placeholder; no real gateway is named.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

DIMS = 768


def stub_vector(text: str) -> list[float]:
    base = (sum(text.encode()) % 89 + 1) / 100.0
    return [round(base + i * 1e-4, 6) for i in range(DIMS)]


class StubGateway:
    """OpenAI-compatible /v1/embeddings stub that records each request."""

    def __init__(self, *, single_input_only: bool = True):
        self.single_input_only = single_input_only
        self.requests: list[list[str]] = []  # the input list of every request, in arrival order
        self.reject_4xx: set[str] = set()
        self.fail_5xx: set[str] = set()
        self._lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
                with stub._lock:
                    stub.requests.append(list(inputs))
                code, payload = 200, None
                if stub.single_input_only and len(inputs) > 1:
                    code, payload = 400, {"error": {"message": "google-vertex/gemini-embedding-001 accepts only one input per request"}}
                elif any(t in stub.fail_5xx for t in inputs):
                    code, payload = 503, {"error": {"message": "upstream unavailable"}}
                elif any(t in stub.reject_4xx for t in inputs):
                    code, payload = 400, {"error": {"message": "invalid input"}}
                if payload is None:
                    payload = {
                        "object": "list", "model": body.get("model", "stub"),
                        "data": [{"object": "embedding", "index": i, "embedding": stub_vector(t)} for i, t in enumerate(inputs)],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    }
                raw = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    s = StubGateway()
    yield s
    s.close()


@pytest.fixture
def make_svc(flags, stores, llm_key, unique_ns, stub, monkeypatch):
    from qdrant_client import QdrantClient

    # get_mem0_config exports OPENAI_BASE_URL / OPENAI_API_KEY; restore them afterwards.
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )

    built: list = []

    def build(*, gateway: bool = True, batch_size: int = 0):
        flags(
            qdrant_collection=unique_ns.qdrant_collection,
            qdrant_url=stores.qdrant_url,
            llm_gateway_enabled=gateway,
            llm_gateway_base_url=stub.base_url,
            llm_gateway_api_key="stub-key",
            embedder_max_batch_size=batch_size,
        )
        from memory_service import MemoryService

        svc = MemoryService()
        svc._get_memory()
        # Raw writes only: no graph episode, no extraction LLM. Detach the graph
        # handles for the test body but remember them, so teardown can close the
        # Neo4j driver and stop the bridge loop thread (dropping them leaks a
        # thread, an event loop and driver sockets per service).
        built.append((svc, svc._graphiti, svc._bridge))
        svc._graphiti = None
        svc._bridge = None
        return svc

    yield build

    for svc, graphiti, bridge in built:
        svc._graphiti, svc._bridge = graphiti, bridge
        for close in (
            svc.close,  # graphiti driver + bridge loop
            svc._memory.vector_store.client.close,  # Qdrant HTTP client
            svc._memory.embedding_model.client.close,  # gateway/OpenAI HTTP client pool
        ):
            try:
                close()
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass


def _items(uid, texts):
    return [{"content": t, "user_id": uid, "category": "domain_knowledge", "scope": "global", "visibility": "private"}
            for t in texts]


def _texts(tag, n):
    return [f"gateway stub fact {tag} number {i}" for i in range(n)]


def _assert_each_point_holds_its_own_texts_vector(svc, uid, texts):
    """Every stored point's vector is the stub's vector for ITS text (order within a batch cannot be swapped)."""
    stored = _stored_vectors(svc, uid)
    assert set(stored) == set(texts)
    assert len({tuple(stub_vector(t)[:3]) for t in texts}) == len(texts), "test texts must embed distinctly"
    for text in texts:
        assert stored[text] == pytest.approx(stub_vector(text), abs=1e-5), text


def _stored_vectors(svc, uid):
    rows = svc._scroll_all_user_memories(uid)
    vectors = svc._fetch_dense_vectors([r["id"] for r in rows])
    return {r["payload"]["data"]: vectors[r["id"]] for r in rows}


def test_default_gateway_sends_one_input_per_request_and_stores_the_stub_vectors(make_svc, stub, unique_ns):
    svc = make_svc()
    texts = _texts(uuid.uuid4().hex[:6], 5)

    svc.store_raw_batch(_items(unique_ns.user_id, texts))

    assert sorted(stub.requests) == sorted([t] for t in texts), stub.requests  # one text per request
    _assert_each_point_holds_its_own_texts_vector(svc, unique_ns.user_id, texts)


def test_an_explicit_batch_size_chunks_the_inputs(make_svc, stub, unique_ns):
    stub.single_input_only = False
    svc = make_svc(batch_size=3)
    texts = _texts(uuid.uuid4().hex[:6], 7)

    svc.store_raw_batch(_items(unique_ns.user_id, texts))

    # exact chunk contents, in order: chunks are sent one after another in input order
    assert stub.requests == [texts[0:3], texts[3:6], texts[6:7]], stub.requests
    _assert_each_point_holds_its_own_texts_vector(svc, unique_ns.user_id, texts)


def test_a_batched_call_rejected_for_single_input_falls_back_to_per_item(make_svc, stub, unique_ns):
    stub.single_input_only = True  # rejects any multi-input request
    svc = make_svc(batch_size=3)
    texts = _texts(uuid.uuid4().hex[:6], 3)

    svc.store_raw_batch(_items(unique_ns.user_id, texts))

    # one rejected batched call carrying all three texts in order, then one call per text
    assert stub.requests[0] == texts, stub.requests
    assert sorted(stub.requests[1:]) == sorted([t] for t in texts), stub.requests
    _assert_each_point_holds_its_own_texts_vector(svc, unique_ns.user_id, texts)


def test_a_4xx_on_one_input_fails_the_write_naming_its_index(make_svc, stub, unique_ns):
    svc = make_svc()
    texts = _texts(uuid.uuid4().hex[:6], 4)
    stub.reject_4xx.add(texts[2])

    with pytest.raises(Exception, match=r"input index 2") as err:
        svc.store_raw_batch(_items(unique_ns.user_id, texts))

    assert "invalid input" in str(err.value) or "400" in str(err.value)
    assert svc._scroll_all_user_memories(unique_ns.user_id) == []  # nothing half-written


def test_a_5xx_propagates_as_a_clear_error(make_svc, stub, unique_ns):
    svc = make_svc()
    texts = _texts(uuid.uuid4().hex[:6], 2)
    stub.fail_5xx.add(texts[1])

    with pytest.raises(Exception) as err:
        svc.store_raw_batch(_items(unique_ns.user_id, texts))

    assert "index 1" in str(err.value) and "503" in str(err.value), str(err.value)
    assert svc._scroll_all_user_memories(unique_ns.user_id) == []


def test_flag_off_the_stub_receives_nothing(make_svc, stub, unique_ns):
    svc = make_svc(gateway=False)  # the default AI Studio embedder (real key from llm_key)
    svc.store_raw_batch(_items(unique_ns.user_id, _texts(uuid.uuid4().hex[:6], 2)))
    assert stub.requests == []
