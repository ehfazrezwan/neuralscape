"""Deterministic fake model provider for mechanics-only service benchmarks.

Serves the Gemini REST surface (``generateContent``, ``embedContent``,
``batchEmbedContents``) and the OpenAI-compatible surface
(``/v1/chat/completions``, ``/v1/embeddings``) with a fixed, configurable
latency per request.  Outputs are pure functions of the request, so repeated
runs see identical provider behaviour.  Standard library only.

This stands in for provider time to isolate service mechanics; it says
nothing about provider-bound capacity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DEFAULT_DIMENSIONS = 768
_MODEL_METHOD = re.compile(r"models/(?P<model>[^/:]+):(?P<method>\w+)$")


def deterministic_embedding(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    """Unit-length vector derived only from the text."""

    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        values.extend((byte - 127.5) / 127.5 for byte in digest)
        counter += 1
    values = values[:dimensions]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [round(v / norm, 6) for v in values]


def extraction_text(prompt: str) -> str:
    """Two deterministic facts whose content depends on the prompt."""

    digest = hashlib.sha256(prompt.encode()).hexdigest()
    facts = [
        f"[preference] The user prefers option {digest[:12]} for workflow {digest[12:16]}",
        f"[tech_stack] The project uses component {digest[16:28]} version {int(digest[28:30], 16)}",
    ]
    return json.dumps({"facts": facts})


def minimal_instance(schema: Any, root: Any = None, depth: int = 0) -> Any:
    """Smallest value satisfying a (Gemini or JSON) schema; arrays stay empty."""

    root = schema if root is None else root
    if not isinstance(schema, dict) or depth > 12:
        return None
    if "$ref" in schema:
        target: Any = root
        for part in str(schema["$ref"]).lstrip("#/").split("/"):
            target = target.get(part, {}) if isinstance(target, dict) else {}
        return minimal_instance(target, root, depth + 1)
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and options:
            chosen = [o for o in options if o.get("type") not in ("null", "NULL")] or options
            return minimal_instance(chosen[0], root, depth + 1)
    if schema.get("enum"):
        return schema["enum"][0]
    kind = str(schema.get("type", "")).lower()
    if kind == "object" or "properties" in schema:
        return {
            name: minimal_instance(sub, root, depth + 1)
            for name, sub in (schema.get("properties") or {}).items()
        }
    if kind == "array":
        return []
    if kind == "integer":
        return 0
    if kind == "number":
        return 0.0
    if kind == "boolean":
        return False
    if kind == "string":
        return ""
    return None


def _prompt_text(body: dict[str, Any]) -> str:
    parts: list[str] = []
    contents = body.get("contents")
    if isinstance(contents, str):
        parts.append(contents)
    elif isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for part in item.get("parts", []) or []:
                    if isinstance(part, dict) and "text" in part:
                        parts.append(str(part["text"]))
    return "\n".join(parts)


def _gemini_schema(body: dict[str, Any]) -> Any:
    config = body.get("generationConfig") or {}
    return config.get("responseJsonSchema") or config.get("responseSchema")


class Counters:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.values: dict[str, int] = {}

    def bump(self, key: str) -> None:
        with self._lock:
            self.values[key] = self.values.get(key, 0) + 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.values)


class FakeProviderHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    generate_latency_s = 0.0
    embed_latency_s = 0.0
    counters = Counters()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send(self, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/_stats"):
            self._send(200, self.counters.snapshot())
        elif self.path.startswith("/_health"):
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return
        path = self.path.split("?", 1)[0].rstrip("/")
        match = _MODEL_METHOD.search(path)
        if match:
            self._gemini(match["model"], match["method"], body)
        elif path.endswith("/chat/completions"):
            self._openai_chat(body)
        elif path.endswith("/embeddings"):
            self._openai_embeddings(body)
        else:
            self.counters.bump("unknown")
            self._send(404, {"error": "unknown path"})

    def _gemini(self, model: str, method: str, body: dict[str, Any]) -> None:
        if method == "generateContent":
            self.counters.bump("gemini.generate")
            time.sleep(self.generate_latency_s)
            schema = _gemini_schema(body)
            prompt = _prompt_text(body)
            if schema is not None:
                text = json.dumps(minimal_instance(schema))
            else:
                text = extraction_text(prompt)
            self._send(
                200,
                {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": text}]},
                            "finishReason": "STOP",
                            "index": 0,
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": max(1, len(prompt) // 4),
                        "candidatesTokenCount": max(1, len(text) // 4),
                        "totalTokenCount": max(1, len(prompt) // 4) + max(1, len(text) // 4),
                    },
                    "modelVersion": model,
                },
            )
        elif method in ("embedContent", "batchEmbedContents"):
            self.counters.bump(f"gemini.{method}")
            time.sleep(self.embed_latency_s)
            requests = body.get("requests") if method == "batchEmbedContents" else [body]
            vectors = []
            for item in requests or []:
                text = "".join(
                    str(p.get("text", ""))
                    for p in (item.get("content") or {}).get("parts", [])
                    if isinstance(p, dict)
                )
                dims = int(item.get("outputDimensionality") or DEFAULT_DIMENSIONS)
                vectors.append({"values": deterministic_embedding(text, dims)})
            if method == "embedContent":
                self._send(200, {"embedding": vectors[0]})
            else:
                self._send(200, {"embeddings": vectors})
        else:
            self.counters.bump("unknown")
            self._send(404, {"error": f"unsupported method {method}"})

    def _openai_chat(self, body: dict[str, Any]) -> None:
        self.counters.bump("openai.chat")
        time.sleep(self.generate_latency_s)
        messages = body.get("messages") or []
        prompt = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        schema_wrapper = (body.get("response_format") or {}).get("json_schema")
        if schema_wrapper:
            text = json.dumps(minimal_instance(schema_wrapper.get("schema", schema_wrapper)))
        else:
            text = extraction_text(prompt)
        self._send(
            200,
            {
                "id": "chatcmpl-" + hashlib.sha256(prompt.encode()).hexdigest()[:16],
                "object": "chat.completion",
                "model": body.get("model", "fake"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": max(1, len(prompt) // 4),
                    "completion_tokens": max(1, len(text) // 4),
                    "total_tokens": max(1, len(prompt) // 4) + max(1, len(text) // 4),
                },
            },
        )

    def _openai_embeddings(self, body: dict[str, Any]) -> None:
        self.counters.bump("openai.embeddings")
        time.sleep(self.embed_latency_s)
        raw = body.get("input", "")
        inputs = raw if isinstance(raw, list) else [raw]
        dims = int(body.get("dimensions") or DEFAULT_DIMENSIONS)
        self._send(
            200,
            {
                "object": "list",
                "model": body.get("model", "fake"),
                "data": [
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": deterministic_embedding(str(text), dims),
                    }
                    for index, text in enumerate(inputs)
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )


def make_server(
    host: str, port: int, generate_latency_ms: float, embed_latency_ms: float
) -> ThreadingHTTPServer:
    handler = type(
        "ConfiguredHandler",
        (FakeProviderHandler,),
        {
            "generate_latency_s": generate_latency_ms / 1000.0,
            "embed_latency_s": embed_latency_ms / 1000.0,
            "counters": Counters(),
        },
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--generate-latency-ms", type=float, default=50.0)
    parser.add_argument("--embed-latency-ms", type=float, default=20.0)
    args = parser.parse_args()
    make_server(
        args.host, args.port, args.generate_latency_ms, args.embed_latency_ms
    ).serve_forever()


if __name__ == "__main__":
    main()
