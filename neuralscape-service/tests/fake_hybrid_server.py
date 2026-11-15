"""A tiny local stand-in for the hybrid decision endpoint.

It speaks the wire contract HybridDecisions validates (model echo, per-question
answers, probability distributions) so the NeuralScape-side wiring can be
exercised against real stores. It says NOTHING about the real model's
accuracy: the real third-party decision API (key-gated) is not exercised by tests.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

import pytest


def _distribution(criteria: dict, choice: str) -> dict:
    """Valid distribution: .99 on the choice, remainder on UNCERTAIN/others."""
    probs = {name: 0.0 for name in criteria}
    probs[choice] = 0.99
    rest = 'UNCERTAIN' if 'UNCERTAIN' in criteria and choice != 'UNCERTAIN' else next(
        name for name in criteria if name != choice)
    probs[rest] += 0.01
    return {'type': 'choice', 'choice': choice, 'confidence': 0.99, 'probabilities': probs}


def _noul(score: float) -> dict:
    return {'type': 'noul', 'noul': score}


def default_policy(body: dict) -> dict:
    """Deterministic, keyword-driven decisions for each NeuralScape task shape."""
    questions, state = body['questions'], body.get('state') or {}
    answers = {}
    for key, q in questions.items():
        instructions = q.get('instructions')
        if key.startswith('retain-'):
            answers[key] = _noul(0.99)
        elif key.startswith('standalone-'):
            answers[key] = _noul(0.99)
        elif key.startswith('category-'):
            text = state['statements'][int(key.split('-')[1])].lower()
            choice = ('preference' if 'prefer' in text else
                      'tech_stack' if any(w in text for w in ('uses', 'postgres', 'redis')) else
                      'domain_knowledge')
            answers[key] = _distribution(q['criteria'], choice)
        elif q['type'] == 'choice' and isinstance(instructions, dict) and 'type_label' in instructions:
            label = instructions['type_label'].lower()
            answers[key] = _distribution(q['criteria'], 'preference' if 'gizmo' in label else 'UNCERTAIN')
        elif q['type'] == 'choice' and isinstance(instructions, dict) and 'new_fact' in instructions:
            answers[key] = _distribution(q['criteria'], 'DISTINCT')
        elif q['type'] == 'choice':
            answers[key] = _distribution(q['criteria'], 'NONE' if 'NONE' in q['criteria'] else next(iter(q['criteria'])))
        elif q['type'] == 'noul':
            candidate = (instructions or {}).get('candidate') or {}
            text = json.dumps(candidate).lower()
            answers[key] = _noul(0.97 if 'rerank-winner' in text else 0.05)
    return answers


class FakeHybrid:
    """Context-managed local server; records every request body it receives."""

    def __init__(self, policy: Callable[[dict], dict] = default_policy, *, status: int = 200,
                 model: str | None = None):
        self.policy, self.status, self.model = policy, status, model
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                length = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(length))
                outer.requests.append(body)
                if outer.status != 200:
                    self.send_response(outer.status)
                    self.end_headers()
                    return
                payload = json.dumps({'model': outer.model or body['model'],
                                      'answers': outer.policy(body),
                                      'usage': {'input_tokens': 10, 'output_tokens': 2}}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):  # silence stderr noise
                pass

        self._server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self._server.server_address[1]}/decide'

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def fake_hybrid(monkeypatch):
    """Point the hybrid decision adapter at a local fake and give it a (fake) key."""
    from config import settings

    with FakeHybrid() as server:
        monkeypatch.setattr(settings, 'hybrid_inference_endpoint', server.url)
        monkeypatch.setattr(settings, 'hybrid_inference_api_key', 'fake-local-key')
        monkeypatch.setattr(settings, 'hybrid_inference_model', 'test-model')
        yield server
