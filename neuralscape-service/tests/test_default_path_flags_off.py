"""With every flag off the default path is the one that existed before: nothing flag-only is even imported.

The probe runs in a fresh interpreter (so earlier tests cannot have imported anything): it imports the service, the
worker and the API module, then drives a default search, a graph enrichment and the worker's graph job, and reports
which flag-only modules were loaded.  The in-process tests pin the other default-path hunks.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SERVICE_ROOT = Path(__file__).resolve().parent.parent

#: Modules only a flag brings in; none may be loaded on the default path.
FLAG_ONLY = (
    "engine_adapters_current",
    "engine_adapters_switch",
    "engine_adapters_pgvector",
    "engine_routing",
    "usage_store",
    "usage_wrappers",
    "usage_scenarios",
    "usage_scenario_pilot",
    "contracts_usage_reconcile",
    "psycopg",
)

PROBE = textwrap.dedent(
    """
    import asyncio, json, sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import qdrant_client  # the pool imports it lazily; not part of this comparison
    import memory_service, worker, main
    from memory_service import MemoryService

    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = None
    svc._bridge = None
    svc._memory.embedding_model.embed.return_value = [0.1] * 8
    result = MagicMock(); result.points = []
    svc._memory.vector_store.client.query_points.return_value = result
    svc.search(query="q", user_id="u", limit=3)          # a default search, graph leg submitted
    svc._graphiti = MagicMock(); svc._bridge = MagicMock()
    svc._memory.graph.add.return_value = {"episode_uuid": "e1", "episode_name": "n1"}
    svc._memory.get.return_value = {"id": "m1", "memory": "c"}
    svc._attach_memory_id_to_graph_nodes = lambda **kw: None
    svc._persist_graph_episode_ref = lambda *a, **kw: None
    svc._maybe_supersede_contradicted = lambda **kw: None
    svc.enrich_graph(content="c", user_id="u", project_id=None, visibility="private", memory_id="m1")
    row = SimpleNamespace(id="m1", memory="c", visibility="private", project_id=None, workspace=None, occurred_at=None)
    svc.get_memory = lambda memory_id: row
    asyncio.run(worker.process_graph_enrichment({"service": svc}, "m1", "c", "u", None, "private"))
    asyncio.run(worker._bind_usage_attempt({"job_id": "j", "redis": None}))
    import provider_routing
    from google import genai
    from config import settings
    before = genai.Client.__init__
    provider_routing.apply_from_settings(settings)
    unchanged = genai.Client.__init__ is before
    print("PROBE " + json.dumps({"loaded": sorted(n for n in sys.modules if n in %r), "client_unchanged": unchanged}))
    """
) % (FLAG_ONLY,)


def test_a_default_search_enrichment_and_graph_job_import_nothing_flag_only():
    env = {**os.environ, "NEURALSCAPE_AUTH_DISABLED": "true", "PYTHONDONTWRITEBYTECODE": "1"}
    for name in ("USAGE_STORE_ENABLED", "ENGINE_ROUTING_ENABLED", "ENGINE_ROUTING_RECALL_ENABLED",
                 "NS_GEMINI_BASE_URL", "GRAPH_DELETE_RACE_CLEANUP_ENABLED", "GRAPH_SUPPORT_AWARE_REMOVAL_ENABLED"):
        env.pop(name, None)
    done = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=SERVICE_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    line = next(l for l in done.stdout.splitlines() if l.startswith("PROBE "))
    report = json.loads(line[len("PROBE "):])
    assert report["loaded"] == [], f"flag-only modules loaded on the default path: {report['loaded']}"
    assert report["client_unchanged"] is True  # no base URL: the SDK client class is left alone


def test_the_graph_job_calls_enrich_graph_exactly_as_before_with_the_cleanup_flag_off():
    """No extra keyword reaches ``enrich_graph``: the call is the one the worker always made."""
    import worker
    from config import settings

    assert settings.graph_delete_race_cleanup_enabled is False
    seen = []
    row = SimpleNamespace(id="m1", memory="c", visibility="private", project_id=None, workspace=None, occurred_at=None)

    class Service:
        def get_memory(self, memory_id):
            return row

        def enrich_graph(self, **kwargs):
            seen.append(kwargs)
            return True

    asyncio.run(worker.process_graph_enrichment({"service": Service()}, "m1", "c", "u", None, "private"))
    assert sorted(seen[0]) == sorted([
        "content", "user_id", "project_id", "visibility", "memory_id", "source_ref", "graph_ontology",
        "occurred_at", "workspace", "result_out",
    ])  # flag off: exactly dev's kwargs (result_out only); no "outcome"


def test_the_usage_hooks_are_no_ops_without_a_recorder():
    import usage_capture as uc
    from memory.retry import retry_transient

    assert uc.get_usage_recorder() is None
    calls = []

    def generate_content(*args, **kwargs):
        calls.append((args, kwargs))
        return "response"

    assert retry_transient(generate_content, "a", model="m", operation="x") == "response"
    assert calls == [(("a",), {"model": "m"})]  # one call, the same arguments
    with uc.ensure_attempt("alice", "search"):
        assert uc.current_attempt_context() is None
    uc.record_generation_usage("response", operation="x", model="m")  # returns, records nothing


def test_the_job_start_hook_reads_nothing_from_redis_without_a_recorder():
    import usage_capture as uc
    import worker

    assert uc.get_usage_recorder() is None
    redis = MagicMock()
    asyncio.run(worker._bind_usage_attempt({"job_id": "j", "redis": redis}))
    redis.get.assert_not_called()
    assert uc.current_attempt_context() is None


def test_a_default_search_is_not_routed_and_asks_the_engine_nothing(monkeypatch):
    import qdrant_client  # noqa: F401
    from config import settings
    from memory_service import MemoryService

    assert settings.engine_routing_enabled is False and settings.engine_routing_recall_enabled is False
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._memory.embedding_model.embed.return_value = [0.1] * 8
    result = MagicMock()
    result.points = []
    svc._memory.vector_store.client.query_points.return_value = result
    monkeypatch.setattr(svc, "_engine_recall", lambda *a, **k: pytest.fail("the engine was asked"))
    assert svc.search(query="q", user_id="u", limit=3) == []
