"""Flag-on proof for the code-intel engines (plat-05), against real stores.

Code indexing is operator-side by design: the repo must be on the server disk and
mapped in ``CODE_REPOS``. These tests map a tiny fixture repo that lives next to
this file and drive the real path:

    process_code_index (the ingest worker function) -> registry/router ->
    bound CodeKnowledgeSystem -> query / neighbors / path / locate / impact

* native (``CODE_NATIVE_ENABLED=true``): real Neo4j (and Qdrant/Redis through the
  shared service). The dense leg is switched off (``code_embedder=off``) so no
  embedding or LLM call is spent; locate runs on BM25 symbol cards.
* graphify-lib: in-process, on by default with the code-graph extra.
* CBM (``CBM_ENABLED=true``): needs the cbm-bridge container and binary, which the
  standard store stack does not run. It runs when ``CBM_BRIDGE_URL`` is reachable
  and skips with that reason otherwise.

Needs the fixtures from the integration-test convention (``stores``, ``llm_key``,
``flags``, ``unique_ns``).
"""

from __future__ import annotations

import asyncio
import importlib
import os
import urllib.request
from pathlib import Path

import pytest

FIXTURE_REPO = Path(__file__).resolve().parent / "fixtures" / "code_demo_repo"


def _recall(system, **kw):
    from knowledge.base import RecallRequest

    return system.recall(RecallRequest(**kw))


def _bind(system_name, code_space, user_id, operation):
    """Resolve through the router, then bind the per-code_space engine."""
    import config
    from knowledge.code_dispatch import resolve_bound_code_system
    from knowledge.router import resolve_systems

    decision = resolve_systems(
        query="code question",
        knowledge_system=system_name,
        operation=operation,
        is_code_tool=True,
    )
    assert decision.layer == 1
    assert decision.systems and decision.systems[0].info.name == system_name, decision.rationale
    bound = resolve_bound_code_system(system_name, code_space, user_id, config.settings)
    assert bound is not None, f"{system_name} did not bind for {code_space}"
    return bound


@pytest.fixture
def repo_dir(tmp_path) -> Path:
    """A throwaway copy of the fixture repo (engines write caches next to the source)."""
    import shutil

    dest = tmp_path / "demo"
    shutil.copytree(FIXTURE_REPO, dest)
    return dest


@pytest.fixture
def code_repos(flags, monkeypatch, repo_dir):
    """CODE_REPOS -> the fixture repo copy; dense/LLM legs off (deterministic)."""
    import json

    import config

    monkeypatch.setattr(config.settings, "code_repos", {"demo": str(repo_dir)})
    monkeypatch.setenv("CODE_REPOS", json.dumps({"demo": str(repo_dir)}))
    flags(code_embedder="off", code_index_embeddings=False)
    from adapters.code_graph import query as cg_query

    yield config.settings
    for key in [k for k in cg_query._ctx_cache if "--demo" in str(k)]:
        cg_query._ctx_cache.pop(key, None)


@pytest.fixture
def registry_snapshot():
    """Restore the knowledge registry exactly as it was, even if the test fails.

    Registration runs at import time, so the flag-on tests reload ``knowledge``;
    that replaces entries (a fresh ns-memory base, new code systems). The
    finalizer puts back the original objects so nothing leaks into later tests.
    """
    from knowledge.registry import KNOWLEDGE_REGISTRY

    saved = dict(KNOWLEDGE_REGISTRY)
    try:
        yield saved
    finally:
        KNOWLEDGE_REGISTRY.clear()
        KNOWLEDGE_REGISTRY.update(saved)


@pytest.fixture
def native_on(monkeypatch, code_repos, registry_snapshot):
    """Re-run knowledge registration with CODE_NATIVE_ENABLED=true."""
    import knowledge
    from knowledge.registry import KNOWLEDGE_REGISTRY

    monkeypatch.setenv("CODE_NATIVE_ENABLED", "true")
    importlib.reload(knowledge)
    assert "code-native" in KNOWLEDGE_REGISTRY, "CODE_NATIVE_ENABLED=true did not register code-native"
    yield


def test_native_engine_end_to_end_through_registry_and_router(stores, llm_key, unique_ns, native_on, code_repos, repo_dir):
    from knowledge import code_dispatch, index_store
    from knowledge.router import _PROJECT_CONFIGS, resolve_systems
    from memory_service import get_shared_service
    import worker

    user_id = unique_ns.user_id
    code_space = f"code--{user_id}--demo"

    def cleanup():
        code_dispatch.teardown_code_space("code-native", code_space, user_id, code_repos)
        _PROJECT_CONFIGS.pop(unique_ns.project_id, None)
        r = index_store._redis()
        if r is not None:
            r.delete(index_store._KEY_PREFIX + code_space, index_store._CFG_PREFIX + unique_ns.project_id)

    unique_ns.on_cleanup(cleanup)

    # 1. Index exactly as the ingest worker does.
    result = asyncio.run(
        worker.process_code_index(
            {"service": get_shared_service()},
            {
                "system": "code-native",
                "repo_source": str(repo_dir),
                "code_space": code_space,
                "project_id": unique_ns.project_id,
                "user_id": user_id,
            },
        )
    )
    assert result["ok"] is True, result
    assert result["symbols_indexed"] >= 5, result  # Argument, __init__, render, format_name, parse_arguments, main
    assert result["files_indexed"] >= 2, result

    # The project now routes code tools to the indexed engine (layer 2).
    decision = resolve_systems("anything", project_id=unique_ns.project_id, is_code_tool=True)
    assert decision.layer == 2 and decision.systems[0].info.name == "code-native"
    assert index_store.get_index(code_space)["symbols"] == result["symbols_indexed"]

    # 2. Read ops through registry -> router -> bound system.
    q = _recall(_bind("code-native", code_space, user_id, "query"), query="format_name", operation="query")
    assert "format_name" in q.content, q.content

    n = _recall(
        _bind("code-native", code_space, user_id, "neighbors"),
        query="Argument", label="Argument", operation="neighbors",
    )
    assert "demo.core.Argument" in n.content, n.content
    assert "demo.core.Argument.render" in n.content, n.content  # DEFINES edge

    p = _recall(
        _bind("code-native", code_space, user_id, "path"),
        query="path", source="demo.core.Argument", target="demo.core.Argument.render", operation="path",
    )
    assert "demo.core.Argument.render" in p.content and "hops" in p.content, p.content

    loc = _recall(
        _bind("code-native", code_space, user_id, "locate"),
        query="parse raw strings into Argument objects", operation="locate", limit=5,
    )
    assert any(h["fqn"] == "demo.core.parse_arguments" for h in loc.hits), loc.content

    imp = _recall(
        _bind("code-native", code_space, user_id, "impact"),
        query="format_name", label="format_name", operation="impact",
    )
    assert "demo.core.format_name" in imp.content, imp.content

    # ICE-4: an ambiguous bare label resolves deterministically (same node twice).
    first = _recall(_bind("code-native", code_space, user_id, "neighbors"), query="render", label="render", operation="neighbors")
    second = _recall(_bind("code-native", code_space, user_id, "neighbors"), query="render", label="render", operation="neighbors")
    assert first.content == second.content

    # 3. Operator-side by design: a path outside CODE_REPOS is refused for the NATIVE
    #    engine too (real resolver, real stores), and the error names CODE_REPOS.
    for source in ("/nope", str(repo_dir / "..")):
        missing = asyncio.run(
            worker.process_code_index(
                {"service": get_shared_service()},
                {"system": "code-native", "repo_source": source, "code_space": f"code--{user_id}--nope", "user_id": user_id},
            )
        )
        assert missing["ok"] is False
        assert "CODE_REPOS" in missing["error"] and "server" in missing["error"]

    # 4. Another user cannot index into this user's space.
    foreign = asyncio.run(
        worker.process_code_index(
            {"service": get_shared_service()},
            {"system": "code-native", "repo_source": str(repo_dir), "code_space": code_space, "user_id": "someone-else"},
        )
    )
    assert foreign["ok"] is False and unique_ns.user_id in foreign["error"]


def test_graphify_lib_binds_and_answers_for_code_repos_fixture(unique_ns, code_repos, registry_snapshot):
    """graphify-lib is on by default with the code-graph extra: no flag, no stores."""
    from knowledge.registry import get_system

    assert get_system("code-graphify-lib") is not None, "code-graphify-lib not registered (code-graph extra?)"
    code_space = f"code--{unique_ns.user_id}--demo"
    bound = _bind("code-graphify-lib", code_space, unique_ns.user_id, "query")
    assert bound.health().status == "ok"
    q = _recall(bound, query="format_name", operation="query")
    assert "format_name" in q.content, q.content
    n = _recall(
        _bind("code-graphify-lib", code_space, unique_ns.user_id, "neighbors"),
        query="parse_arguments", label="parse_arguments", operation="neighbors",
    )
    assert "parse_arguments" in n.content, n.content


def _cbm_bridge_url() -> str | None:
    url = os.environ.get("CBM_BRIDGE_URL")
    if not url:
        return None
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=3) as resp:  # noqa: S310
            return url if resp.status == 200 else None
    except Exception:  # noqa: BLE001
        return None


def test_cbm_engine_indexes_and_answers_when_bridge_reachable(unique_ns, code_repos, repo_dir, registry_snapshot, monkeypatch):
    """Needs the cbm-bridge container + CBM binary (not in the standard store stack).

    The bridge indexes by path, so the repo copy's directory must be visible to the
    bridge container too (run with TMPDIR on a shared mount).
    """
    url = _cbm_bridge_url()
    if url is None:
        pytest.skip(
            "cbm-bridge not reachable: set CBM_BRIDGE_URL to a running cbm-bridge "
            "(the gate's store stack does not run it)"
        )
    import knowledge

    monkeypatch.setenv("CBM_ENABLED", "true")
    importlib.reload(knowledge)
    import worker
    from memory_service import get_shared_service

    user_id = unique_ns.user_id
    code_space = f"code--{user_id}--demo"
    from knowledge.code_dispatch import teardown_code_space

    unique_ns.on_cleanup(lambda: teardown_code_space("code-cbm", code_space, user_id, code_repos))
    result = asyncio.run(
        worker.process_code_index(
            {"service": get_shared_service()},
            {"system": "code-cbm", "repo_source": str(repo_dir), "code_space": code_space, "user_id": user_id},
        )
    )
    assert result["ok"] is True, result
    bound = _bind("code-cbm", code_space, user_id, "query")
    q = _recall(bound, query="format_name", operation="query")
    assert "format_name" in q.content, q.content
    loc = _recall(_bind("code-cbm", code_space, user_id, "locate"), query="format_name", operation="locate", limit=5)
    assert loc.hits, loc.content
