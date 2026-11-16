"""Real Redis: a config rewritten by the worker path reaches a process that cached it."""

from __future__ import annotations

import time

from knowledge import index_store, router
from knowledge.router import ProjectKnowledgeConfig, get_project_config, set_project_config


def test_worker_rewrite_visible_to_api_router_after_bound(stores, unique_ns, monkeypatch):
    project_id = unique_ns.project_id
    import config

    monkeypatch.setattr(config.settings, "redis_url", stores.redis_url)
    index_store._reset_for_tests()
    monkeypatch.setattr(index_store, "config_ttl", lambda: 0.3)

    def cleanup():
        router._PROJECT_CONFIGS.pop(project_id, None)
        router._PROJECT_CONFIG_AT.pop(project_id, None)
        r = index_store._redis()
        if r is not None:
            r.delete(index_store._CFG_PREFIX + project_id)
        index_store._reset_for_tests()

    unique_ns.on_cleanup(cleanup)
    assert index_store._redis() is not None, "real Redis expected"

    # API process: write through the router, then read (cached in the mirror).
    set_project_config(ProjectKnowledgeConfig(project_id=project_id, code_systems=["code-cbm"], code_space="code--u--old"))
    assert get_project_config(project_id).code_space == "code--u--old"

    # Worker process: rewrites through the worker's own write function
    # (router.set_project_config), then this process's caches are put back as
    # they were, because in production that write happened in another process.
    snap = (
        dict(router._PROJECT_CONFIGS),
        dict(router._PROJECT_CONFIG_AT),
        dict(index_store._mem_cfg),
        dict(index_store._mem_cfg_at),
    )
    set_project_config(ProjectKnowledgeConfig(project_id=project_id, code_systems=["code-cbm"], code_space="code--u--new"))
    for live, saved in zip((router._PROJECT_CONFIGS, router._PROJECT_CONFIG_AT, index_store._mem_cfg, index_store._mem_cfg_at), snap):
        live.clear()
        live.update(saved)
    assert get_project_config(project_id).code_space == "code--u--old"  # stale until the bound

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if get_project_config(project_id).code_space == "code--u--new":
            break
        time.sleep(0.1)
    assert get_project_config(project_id).code_space == "code--u--new"
