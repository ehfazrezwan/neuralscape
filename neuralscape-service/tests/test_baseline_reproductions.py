"""Baseline reproductions of known legacy defects, with passing controls.

Each known defect is a strict expected failure: it asserts the behavior the
product should have and is marked ``xfail(strict=True)`` with its reason.  The
defect is therefore demonstrated on the unchanged legacy path, and the moment a
fix lands the test XPASSes, fails the run and forces the registry
(``evidence_registry.SCENARIOS``) and this expectation to be updated.  These
are not tests that the defects are acceptable.

Controls are ordinary passing assertions that pin neighbouring behavior that
works today.  Everything here is unit-level with fakes and deterministic
barriers (no sleeps).  Scenarios whose registry entry has
``needs_real_service=True`` must be reproduced against real stores before they
count as qualified evidence.
"""

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_DIR.parent


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Namespace and identity collisions
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "legacy group ids join user and project with a delimiter that legal ids "
        "may contain, so different (user, project) tuples can share one namespace"
    ),
)
def test_distinct_namespace_tuples_have_distinct_group_ids() -> None:
    from memory.groups import _build_group_id

    colliding_user = _build_group_id("private", "alice--project--payroll")
    plain_project = _build_group_id("private", "alice", project_id="payroll")

    assert colliding_user != plain_project


def test_ordinary_namespace_tuples_stay_distinct() -> None:
    from memory.groups import _build_group_id

    ids = {
        _build_group_id("private", "alice"),
        _build_group_id("private", "bob"),
        _build_group_id("private", "alice", project_id="payroll"),
        _build_group_id("private", "alice", project_id="hiring"),
        _build_group_id("shared", "alice"),
        _build_group_id("shared", "alice", project_id="payroll"),
        _build_group_id("private", "alice", workspace="reference"),
    }
    assert len(ids) == 7
    assert _build_group_id("private", "alice") == "user--alice"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="legacy email slugging collapses every disallowed run to one '-', so distinct emails can share a user id",
)
def test_distinct_emails_have_distinct_user_ids() -> None:
    from identity import derive_user_id

    assert derive_user_id("a+b@example.com", {}) != derive_user_id("a-b@example.com", {})


def test_ordinary_emails_slug_deterministically() -> None:
    from identity import derive_user_id

    alice = derive_user_id("Alice.Smith@Example.com", {})
    assert alice == derive_user_id("alice.smith@example.com", {}) == "alice.smith-example.com"
    assert alice != derive_user_id("bob.smith@example.com", {})
    assert derive_user_id("anyone@example.com", {"anyone@example.com": "mapped-id"}) == "mapped-id"


# ---------------------------------------------------------------------------
# Queued graph publication uses the visibility captured at enqueue time
# ---------------------------------------------------------------------------


class _GraphService:
    def __init__(self, current_visibility: str | None) -> None:
        self.current_visibility = current_visibility
        self.enrich_calls: list[dict] = []

    def get_memory(self, memory_id: str):
        if self.current_visibility is None:
            return None
        return SimpleNamespace(id=memory_id, visibility=self.current_visibility, occurred_at=None)

    def enrich_graph(self, **kwargs) -> bool:
        self.enrich_calls.append(kwargs)
        return True


def _enrich(service: _GraphService, queued_visibility: str) -> dict:
    from worker import process_graph_enrichment

    return _run(
        process_graph_enrichment(
            {"service": service},
            "memory-1",
            "queued content",
            "alice",
            None,
            queued_visibility,
        )
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "the graph job publishes the visibility it was enqueued with and never re-reads the "
        "memory's current visibility, so narrowing a memory while a job is queued is not honored"
    ),
)
def test_queued_graph_job_publishes_current_visibility() -> None:
    service = _GraphService(current_visibility="private")  # narrowed after enqueue

    _enrich(service, queued_visibility="shared")

    assert service.enrich_calls[0]["visibility"] == "private"


def test_queued_graph_job_skips_a_deleted_memory() -> None:
    service = _GraphService(current_visibility=None)

    result = _enrich(service, queued_visibility="shared")

    assert result == {"memory_id": "memory-1", "enriched": False, "skipped": "memory_missing"}
    assert service.enrich_calls == []


def test_queued_graph_job_with_unchanged_visibility_enriches() -> None:
    service = _GraphService(current_visibility="shared")

    result = _enrich(service, queued_visibility="shared")

    assert result["enriched"] is True
    assert service.enrich_calls[0]["visibility"] == "shared"


# ---------------------------------------------------------------------------
# Job keys omit semantic options
# ---------------------------------------------------------------------------


class _RecordingPool:
    def __init__(self) -> None:
        self.job_ids: list[str] = []

    async def enqueue_job(self, name, *args, _job_id=None, **kwargs):
        self.job_ids.append(_job_id)
        return SimpleNamespace(job_id=_job_id)

    async def zadd(self, *args, **kwargs):
        return None

    async def zremrangebyscore(self, *args, **kwargs):
        return None

    async def expire(self, *args, **kwargs):
        return None

    async def set(self, *args, **kwargs):
        return None


def _manager():
    from task_manager import TaskManager

    manager = TaskManager()
    manager.pool = _RecordingPool()
    return manager


MESSAGES = [{"role": "user", "content": "Remember that deploys freeze on Fridays."}]


def _store_id(**kwargs) -> str:
    manager = _manager()
    return _run(manager.enqueue_store(MESSAGES, kwargs.pop("user_id", "alice"), **kwargs))


def _raw_id(**kwargs) -> str:
    manager = _manager()
    arguments = {"content": "Deploys freeze on Fridays.", "user_id": "alice", "category": "convention"}
    arguments.update(kwargs)
    return _run(manager.enqueue_raw(**arguments))


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "the extraction job id is built from message text and event time only, so the same text "
        "sent to a different project, agent or run coalesces onto one job and the later write is dropped"
    ),
)
def test_store_job_id_depends_on_project_and_agent_scope() -> None:
    base = _store_id(project_id="alpha")

    assert _store_id(project_id="beta") != base
    assert _store_id(project_id="alpha", agent_id="agent-2") != base
    assert _store_id(project_id="alpha", run_id="run-2") != base


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "the raw-write job id covers visibility, scope, project and the sensitivity override but "
        "not category, tags or agent, so writes differing only in those options coalesce"
    ),
)
def test_raw_job_id_depends_on_category_and_tags() -> None:
    base = _raw_id()

    assert _raw_id(category="preference") != base
    assert _raw_id(tags=["release"]) != base
    assert _raw_id(agent_id="agent-2") != base


def test_job_ids_are_stable_and_user_scoped() -> None:
    assert _store_id(project_id="alpha") == _store_id(project_id="alpha")
    assert _store_id(project_id="alpha") != _store_id(project_id="alpha", user_id="bob")
    assert _raw_id() == _raw_id()
    assert _raw_id() != _raw_id(user_id="bob")
    # Already part of the raw key today:
    assert _raw_id(project_id="alpha") != _raw_id(project_id="beta")
    assert _raw_id(visibility="private") != _raw_id(visibility="shared")
    assert _store_id(project_id="alpha", occurred_at="2026-01-01") != _store_id(project_id="alpha")


# ---------------------------------------------------------------------------
# Stale revisions
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the edit request has no revision or etag precondition, so a stale edit cannot be refused (schema-level; concurrent-edit reproduction needs real stores)",
)
def test_patch_request_carries_an_expected_revision() -> None:
    from schemas import PatchMemoryRequest

    names = set(PatchMemoryRequest.model_fields)

    assert any(re.search(r"revision|version|etag|if_match", name) for name in names)


def test_patch_request_applies_only_sent_fields() -> None:
    from schemas import PatchMemoryRequest

    request = PatchMemoryRequest.model_validate({"content": "updated text"})

    assert request.model_fields_set == {"content"}
    explicit_null = PatchMemoryRequest.model_validate({"tags": None})
    assert explicit_null.model_fields_set == {"tags"}


# ---------------------------------------------------------------------------
# Graph removal and shared support
# ---------------------------------------------------------------------------


def _remove_episode(monkeypatch, episode_uuid: str, edges: list):
    import graphiti_core.graphiti as graphiti_module

    deleted_edges: list[list[str]] = []
    deleted_nodes: list[list[str]] = []
    deleted_episodes: list[str] = []

    async def delete_episode():
        deleted_episodes.append(episode_uuid)

    episode = SimpleNamespace(
        uuid=episode_uuid,
        entity_edges=[edge.uuid for edge in edges],
        delete=lambda driver: delete_episode(),
    )

    async def get_episode(driver, uuid):
        return episode

    async def get_edges(driver, uuids):
        return edges

    async def no_nodes(driver, episodes):
        return []

    async def record_edges(driver, uuids):
        deleted_edges.append(list(uuids))

    async def record_nodes(driver, uuids):
        deleted_nodes.append(list(uuids))

    monkeypatch.setattr(graphiti_module.EpisodicNode, "get_by_uuid", staticmethod(get_episode))
    monkeypatch.setattr(graphiti_module.EntityEdge, "get_by_uuids", staticmethod(get_edges))
    monkeypatch.setattr(graphiti_module, "get_mentioned_nodes", no_nodes)
    monkeypatch.setattr(graphiti_module.Edge, "delete_by_uuids", staticmethod(record_edges))
    monkeypatch.setattr(graphiti_module.Node, "delete_by_uuids", staticmethod(record_nodes))

    client = object.__new__(graphiti_module.Graphiti)
    client.driver = object()
    _run(client.remove_episode(episode_uuid))
    return [uuid for batch in deleted_edges for uuid in batch], deleted_episodes


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "removing an episode deletes every edge whose first supporting episode is the removed one, "
        "even when other episodes still support the edge (reproduced on a real graph by the engines lane)"
    ),
)
def test_graph_removal_keeps_edges_supported_by_other_episodes(monkeypatch) -> None:
    shared_support = SimpleNamespace(uuid="edge-shared", episodes=["episode-1", "episode-2"])

    deleted, _ = _remove_episode(monkeypatch, "episode-1", [shared_support])

    assert "edge-shared" not in deleted


def test_graph_removal_deletes_unsupported_edges_only_from_removed_episode(monkeypatch) -> None:
    only_here = SimpleNamespace(uuid="edge-only", episodes=["episode-1"])
    owned_elsewhere = SimpleNamespace(uuid="edge-other", episodes=["episode-2", "episode-1"])

    deleted, removed = _remove_episode(monkeypatch, "episode-1", [only_here, owned_elsewhere])

    assert "edge-only" in deleted
    assert "edge-other" not in deleted
    assert removed == ["episode-1"]


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


class _Pool:
    def __init__(self, reachable: bool = True) -> None:
        self.reachable = reachable

    async def ping(self) -> bool:
        if not self.reachable:
            raise ConnectionError("queue unreachable")
        return True


class _UnreachableQdrantMemory:
    """An initialized client whose backing vector store cannot be reached."""

    def __getattr__(self, name):
        raise ConnectionError("vector store unreachable")


def _health(monkeypatch, *, memory, pool_reachable: bool = True) -> dict:
    import knowledge
    import main

    monkeypatch.setattr(knowledge, "list_systems", lambda: [])
    monkeypatch.setattr(main, "_service", SimpleNamespace(_memory=memory, _graphiti=object()))
    monkeypatch.setattr(main, "_task_manager", SimpleNamespace(pool=_Pool(pool_reachable)))
    response = _run(main.health())
    if hasattr(response, "body"):
        import json

        return {"status_code": response.status_code, **json.loads(response.body)}
    return {"status_code": 200, **response}


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "health marks the vector store 'ok' once its client object exists and never probes the store, "
        "so an unreachable store is reported healthy"
    ),
)
def test_health_reports_an_unreachable_vector_store(monkeypatch) -> None:
    report = _health(monkeypatch, memory=_UnreachableQdrantMemory())

    assert report["status"] != "ok"
    assert report["checks"]["vector_store"] != "ok"


def test_health_reports_an_uninitialized_vector_store_as_degraded(monkeypatch) -> None:
    report = _health(monkeypatch, memory=None)
    assert report["status"] == "degraded"
    assert report["checks"]["vector_store"] == "not_initialized"

    unreachable_queue = _health(monkeypatch, memory=object(), pool_reachable=False)
    assert unreachable_queue["status"] == "degraded"
    assert unreachable_queue["checks"]["redis"] == "unreachable"

    healthy = _health(monkeypatch, memory=object())
    assert healthy["status"] == "ok"


# ---------------------------------------------------------------------------
# Backup coverage of the compose profile
# ---------------------------------------------------------------------------

PERSISTENT_VOLUMES = {"neo4j_data", "qdrant_data", "redis_data", "ingest_data", "exemplar_data"}


def _compose_volumes() -> set[str]:
    compose = REPO_ROOT / "docker-compose.yml"
    if not compose.is_file():  # the packaged test image carries only the service tree
        return set()
    names: set[str] = set()
    in_block = False
    for line in compose.read_text(encoding="utf-8").splitlines():
        if re.match(r"^volumes:\s*$", line):
            in_block = True
            continue
        if in_block:
            if re.match(r"^\S", line):
                in_block = False
                continue
            match = re.match(r"^  ([A-Za-z0-9_]+):", line)
            if match:
                names.add(match.group(1))
    return names


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "the compose profile ships no backup tooling that names its persistent volumes; the only "
        "backup definition is an opt-in cluster add-on, and the deployment guide still lists backups as to-do"
    ),
)
def test_compose_backup_tooling_covers_every_persistent_volume() -> None:
    tooling = [
        path
        for directory in ("scripts", "deploy")
        for path in (REPO_ROOT / directory).rglob("*")
        if path.is_file() and "backup" in path.name.lower() and "terraform" not in path.parts
    ]
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in tooling)

    assert tooling
    assert all(volume in text for volume in PERSISTENT_VOLUMES)


def test_compose_declares_the_persistent_volumes() -> None:
    declared = _compose_volumes()
    if not declared:  # not available in the packaged test image
        return
    assert PERSISTENT_VOLUMES <= declared
