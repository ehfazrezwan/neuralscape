"""Flag-on proof against real Qdrant: the authoritative standards tier and the process registry.

STANDARDS_ENABLED / PROCESSES_ENABLED / DICTATOR_USER_IDS are off by default.
Through the real ``MemoryService`` over a throwaway collection, with the flags on:

* a dictator's ``standard`` write is stored (global scope, no project) and a
  non-dictator's raises ``PermissionError`` (the ``store_raw`` backstop); the
  synchronous REST/MCP gate ``main._authorize_standard_write`` agrees (403);
* every user's session-start global and project context carries the standard
  tagged ``critical``/``always``, and not an untagged one (always-inject is a
  subset); an untagged standard is still found by recall, by any user;
* flag-off contrast: the same write is refused, nothing is injected into
  context and the standard pool is not searched;
* the process registry (``process:<slug>`` standards) lists / serves a process
  bundle for any caller when PROCESSES_ENABLED, and is empty/None when off.

Needs Qdrant, Neo4j (the service initialises its graph store) and
``GOOGLE_API_KEY`` (embeddings). On the gate missing infrastructure fails the test.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import uuid

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns):
    """A MemoryService over a throwaway Qdrant collection, standards on, one dictator."""
    from qdrant_client import QdrantClient

    flags(
        qdrant_collection=unique_ns.qdrant_collection,
        qdrant_url=stores.qdrant_url,
        standards_enabled=True,
        processes_enabled=True,
        dictator_user_ids=unique_ns.user_id,
    )
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    return service


def _standard(svc, user, text, tags=None, category="convention"):
    return svc.store_raw(
        content=text, user_id=user, category=category, visibility="standard",
        tags=tags or [], add_to_graph=False,
    )


def _ids(items):
    return {getattr(i, "id", None) for i in items}


def test_dictator_writes_a_standard_and_a_non_dictator_is_refused(svc, unique_ns):
    me, other = unique_ns.user_id, unique_ns.user_id + "-other"
    tok = uuid.uuid4().hex[:8]
    stored = _standard(svc, me, f"Always sign release tags zq{tok}.", tags=["critical"])
    assert stored
    row = svc.get_memory(stored[0].id)
    assert row.visibility == "standard" and row.scope == "global" and row.project_id is None

    with pytest.raises(PermissionError, match="not authorized"):
        _standard(svc, other, f"Forged directive zq{tok}.", tags=["critical"])
    # the refused write left no row behind
    assert not [r for r in svc._get_standards(critical_only=False) if "Forged" in r.memory]


def test_synchronous_gate_agrees_with_the_service(svc, unique_ns):
    import main
    from fastapi import HTTPException

    main._authorize_standard_write(unique_ns.user_id, "standard")  # dictator passes
    with pytest.raises(HTTPException) as exc:
        main._authorize_standard_write(unique_ns.user_id + "-other", "standard")
    assert exc.value.status_code == 403
    main._authorize_standard_write(unique_ns.user_id + "-other", "private")  # other tiers untouched


def test_every_users_context_gets_critical_standards_only(svc, unique_ns):
    me, other = unique_ns.user_id, unique_ns.user_id + "-reader"
    tok = uuid.uuid4().hex[:8]
    crit = _standard(svc, me, f"Never commit secrets zq{tok}.", tags=["critical"])[0].id
    alw = _standard(svc, me, f"Run tests before merging zq{tok}.", tags=["always"])[0].id
    plain = _standard(svc, me, f"Prefer small commits zq{tok}.")[0].id

    for reader in (me, other, unique_ns.user_id + "-nobody"):
        g = svc.get_global_context(reader)
        assert {crit, alw} <= _ids(g.standards), f"{reader}: critical standards missing from global context"
        assert plain not in _ids(g.standards), "an untagged standard must not be always-injected"
        p = svc.get_project_context(reader, project_id=unique_ns.project_id)
        assert {crit, alw} <= _ids(p.standards)
        assert plain not in _ids(p.standards)


def test_untagged_standard_is_found_by_recall_for_any_user(svc, unique_ns):
    me, other = unique_ns.user_id, unique_ns.user_id + "-reader"
    tok = f"Quillbard{uuid.uuid4().hex[:8]}"
    plain = _standard(svc, me, f"The {tok} rule: every service exposes a health endpoint.")[0].id
    hits = svc.search(query=f"{tok} health endpoint", user_id=other, limit=10)
    assert plain in _ids(hits), "a standard must be recallable by non-dictators"


def test_flag_off_contrast_refuses_writes_and_injects_nothing(svc, unique_ns, flags):
    me = unique_ns.user_id
    tok = uuid.uuid4().hex[:8]
    sid = _standard(svc, me, f"Critical rule zq{tok}.", tags=["critical"])[0].id
    assert sid in _ids(svc.get_global_context(me).standards)  # on: injected

    flags(standards_enabled=False)
    assert svc.get_global_context(me).standards == []
    assert svc._get_standards(critical_only=False) == []
    with pytest.raises(PermissionError, match="disabled"):
        _standard(svc, me, f"Another rule zq{tok}.")
    # The writer's OWN row legitimately comes back through the personal pool
    # (user_id filter, any visibility), so an id check on the writer proves
    # nothing about the standard pool. Assert by pool instead:
    #  1. the standard pool leg is never queried, and
    #  2. a different user (who owns no row; the shared pool only matches
    #     visibility=shared) gets nothing: the standard pool is the only way
    #     they could have reached it.
    other = unique_ns.user_id + "-reader"
    standard_pool_calls = []
    real_pool = svc._search_standard_pool

    def spy(**kwargs):
        standard_pool_calls.append(kwargs)
        return real_pool(**kwargs)

    svc._search_standard_pool = spy
    svc.search(query=f"Critical rule zq{tok}", user_id=me, limit=10)
    other_hits = svc.search(query=f"Critical rule zq{tok}", user_id=other, limit=10)
    assert standard_pool_calls == [], "the standard pool must not be searched when the tier is off"
    assert sid not in _ids(other_hits), "a non-writer must not reach a standard when the tier is off"

    # Same probe with the tier back on: the spy and the reader both see it,
    # so the two assertions above are not vacuous.
    flags(standards_enabled=True)
    on_hits = svc.search(query=f"Critical rule zq{tok}", user_id=other, limit=10)
    assert standard_pool_calls, "tier on: the standard pool leg must be queried"
    assert sid in _ids(on_hits), "tier on: a non-writer must reach the standard"


def test_process_registry_lists_and_serves_a_bundle_for_any_caller(svc, unique_ns):
    me, other = unique_ns.user_id, unique_ns.user_id + "-reader"
    slug = f"rel-{uuid.uuid4().hex[:6]}"
    _standard(svc, me, "Release train\nHow we ship a release.", tags=[f"process:{slug}", "process-def"])
    _standard(svc, me, "Cut the release branch", tags=[f"process:{slug}", "process-step:1"])
    _standard(svc, me, "Tag and publish", tags=[f"process:{slug}", "process-step:10"])
    _standard(svc, me, "Verify the changelog", tags=[f"process:{slug}", "process-step:2"])
    _standard(svc, me, "Never release on Fridays", tags=[f"process:{slug}"])

    listed = {p["slug"]: p for p in svc.list_processes()}
    assert listed[slug]["title"] == "Release train" and listed[slug]["description"] == "How we ship a release."
    bundle = svc.get_process(slug)
    assert bundle["steps"] == ["Cut the release branch", "Verify the changelog", "Tag and publish"]
    assert bundle["guidelines"] == ["Never release on Fridays"]
    assert bundle["definition"].startswith("Release train")
    assert svc.get_process("no-such-process") is None
    # a non-dictator cannot register a process (it is a standard-tier write)
    with pytest.raises(PermissionError):
        _standard(svc, other, "Rogue\nprocess", tags=["process:rogue", "process-def"])
    assert "rogue" not in {p["slug"] for p in svc.list_processes()}


def test_process_registry_off_is_empty_and_refuses_nothing_else(svc, unique_ns, flags):
    me = unique_ns.user_id
    slug = f"off-{uuid.uuid4().hex[:6]}"
    _standard(svc, me, "Offline flow\nSummary", tags=[f"process:{slug}", "process-def"])
    assert slug in {p["slug"] for p in svc.list_processes()}  # on

    flags(processes_enabled=False)
    assert svc.list_processes() == []
    assert svc.get_process(slug) is None
    # the standards tier itself stays on: only the registry is gated
    assert svc._get_standards(critical_only=False)
