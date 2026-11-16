"""Real-Qdrant proof for the session-start project context bundle.

Atlas node ``read.project-context``; the proof deferred from read-04. The unit
tests mock the Qdrant client, so they cannot show that the nested filter, the
server-side ``order_by created_at`` and the exact ``count`` behave on a real
server. This does, through ``MemoryService.get_project_context`` over a
throwaway collection:

* the nested filter returns the caller's own rows (global scope plus this
  project) UNION the project's ``visibility=shared`` rows from other writers,
  memory workspace only; a teammate's private row, a shared row of another
  project, a shared row in a reference workspace, the caller's own row of
  another project and a tombstoned row are never returned;
* ``order_by created_at`` is really used (no fallback warning) and paging over
  a tie group (several rows sharing one ``created_at``, cut by a page edge)
  covers every row exactly once, in (created_at, id) descending order;
* rows with no ``created_at`` come last;
* ``total`` is the exact count of the matching set on every page, and
  ``has_more`` is true exactly until the last page.

Rows are written with ``store_raw`` (the real write path); only ``created_at``
is then edited in place with ``set_payload``/``delete_payload``, because a real
write always stamps "now" and a tie group or an undated legacy row cannot be
produced any other way.

Needs Neo4j (the service initialises its graph store), Qdrant and
``GOOGLE_API_KEY`` (embeddings). On the gate (``NS_INTEGRATION_REQUIRED=1``)
missing infrastructure fails the test.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import logging
import uuid

import pytest

T_NEWEST = "2031-03-01T10:00:00+00:00"
T_TIE = "2031-02-01T10:00:00+00:00"  # the tie group the page edge falls in
T_OLDER = "2031-01-01T10:00:00+00:00"


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns):
    """A MemoryService over a throwaway Qdrant collection (dropped at teardown)."""
    from qdrant_client import QdrantClient

    flags(qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url)
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    return service


def test_project_context_nested_filter_ordering_paging_and_count_on_real_qdrant(svc, unique_ns, caplog):
    from extensions.dreaming.consolidate import _tombstone

    me = unique_ns.user_id
    mate = f"{me}-mate"
    project = unique_ns.project_id
    other_project = f"{project}-other"
    tag = uuid.uuid4().hex[:6]
    client = svc._get_memory().vector_store.client
    coll = unique_ns.qdrant_collection

    def store(label, *, user, category="decision", scope="project", project_id=project,
              visibility="private", workspace=None):
        out = svc.store_raw(
            content=f"Context row {label} zq{tag}: a distinct fact about {label}.",
            user_id=user, category=category, scope=scope,
            project_id=project_id if scope == "project" else None,
            visibility=visibility, workspace=workspace, add_to_graph=False,
        )
        by_label[label] = out[0].id
        return out[0].id

    # ── visible rows (id -> created_at; None = undated) ────────────────────
    visible: dict[str, str | None] = {}
    by_label: dict[str, str] = {}
    # newest group (2 rows): own project rows
    for label in ("own-new-1", "own-new-2"):
        visible[store(label, user=me)] = T_NEWEST
    # the tie group (4 rows, the page edge falls inside it): own + teammate's shared
    for label in ("own-tie-1", "own-tie-2"):
        visible[store(label, user=me)] = T_TIE
    for label in ("mate-shared-tie-1", "mate-shared-tie-2"):
        visible[store(label, user=mate, visibility="shared")] = T_TIE
    # an older own GLOBAL-scope row (no project at all) is part of "own"
    visible[store("own-global", user=me, scope="global", category="preference")] = T_OLDER
    # two undated rows: invisible to order_by, must come last
    for label in ("own-undated-1", "mate-shared-undated"):
        mine = label.startswith("own")
        visible[store(label, user=me if mine else mate, visibility="private" if mine else "shared")] = None

    # ── rows that must never be returned ───────────────────────────────────
    hidden = {
        "mate-private": store("mate-private", user=mate),  # a teammate's private row
        "mate-shared-other-project": store(
            "mate-shared-other-project", user=mate, visibility="shared", project_id=other_project
        ),
        "mate-shared-reference": store(
            "mate-shared-reference", user=mate, visibility="shared", workspace="reference"
        ),
        "own-other-project": store("own-other-project", user=me, project_id=other_project),
        "own-tombstoned": store("own-tombstoned", user=me),
    }
    for mid in hidden.values():
        client.set_payload(collection_name=coll, payload={"created_at": T_NEWEST}, points=[mid])
    _tombstone(svc, hidden["own-tombstoned"], superseded_by=None)

    for mid, created in visible.items():
        if created is None:
            client.delete_payload(collection_name=coll, keys=["created_at"], points=[mid])
        else:
            client.set_payload(collection_name=coll, payload={"created_at": created}, points=[mid])

    expected_order = sorted(
        visible,
        key=lambda mid: ((visible[mid] is not None), visible[mid] or "", mid),  # dated first by (ts, id), undated last
        reverse=True,
    )
    # sorted(reverse=True) puts the undated group (False) after the dated group (True): newest first.
    undated_ids = {mid for mid, created in visible.items() if created is None}
    assert len(expected_order) == 9 and len(undated_ids) == 2

    # ── the whole bundle: exact total, nothing hidden, undated last ────────
    with caplog.at_level(logging.WARNING, logger="memory.reads"):
        full = svc.get_project_context(me, project, limit=50)
        assert full.total == len(visible), f"count must be exact: {full.total} != {len(visible)}"
        assert full.returned == len(visible) and full.has_more is False
        returned = [m.id for bucket in full.categories.values() for m in bucket]
        assert set(returned) == set(visible), set(returned) ^ set(visible)
        assert not set(returned) & set(hidden.values()), "a row that must stay hidden was returned"

        # ── paging over the tie group: every row exactly once, in order ────
        paged: list[str] = []
        flags_has_more: list[bool] = []
        for offset in (0, 3, 6):  # pages of 3: the edge at 3 falls inside the 4-row tie group (positions 2-5)
            page = svc.get_project_context(me, project, limit=3, offset=offset)
            assert page.total == len(visible), f"total changed at offset {offset}: {page.total}"
            assert page.offset == offset and page.limit == 3
            ids = [m.id for bucket in page.categories.values() for m in bucket]
            assert page.returned == len(ids) == 3
            paged.extend(ids)
            flags_has_more.append(page.has_more)
        beyond = svc.get_project_context(me, project, limit=3, offset=9)
        assert beyond.returned == 0 and beyond.has_more is False and beyond.total == len(visible)

    assert sorted(paged) == sorted(visible), "paging skipped or repeated a row at a page edge"
    assert len(set(paged)) == len(paged)
    assert flags_has_more == [True, True, False]
    assert paged[-2:] and set(paged[-2:]) == undated_ids, "undated rows must come last"
    assert paged == expected_order, "pages must follow (created_at, id) descending, undated last"
    # The tie group straddles the first page edge: rows 3 and 4 share one created_at.
    tie_positions = [i for i, mid in enumerate(paged) if visible[mid] == T_TIE]
    assert tie_positions == [2, 3, 4, 5], tie_positions

    # ── the server-side ordered scroll was used, not the Python fallback ───
    assert not [r for r in caplog.records if "order_by scroll failed" in r.getMessage()], (
        "get_project_context fell back to the Python sort: Qdrant rejected order_by"
    )

    # ── the teammate's own view of the same project ────────────────────────
    mate_view = svc.get_project_context(mate, project, limit=50)
    mate_ids = {m.id for bucket in mate_view.categories.values() for m in bucket}
    my_private = {by_label[l] for l in ("own-new-1", "own-new-2", "own-tie-1", "own-tie-2", "own-global", "own-undated-1")}
    mate_own = {by_label[l] for l in ("mate-private", "mate-shared-tie-1", "mate-shared-tie-2", "mate-shared-undated")}
    assert mate_own <= mate_ids, "a user sees their own rows and the project's shared rows"
    assert not mate_ids & my_private, "the owner's private rows leaked to a teammate"
    assert not mate_ids & {hidden["own-other-project"], hidden["own-tombstoned"], hidden["mate-shared-other-project"]}
    # The reference-workspace fence applies to the SHARED branch only (it is why
    # the owner's view above never lists this row); the author's own branch has
    # no workspace filter, so the teammate sees their own reference row.
    assert hidden["mate-shared-reference"] in mate_ids
    assert mate_view.total == len(mate_ids) == mate_view.returned
