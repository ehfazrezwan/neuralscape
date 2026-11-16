"""Edit concurrency and graph visibility through REST.

1. A PATCH carrying a stale ``expected_revision`` is refused with 409 and writes
   nothing; the current revision applies. (``revision`` is returned by reads.)
2. A memory written ``shared`` and narrowed to ``private`` right after the write
   never appears in the shared graph pool, whichever of the queued graph job and
   the edit's own re-ingest runs first: user B (who can read the project's shared
   pool) finds nothing, while user A finds the episode in their own pool.

Not covered here, on purpose: ``DELETE /v1/memories`` answering 503 when the
service cannot list the caller's rows. That needs the vector store to fail
mid-request, which a black-box client cannot provoke; it is proven by the unit
tests (``tests/test_legacy_rest_routes.py``) and ``integration_tests``.

Assertions use unique nonsense markers only, never LLM wording.
"""

from __future__ import annotations

import re

import pytest

from nsaccept import eventually, expect_check_fails, negative_control, proves


def _tag(ctx) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", ctx.run_id)[:10] or "run"


def _find(client, project_id, marker):
    r = client.get("/v1/memories", params={"project_id": project_id, "limit": 100})
    assert r.status_code == 200, f"GET /v1/memories -> {r.status_code}"
    hits = [m for m in r.json() if marker in m.get("memory", "")]
    assert hits, f"no stored memory carries {marker}"
    return hits[0]


def _episodes_text(client, project_id) -> str:
    r = client.get("/v1/graph/episodes", params={"project_id": project_id, "limit": 100})
    assert r.status_code == 200, f"GET /v1/graph/episodes -> {r.status_code}"
    return r.text


def _graph_search_text(client, project_id, query) -> str:
    r = client.post("/v1/graph/search", json={"query": query, "project_id": project_id, "limit": 20})
    assert r.status_code == 200, f"POST /v1/graph/search -> {r.status_code}"
    return r.text


# ── 1. stale PATCH is refused ────────────────────────────────────────


@pytest.fixture(scope="module")
def editable(ctx, a):
    marker = f"zq-{ctx.run_id}-revision"
    task = a.store_raw(
        f"{marker} The acceptance probe keeps this note for the revision scenario.",
        "tech_stack", scope="project", project_id=ctx.project_id, visibility="private",
    )
    assert a.poll_task(task, timeout=120).get("status") == "completed", "write did not complete"
    row = eventually(lambda: _find(a, ctx.project_id, marker), timeout=90, what="memory stored")
    return {"id": row["id"], "revision": row.get("revision"), "marker": marker}


def check_stale_patch_is_refused(client, memory_id: str, stale_revision: str, tag: str) -> None:
    """The real check: a PATCH quoting ``stale_revision`` is 409 and its change is not stored."""
    r = client.request("PATCH", f"/v1/memories/{memory_id}",
                       json={"tags": [tag], "expected_revision": stale_revision})
    assert r.status_code == 409, f"stale PATCH -> {r.status_code}: {r.text[:200]}"
    got = client.get(f"/v1/memories/{memory_id}").json()
    assert tag not in (got.get("tags") or []), "refused PATCH still wrote its tag"


@proves("edit.patch-memory")
def test_read_returns_a_revision_and_a_current_patch_applies(a, editable):
    assert editable["revision"] is not None, "GET /v1/memories rows carry no revision"
    r = a.request("PATCH", f"/v1/memories/{editable['id']}",
                  json={"tags": ["acc-first"], "expected_revision": editable["revision"]})
    assert r.status_code == 200, f"current-revision PATCH -> {r.status_code}: {r.text[:200]}"
    new_rev = r.json()["memory"]["revision"]
    assert new_rev != editable["revision"], "a successful edit did not advance the revision"
    assert "acc-first" in (a.get(f"/v1/memories/{editable['id']}").json().get("tags") or [])
    editable["new_revision"] = new_rev


@proves("edit.patch-memory")
def test_stale_patch_is_409_and_writes_nothing(a, editable):
    assert editable.get("new_revision"), "precondition: the first edit did not apply"
    check_stale_patch_is_refused(a, editable["id"], editable["revision"], "acc-stale")
    r = a.request("PATCH", f"/v1/memories/{editable['id']}",
                  json={"tags": ["acc-stale"], "expected_revision": editable["revision"]})
    detail = r.json().get("detail", {})
    assert detail.get("current_revision") == editable["new_revision"], f"409 body: {r.text[:200]}"


@proves("edit.patch-memory")
def test_patch_without_a_revision_still_applies(a, editable):
    r = a.request("PATCH", f"/v1/memories/{editable['id']}", json={"tags": ["acc-lww"]})
    assert r.status_code == 200, f"unconditional PATCH -> {r.status_code}: {r.text[:200]}"
    assert "acc-lww" in (a.get(f"/v1/memories/{editable['id']}").json().get("tags") or [])


@proves("edit.patch-memory")
@negative_control
def test_stale_check_fails_when_the_revision_is_current(a, editable):
    current = a.get(f"/v1/memories/{editable['id']}").json()["revision"]
    expect_check_fails(check_stale_patch_is_refused, a, editable["id"], current, "acc-neg")


# ── 2. narrowing to private leaves nothing in the shared graph pool ───


@pytest.fixture(scope="module")
def narrowed(ctx, a, b):
    tag = _tag(ctx)
    marker = f"Quillfeather{tag}narrow"
    content = f"The vessel {marker} is moored at Reykjavik harbour and belongs to the survey team."
    task = a.store_raw(content, "tech_stack", scope="project", project_id=ctx.project_id, visibility="shared")
    assert a.poll_task(task, timeout=120).get("status") == "completed", "shared write did not complete"
    row = eventually(lambda: _find(a, ctx.project_id, marker), timeout=90, what="shared memory stored")
    # narrow it straight away: the graph job queued by the write has very likely not run yet
    r = a.request("PATCH", f"/v1/memories/{row['id']}", json={"visibility": "private"})
    assert r.status_code == 200, f"narrowing PATCH -> {r.status_code}: {r.text[:200]}"
    return {"id": row["id"], "marker": marker, "project_id": ctx.project_id}


def check_marker_not_visible_through_graph(client, n) -> None:
    assert n["marker"] not in _episodes_text(client, n["project_id"]), "marker in graph episodes"
    assert n["marker"] not in _graph_search_text(client, n["project_id"], n["marker"]), "marker in graph search"


@proves("model.graph-group-ids", "graph.deferred-enrichment")
def test_owner_reaches_the_narrowed_memory_in_their_own_graph(a, narrowed):
    eventually(
        lambda: _assert_in(_episodes_text(a, narrowed["project_id"]), narrowed["marker"]),
        timeout=300, what="narrowed memory's episode in the owner's private graph",
    )


def _holds_for(check, *, window: float, interval: float) -> None:
    """Assert ``check`` passes on EVERY poll across ``window`` seconds (unlike ``eventually``, which returns on the first pass)."""
    import time

    deadline = time.monotonic() + window
    while True:
        check()
        if time.monotonic() >= deadline:
            return
        time.sleep(interval)


def _assert_in(text: str, marker: str) -> None:
    assert marker in text, "not there yet"


@proves("model.graph-group-ids", "graph.deferred-enrichment")
def test_narrowed_memory_is_absent_from_the_shared_graph_pool(b, narrowed):
    # (ordered after the owner reached it: both jobs have had their chance to run)
    check_marker_not_visible_through_graph(b, narrowed)
    # and stays absent for the whole observation window: a late-running stale job must not publish it either
    _holds_for(lambda: check_marker_not_visible_through_graph(b, narrowed), window=30, interval=5)
    assert b.get(f"/v1/memories/{narrowed['id']}").status_code == 404, "B can read the narrowed memory"


@proves("model.graph-group-ids")
@negative_control
def test_graph_check_fails_for_the_owner(a, narrowed):
    eventually(
        lambda: _assert_in(_episodes_text(a, narrowed["project_id"]), narrowed["marker"]),
        timeout=300, what="owner sees the episode",
    )
    expect_check_fails(check_marker_not_visible_through_graph, a, narrowed)
