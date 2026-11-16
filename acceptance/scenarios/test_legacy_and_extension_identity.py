"""Two identity guarantees that are easy to lose on the older surfaces.

1. Extension routes (conversation-compiler ``/compile``): the user the work is
   stored for is the token's user. A body ``user_id`` naming someone else is
   refused before anything is enqueued, and the task that IS enqueued belongs
   to the caller alone. (``/flush`` and ``/events`` are covered in
   ``test_extensions``.)
2. Legacy mem0-era routes (``POST /memories``, ``DELETE /memories``): the write
   goes through the service's own extraction path, and the delete removes the
   caller's memories and their graph facts and nothing of anyone else's. The
   delete runs as a third, disposable identity so the run's A and B stay intact.

The legacy write is one LLM extraction (plus its inline graph episode).
"""

from __future__ import annotations

import os

import pytest

from conftest import check_search_finds_marker
from nsaccept import eventually, expect_check_fails, negative_control, proves
from nsaccept.client import NSClient
from nsaccept.helpers import blob_has, graph_listing, list_rows, rows_with, write_raw
from nsaccept.tokens import mint_local, mint_via_compose

PREFIX = "/v1/extensions/conversation-compiler"
NO_SUCH_DAY = "1999-01-01"  # a compile of a day with no log is a no-op: no LLM, no writes


# ── the real checks ──────────────────────────────────────────────


def check_compile_refused(client: NSClient, body_user: str) -> None:
    r = client.post(f"{PREFIX}/compile", json={"date": NO_SUCH_DAY, "user_id": body_user})
    assert r.status_code == 400, f"POST /compile naming {body_user!r} -> {r.status_code} (want 400): {r.text[:160]}"


def check_recalls(client: NSClient, marker: str) -> None:
    hits = [r for r in client.search(marker, limit=10) if marker in r.get("memory", "")]
    assert hits, f"{marker} not recalled"


def check_gone(client: NSClient, marker: str) -> None:
    assert not [r for r in client.search(marker, limit=10) if marker in r.get("memory", "")], f"{marker} still recalled"
    assert not rows_with(list_rows(client), marker), f"{marker} still listed"
    episodes = [e for e in graph_listing(client, "episodes") if marker in (e.get("content") or "")]
    assert not episodes, f"{marker} still held by {len(episodes)} graph episode(s)"


def check_episode_present(client: NSClient, marker: str) -> None:
    episodes = [e for e in graph_listing(client, "episodes") if marker in (e.get("content") or "")]
    assert episodes, f"no graph episode carries {marker}"


def check_every_memory_has_an_episode(client: NSClient, memories: list[dict]) -> None:
    """Each stored memory (its id or its text) is carried by some graph episode: its graph enrichment finished."""
    assert memories, "the write produced no memories to wait for"
    episodes = graph_listing(client, "episodes")
    waiting = [
        m.get("id") or m.get("memory", "")[:40]
        for m in memories
        if not any(
            (m.get("memory") and m["memory"] in (e.get("content") or "")) or (m.get("id") and blob_has(e, str(m["id"])))
            for e in episodes
        )
    ]
    assert not waiting, f"{len(waiting)} of {len(memories)} memories have no graph episode yet: {waiting}"


# ── extension route identity ─────────────────────────────────────


@proves("auth.extension-route-identity")
def test_compile_naming_another_user_is_refused_and_the_own_compile_belongs_to_the_caller(ctx, a, b):
    check_compile_refused(a, ctx.user_b)
    check_compile_refused(b, ctx.user_a)
    own = a.post(f"{PREFIX}/compile", json={"date": NO_SUCH_DAY, "user_id": ctx.user_a})
    assert own.status_code == 202, f"POST /compile as the caller -> {own.status_code}: {own.text[:160]}"
    task_id = own.json()["task_id"]
    assert a.task_status(task_id).get("task_id") == task_id, "the caller cannot read its own compile task"
    other = b.get(f"/v1/memories/status/{task_id}")
    assert other.status_code in (403, 404), f"another user read A's compile task -> {other.status_code}"


@proves("auth.extension-route-identity")
@negative_control
def test_compile_refusal_check_fails_for_the_callers_own_id(ctx, a):
    expect_check_fails(check_compile_refused, a, ctx.user_a)


# ── legacy POST /memories + DELETE /memories ─────────────────────


@pytest.fixture(scope="module")
def c(ctx):
    """A third identity, minted for this module only (compose issuer, else the local signing secret)."""
    user = f"acc-{ctx.run_id}-legacy"
    secret = os.environ.get("NS_TOKEN_SECRET", "").strip()
    if ctx.compose_project:
        token = mint_via_compose(ctx.compose_project, user, os.environ.get("NS_API_SERVICE", "neuralscape"))
    elif secret:
        token = mint_local(user, secret)
    else:
        pytest.skip("neither --compose-project nor NS_TOKEN_SECRET: cannot mint a disposable third identity")
    client = NSClient(ctx.base_url, token, user)
    yield client
    client.close()


@pytest.fixture(scope="module")
def legacy(ctx, c) -> dict:
    """The third identity's one legacy write: a two-message conversation carrying a unique marker."""
    marker = f"zq-{ctx.run_id}-teablend"
    messages = [
        {"role": "user", "content": f"For the record, my favourite tea blend is called {marker} and I always drink it in the morning. Please remember my preference."},
        {"role": "assistant", "content": f"Noted: your favourite morning tea blend is {marker}."},
    ]
    out: dict = {"marker": marker, "error": "", "texts": [], "memories": []}
    try:
        r = c.post("/memories", json={"messages": messages}, timeout=300)
        assert r.status_code == 200, f"POST /memories -> {r.status_code}: {r.text[:200]}"
        body = r.json()
        assert body.get("status") == "ok", f"legacy add answered {str(body)[:200]}"
        stored = (body.get("result") or {}).get("memories", [])
        out["texts"] = [m.get("memory", "") for m in stored]
        out["memories"] = stored
        out["visibilities"] = {m.get("visibility") for m in stored if marker in m.get("memory", "")}
    except AssertionError as exc:
        out["error"] = str(exc)
    return out


@proves("delete.legacy-passthrough")
def test_legacy_post_memories_stores_what_the_caller_can_recall(c, legacy):
    assert not legacy["error"], legacy["error"]
    assert any(legacy["marker"] in t for t in legacy["texts"]), f"no stored memory carries the marker: {legacy['texts']}"
    # A personal preference is private by category default; a shared row would survive the legacy
    # delete by design, so the later checks need it private (an extraction LLM decision).
    assert legacy["visibilities"] == {"private"}, f"the extracted fact is {legacy['visibilities']}, not private"
    eventually(lambda: check_recalls(c, legacy["marker"]), timeout=90, what="legacy write recalled")
    eventually(lambda: check_episode_present(c, legacy["marker"]), timeout=180, interval=5, what="legacy write's graph episode")


@proves("delete.legacy-passthrough")
@negative_control
def test_recall_and_gone_checks_fail_where_they_should(ctx, a, c):
    """Order-independent: uses its own fresh write by the third identity, not the legacy fixture."""
    marker = f"zq-{ctx.run_id}-ctrlc"
    w = write_raw(c, f"{marker} The third identity's control note is pinned beside the tide chart.", "preference", marker=marker, scope="global", visibility="private").require()
    assert w.memory_id
    eventually(lambda: check_recalls(c, marker), timeout=90, what="control write recalled")
    expect_check_fails(check_recalls, a, marker)  # A cannot see the third identity's memory
    expect_check_fails(check_gone, c, marker)  # and the "deleted" check cannot pass while it exists
    expect_check_fails(check_every_memory_has_an_episode, c, [{"id": "00000000-0000-0000-0000-000000000000", "memory": f"{marker} no episode holds this text"}])  # a memory with no episode is not "enriched"


@proves("delete.legacy-passthrough")
def test_legacy_delete_naming_another_user_is_refused_and_touches_nothing(ctx, a, c, a_fact):
    assert a_fact.visible_to_a, "precondition: A's fact never became searchable"
    r = c.delete("/memories", params={"user_id": ctx.user_a})
    assert r.status_code == 400, f"DELETE /memories?user_id=<another user> -> {r.status_code} (want 400): {r.text[:160]}"
    check_search_finds_marker(a, a_fact)


@proves("delete.legacy-passthrough")
def test_legacy_delete_removes_the_callers_memories_and_graph_facts_and_nobody_elses(ctx, a, c, legacy, a_fact):
    assert not legacy["error"], legacy["error"]
    assert a_fact.visible_to_a, "precondition: A's fact never became searchable"
    # The write produced more than one memory: deleting while a later one is still being enriched leaves the
    # delete's graph cascade nothing to resolve for it, so wait until EVERY memory has its episode.
    eventually(lambda: check_every_memory_has_an_episode(c, legacy["memories"]), timeout=180, interval=5, what="every memory's graph episode before the delete")
    r = c.delete("/memories", timeout=300)
    assert r.status_code == 200, f"DELETE /memories -> {r.status_code}: {r.text[:200]}"
    assert r.json().get("status") == "ok", f"legacy delete answered {str(r.json())[:200]}"
    eventually(lambda: check_gone(c, legacy["marker"]), timeout=90, what="third identity's memory, listing and episode gone")
    check_search_finds_marker(a, a_fact)  # A's data is untouched
    assert list_rows(a, project_id=ctx.project_id), "A's listing is empty after another user's legacy delete"
