"""Scenario-shared fixtures.

``a_fact`` is the one LLM-free write shared by the memory, MCP and plugin
scenarios: user A stores a private, project-scoped fact carrying a unique
nonsense marker, the task is polled to completion, and we wait (bounded) for
it to become searchable. The fixture records what happened and never asserts;
the scenarios do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nsaccept import eventually
from nsaccept.helpers import Written, write_raw

# Module run order. Modules that delete or wipe data, or stop Qdrant, run after
# every module that reads the session's shared facts; the Qdrant ones last so
# they cannot starve other modules' background (graph) work. All other modules keep
# pytest's default (alphabetical) order.
RUN_LAST = {
    "test_context_and_single_get": 1,  # writes a shared fact into ctx.project_id (would list it for B)
    "test_edit_delete": 2,
    "test_delete_privacy": 3,  # bulk-deletes ctx.project_id
    "test_fault_injection": 4,
    "test_graph_leg_budget": 5,  # pauses Neo4j briefly
    "test_backup_restore": 6,  # wipes every store, then restores a backup: after everything else, fault injection and the Neo4j pause included
}


def pytest_collection_modifyitems(config, items):
    def rank(item) -> int:
        stem = str(getattr(item, "path", item.fspath)).rsplit("/", 1)[-1].removesuffix(".py")
        return RUN_LAST.get(stem, 0)

    items.sort(key=rank)  # stable: only the listed modules move


@dataclass
class Fact:
    marker: str
    content: str
    project_id: str
    task_id: str = ""
    status: dict = field(default_factory=dict)
    visible_to_a: bool = False


def check_search_finds_marker(client, fact: Fact) -> None:
    """The family's real recall check: a /v1/search by ``client`` returns the marker."""
    results = client.search(fact.content, project_id=fact.project_id)
    texts = [r.get("memory", "") for r in results]
    assert any(fact.marker in t for t in texts), (
        f"marker {fact.marker} not in {len(results)} search results"
    )


@pytest.fixture(scope="session")
def a_fact(ctx, a) -> Fact:
    marker = f"zq-{ctx.run_id}-marker"
    fact = Fact(
        marker=marker,
        content=(
            f"{marker} The acceptance probe's favourite colour is vermilion "
            f"and it keeps its notes in the {ctx.project_id} sandbox."
        ),
        project_id=ctx.project_id,
    )
    fact.task_id = a.store_raw(
        fact.content,
        "tech_stack",
        scope="project",
        project_id=ctx.project_id,
        visibility="private",
        tags=["acceptance"],
    )
    fact.status = a.poll_task(fact.task_id, timeout=120)
    if fact.status.get("status") == "completed":
        try:
            eventually(
                lambda: check_search_finds_marker(a, fact),
                timeout=90,
                what="user A's fact searchable",
            )
            fact.visible_to_a = True
        except AssertionError:
            fact.visible_to_a = False
    return fact


# ── shared writes (session scoped: write once, assert many) ─────────────


def shared_project(ctx) -> str:
    """The project holding the run's shared-visibility fact. Separate from
    ``ctx.project_id`` so A's private project stays private in /v1/projects
    (a shared project is listed for every user)."""
    return f"{ctx.project_id}-shared"


@pytest.fixture(scope="session")
def shared_fact(ctx, a) -> Written:
    """A's fact in ``shared_project(ctx)`` written with visibility=shared (readable by user B)."""
    marker = f"zq-{ctx.run_id}-shared"
    return write_raw(
        a,
        f"{marker} The team retrospective room is called Cormorant and has a green door.",
        "tech_stack",
        marker=marker,
        scope="project",
        project_id=shared_project(ctx),
        visibility="shared",
        tags=["acceptance"],
    )


@dataclass
class GraphSeed:
    """Facts written early so the (slow, async) graph worker has time to enrich them."""

    ent_a: str
    ent_b: str
    ent_c: str
    occurred_at: str
    first: Written
    repeat: Written   # byte-identical second write of ``first`` (content-hash dedup)
    sibling: Written  # shares entity ent_a with ``first``


@pytest.fixture(scope="session")
def graph_seed(ctx, a) -> GraphSeed:
    """Request this FIRST in a session so graph enrichment overlaps the other scenarios."""
    ent = lambda tag: f"Zq{ctx.run_id.capitalize()}{tag}"  # noqa: E731 - single-token entity names
    seed_marker = f"zq-{ctx.run_id}-gseed"
    ent_a, ent_b, ent_c = ent("Aaa"), ent("Bbb"), ent("Ccc")
    occurred_at = "2021-03-04T05:06:07Z"
    kw = dict(scope="project", project_id=ctx.project_id, visibility="private")
    text_first = f"{seed_marker} {ent_a} is a service that depends on {ent_b} for its cache layer."
    first = write_raw(a, text_first, "tech_stack", marker=seed_marker, occurred_at=occurred_at, **kw)
    repeat = write_raw(a, text_first, "tech_stack", marker=seed_marker, occurred_at=occurred_at, **kw)
    sib_marker = f"zq-{ctx.run_id}-gsib"
    sibling = write_raw(
        a, f"{sib_marker} {ent_a} is owned by the {ent_c} team.", "tech_stack", marker=sib_marker, **kw
    )
    return GraphSeed(ent_a, ent_b, ent_c, occurred_at, first, repeat, sibling)


@dataclass
class ConvSeed:
    """The suite's ONE LLM-extraction write: a 22-message conversation that also
    crosses the session-summary threshold (20 messages) for its ``session_id``."""

    forge: str
    lagoon: str
    project_id: str
    session_id: str
    write: Written = field(default_factory=lambda: Written("", ""))
    memory_ids: list[str] = field(default_factory=list)
    messages: int = 0


@pytest.fixture(scope="session")
def conv_seed(ctx, a) -> ConvSeed:
    from nsaccept.helpers import memory_ids_in

    forge, lagoon = f"zq-{ctx.run_id}-forge", f"zq-{ctx.run_id}-lagoon"
    session_id = f"acc-{ctx.run_id}-sess"
    seed = ConvSeed(forge, lagoon, ctx.project_id, session_id, Written(forge, ""))
    messages = [
        {
            "role": "user",
            "content": (
                f"For the record: my build tool is {forge}. Separately, the staging "
                f"database for this project is called {lagoon}. Please remember both."
            ),
        },
        {"role": "assistant", "content": f"Noted: build tool {forge}, staging database {lagoon}."},
    ]
    for i in range(10):  # chatter, so the session crosses the 20-message summary threshold
        messages.append({"role": "user", "content": f"Thanks, that is all for now (turn {i + 1})."})
        messages.append({"role": "assistant", "content": "You are welcome."})
    seed.messages = len(messages)
    try:
        resp = a.post(
            "/v1/memories",
            json={"messages": messages, "project_id": ctx.project_id, "run_id": session_id},
        )
        assert resp.status_code == 202, f"POST /v1/memories -> {resp.status_code}: {resp.text[:200]}"
        seed.write.task_id = resp.json()["task_id"]
        seed.write.status = a.poll_task(seed.write.task_id, timeout=240)
        seed.memory_ids = memory_ids_in(seed.write.status)
        seed.write.memory_id = seed.memory_ids[0] if seed.memory_ids else ""
    except AssertionError as exc:
        seed.write.error = str(exc)
    return seed
