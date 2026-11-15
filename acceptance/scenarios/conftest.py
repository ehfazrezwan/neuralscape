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
