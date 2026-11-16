"""Project-less shared facts against the project shared-pool gate (default profile; gap E-05).

The rule the code states (``memory/search.py``, "Dual-scope merge"): a search scoped to a project reads the
shared pool of THAT project plus the project-less (global) shared pool, so a team-wide fact that names no
project is still visible from every project context, while a shared fact that belongs to one project is
not visible from another. A private project-less fact is visible to nobody but its owner.

Writes are LLM-free: one raw fact and one ingested text (passages only). Every read is by user B.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest

from nsaccept import eventually, expect_check_fails, negative_control, proves
from nsaccept.helpers import Written, write_raw

from phase2_kit import Step, call


@dataclass
class Gate:
    project_one: str
    project_two: str
    raw_global: Written
    raw_project: Written
    raw_private: Written
    ingest: Step
    marker_ingest: str


def _q(marker: str, words: str) -> str:
    return f"{marker} {words}"


@pytest.fixture(scope="module")
def gate(ctx, a) -> Gate:
    run = f"{ctx.run_id}{uuid.uuid4().hex[:6]}"  # a rerun with the same NS_RUN_ID must not see leftover text
    one, two = f"{ctx.project_id}-e05a", f"{ctx.project_id}-e05b"
    raw_global = write_raw(a, f"zq-{run}-e05g The tide gauge hut has a blue roof.", "domain_knowledge",
                           marker=f"zq-{run}-e05g", scope="global", visibility="shared")
    raw_project = write_raw(a, f"zq-{run}-e05p The dock office keeps its ledger in a tin box.", "tech_stack",
                            marker=f"zq-{run}-e05p", scope="project", project_id=one, visibility="shared")
    raw_private = write_raw(a, f"zq-{run}-e05x The pilot boat key hangs behind the clock.", "domain_knowledge",
                            marker=f"zq-{run}-e05x", scope="global", visibility="private")
    marker_ingest = f"zq-{run}-e05i"
    ingest = call("ingest", a, "POST", "/v1/ingest/text", json={
        "content": f"{marker_ingest} Gantry lamps are checked before the first tide each morning.",
        "title": f"{marker_ingest}", "category": "domain_knowledge", "scope": "global", "visibility": "shared",
        "extract_facts": False,
    })
    return Gate(one, two, raw_global, raw_project, raw_private, ingest, marker_ingest)


def check_found(client, marker: str, words: str, project: str) -> dict:
    hits = client.search(_q(marker, words), project_id=project, limit=20)
    for hit in hits:
        if marker in hit.get("memory", ""):
            return hit
    raise AssertionError(f"{marker} not among {len(hits)} results scoped to {project}")


def check_absent(client, marker: str, words: str, project: str) -> None:
    """The search WORKED (HTTP 200) and its hit list does not carry the marker: an erroring search is not 'absent'."""
    r = client.post("/v1/search", json={"query": _q(marker, words), "project_id": project, "limit": 20})
    assert r.status_code == 200, f"POST /v1/search -> {r.status_code}: {r.text[:120]}"
    hits = r.json().get("results", [])
    assert isinstance(hits, list), f"search answer has no result list: {str(r.json())[:120]}"
    assert not any(marker in h.get("memory", "") for h in hits), f"{marker} was recalled from {project}"


@proves("model.visibility-tiers", "read.hybrid-search-core")
def test_a_project_less_shared_fact_is_recalled_from_a_project_it_does_not_name(b, gate):
    gate.raw_global.require()
    eventually(lambda: check_found(b, gate.raw_global.marker, "tide gauge hut blue roof", gate.project_two),
               timeout=90, what="B recalls the project-less shared fact from another project")


@proves("model.visibility-tiers", "ingest.text-and-document", "ingest.pipeline")
def test_a_project_less_shared_ingest_is_recalled_from_a_project_it_does_not_name(a, b, gate):
    assert gate.ingest.status == 202, gate.ingest.describe()
    assert gate.ingest.final.get("status") == "completed", f"ingest task ended as {gate.ingest.final}"
    assert (gate.ingest.final.get("result") or {}).get("passages", 0) >= 1, f"no passages indexed: {gate.ingest.final}"
    words = "gantry lamps checked first tide morning"
    eventually(lambda: check_found(a, gate.marker_ingest, words, gate.project_two), timeout=90, what="A recalls its own passage")
    eventually(lambda: check_found(b, gate.marker_ingest, words, gate.project_two), timeout=90, what="B recalls the shared passage")


@proves("model.visibility-tiers", "read.hybrid-search-core")
def test_a_shared_fact_of_one_project_is_recalled_from_that_project_and_not_from_another(b, gate):
    gate.raw_project.require()
    words = "dock office ledger tin box"
    eventually(lambda: check_found(b, gate.raw_project.marker, words, gate.project_one), timeout=90, what="B recalls it from its own project")
    check_absent(b, gate.raw_project.marker, words, gate.project_two)


@proves("model.visibility-tiers")
def test_a_private_project_less_fact_is_recalled_from_no_project_by_the_other_user(a, b, gate):
    gate.raw_private.require()
    words = "pilot boat key hangs behind clock"
    eventually(lambda: check_found(a, gate.raw_private.marker, words, gate.project_two), timeout=90, what="A recalls its own private fact")
    for project in (gate.project_one, gate.project_two):
        check_absent(b, gate.raw_private.marker, words, project)


@proves("model.visibility-tiers")
@negative_control
def test_recall_check_fails_for_a_private_project_less_fact(a, b, gate):
    gate.raw_private.require()
    words = "pilot boat key hangs behind clock"
    eventually(lambda: check_found(a, gate.raw_private.marker, words, gate.project_two), timeout=90, what="A recalls it first")
    check_absent(b, gate.raw_private.marker, words, gate.project_two)  # B's search works and is empty
    expect_check_fails(check_found, b, gate.raw_private.marker, words, gate.project_two)
