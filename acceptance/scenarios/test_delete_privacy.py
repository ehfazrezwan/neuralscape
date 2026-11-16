"""Delete privacy through REST: a deleted fact must be gone everywhere.

User A remembers a 3-fact conversation (ONE graph episode for all three
facts), deletes one fact, and that fact must be absent from memory search,
graph search and episode recall while its siblings are still present. Then a
bulk delete wipes the project and nothing is left. Assertions use unique
nonsense markers only, never LLM wording. Tests in this file run in order and
share the ``convo`` fixture state.
"""

from __future__ import annotations

import re

import pytest

from nsaccept import eventually, expect_check_fails, negative_control, proves


class Convo:
    def __init__(self, tag: str, project_id: str):
        self.tag = tag
        self.project_id = project_id
        # Letters-only invented proper nouns, matched as whole words, case-insensitively.
        # The real extractor keeps an invented proper noun but drops digits/hex glued to it
        # (dev 8c3e130e: "Quillfeather<run tag>" never came back verbatim), so per-run
        # uniqueness comes from the per-run project id, not from the marker (as in #510).
        self.markers = {"dog": "Quillfeather", "city": "Brindlemoor", "tea": "Velvetbrew"}
        self.ids: dict[str, str] = {}     # marker key -> memory id
        self.ready = False


def has(marker: str, text: str) -> bool:
    """Whole-word, case-insensitive marker match (the extractor may re-case words)."""
    return re.search(rf"\b{re.escape(marker)}\b", text or "", re.IGNORECASE) is not None


def _require(cond: bool, msg: str) -> None:
    assert cond, msg


def _memories(client, project_id):
    r = client.get("/v1/memories", params={"project_id": project_id, "limit": 100})
    assert r.status_code == 200, f"GET /v1/memories -> {r.status_code}"
    return r.json()


def _episodes_text(client, project_id) -> str:
    r = client.get("/v1/graph/episodes", params={"project_id": project_id, "limit": 100})
    assert r.status_code == 200, f"GET /v1/graph/episodes -> {r.status_code}"
    return r.text


def _graph_search_text(client, project_id, query) -> str:
    r = client.post("/v1/graph/search", json={"query": query, "project_id": project_id, "limit": 20})
    assert r.status_code == 200, f"POST /v1/graph/search -> {r.status_code}"
    return r.text


@pytest.fixture(scope="module")
def convo(ctx, a) -> Convo:
    tag = re.sub(r"[^A-Za-z0-9]", "", ctx.run_id)[:10] or "run"
    c = Convo(tag, ctx.project_id)
    msgs = [{"role": "user", "content": (
        f"Profile facts. My dog is named {c.markers['dog']}. I live in the town of {c.markers['city']}. "
        f"My favourite tea is the {c.markers['tea']} blend."
    )}, {"role": "assistant", "content": "Noted."}]
    resp = a.post("/v1/memories", json={"messages": msgs, "project_id": c.project_id})
    assert resp.status_code == 202, f"POST /v1/memories -> {resp.status_code}: {resp.text[:200]}"
    status = a.poll_task(resp.json()["task_id"], timeout=300)
    assert status.get("status") == "completed", f"conversation write ended as {status}"

    def have_facts():
        rows = _memories(a, c.project_id)
        for key, marker in c.markers.items():
            hit = [m for m in rows if has(marker, m.get("memory", ""))]
            assert hit, f"no extracted fact carries {marker}"
            c.ids[key] = hit[0]["id"]

    eventually(have_facts, timeout=120, what="three extracted facts stored")
    # graph enrichment runs async on the graph worker
    eventually(
        lambda: _require(has(c.markers["dog"], _episodes_text(a, c.project_id)), "episode not in graph"),
        timeout=300, what="conversation episode in the graph",
    )
    c.ready = True
    return c


def check_victim_absent_everywhere(client, c: Convo) -> None:
    """The real privacy check: the deleted fact is unreachable through every read surface."""
    marker = c.markers["dog"]
    assert client.get(f"/v1/memories/{c.ids['dog']}").status_code == 404, "memory still readable by id"
    hits = client.search(marker, project_id=c.project_id)
    assert not any(has(marker, h.get("memory", "")) for h in hits), "deleted fact still in search"
    assert not has(marker, _graph_search_text(client, c.project_id, marker)), "deleted fact in graph search"
    assert not has(marker, _episodes_text(client, c.project_id)), "deleted fact in episode recall"


@proves("delete.single", "graph.episode-cascade")
@negative_control
def test_privacy_check_fails_before_the_delete(a, convo):
    assert convo.ready
    expect_check_fails(check_victim_absent_everywhere, a, convo)


@proves("delete.single", "graph.episode-cascade", "read.hybrid-search-core")
def test_deleted_fact_is_absent_everywhere_siblings_remain(a, convo):
    assert convo.ready
    r = a.delete(f"/v1/memories/{convo.ids['dog']}")
    assert r.status_code == 200, f"DELETE -> {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert not body.get("graph_jobs_not_enqueued_ids"), f"sibling re-ingest not queued: {body}"
    eventually(lambda: check_victim_absent_everywhere(a, convo), timeout=60, what="deleted fact absent everywhere")
    # siblings are still present as memories ...
    for key in ("city", "tea"):
        assert a.get(f"/v1/memories/{convo.ids[key]}").status_code == 200, f"sibling {key} lost"
    # ... and regain graph evidence once their re-ingest has run
    eventually(
        lambda: _require(
            has(convo.markers["city"], _episodes_text(a, convo.project_id))
            and has(convo.markers["tea"], _episodes_text(a, convo.project_id)),
            "siblings not back in the graph yet",
        ),
        timeout=300, what="siblings re-enriched into the graph",
    )
    assert not has(convo.markers["dog"], _episodes_text(a, convo.project_id)), "deleted fact reappeared"


@proves("delete.single", "auth.request-identity-resolution")
def test_other_user_cannot_delete_a_fact(a, b, convo):
    assert convo.ready
    r = b.delete(f"/v1/memories/{convo.ids['city']}")
    assert r.status_code in (403, 404), f"user B DELETE -> {r.status_code}"
    assert a.get(f"/v1/memories/{convo.ids['city']}").status_code == 200, "A's fact vanished"


@proves("delete.bulk", "graph.episode-cascade")
def test_bulk_delete_leaves_nothing(a, convo):
    assert convo.ready
    r = a.request("DELETE", "/v1/memories", json={"project_id": convo.project_id})
    assert r.status_code == 200, f"bulk DELETE -> {r.status_code}: {r.text[:200]}"
    assert _memories(a, convo.project_id) == [], "memories remain after bulk delete"
    for marker in convo.markers.values():
        assert not any(has(marker, h.get("memory", "")) for h in a.search(marker, project_id=convo.project_id))
    eventually(
        lambda: _require(
            not any(has(m, _episodes_text(a, convo.project_id)) for m in convo.markers.values()),
            "episode text remains after bulk delete",
        ),
        timeout=60, what="no episode text after bulk delete",
    )
