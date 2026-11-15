"""A queued graph job enriches the memory's CURRENT state, not the state it was enqueued with.

Graph enrichment runs minutes after the write. Whatever the owner did meanwhile
(narrow shared -> private, move project, rewrite the text) must win over the
values frozen in the job: publishing the queued ones can put a memory into a
shared graph group its owner has since withdrawn it from.
"""

import asyncio
from types import SimpleNamespace

from worker import process_graph_enrichment


class Service:
    def __init__(self, **row):
        self.row = SimpleNamespace(id="m1", occurred_at=None, **row) if row is not None else None
        self.enrich_calls: list[dict] = []

    def get_memory(self, memory_id):
        return self.row

    def enrich_graph(self, **kwargs):
        self.enrich_calls.append(kwargs)
        return True


def _run(service, *, content="the text", visibility="shared", project_id=None):
    return asyncio.run(process_graph_enrichment(
        {"service": service}, "m1", content, "alice", project_id, visibility,
    ))


def test_narrowed_to_private_while_queued_publishes_private():
    svc = Service(visibility="private", project_id=None, memory="the text", workspace=None)
    out = _run(svc, visibility="shared")
    assert out["enriched"] is True
    assert svc.enrich_calls[0]["visibility"] == "private"


def test_widened_while_queued_publishes_the_current_wider_visibility():
    svc = Service(visibility="shared", project_id=None, memory="the text", workspace=None)
    _run(svc, visibility="private")
    assert svc.enrich_calls[0]["visibility"] == "shared"


def test_project_moved_while_queued_uses_the_current_project():
    svc = Service(visibility="private", project_id="new-proj", memory="the text", workspace=None)
    _run(svc, project_id="old-proj", visibility="private")
    assert svc.enrich_calls[0]["project_id"] == "new-proj"


def test_project_cleared_while_queued_is_honoured():
    svc = Service(visibility="private", project_id=None, memory="the text", workspace=None)
    _run(svc, project_id="old-proj", visibility="private")
    assert svc.enrich_calls[0]["project_id"] is None


def test_workspace_comes_from_the_row():
    svc = Service(visibility="private", project_id=None, memory="the text", workspace="reference")
    _run(svc, visibility="private")
    assert svc.enrich_calls[0]["workspace"] == "reference"


def test_legacy_row_without_visibility_is_treated_as_private():
    svc = Service(visibility=None, project_id=None, memory="the text", workspace=None)
    _run(svc, visibility="shared")
    assert svc.enrich_calls[0]["visibility"] == "private"


def test_content_rewritten_while_queued_is_skipped_not_published():
    svc = Service(visibility="private", project_id=None, memory="rewritten text", workspace=None)
    out = _run(svc, content="the old text", visibility="private")
    assert out == {"memory_id": "m1", "enriched": False, "skipped": "content_changed"}
    assert svc.enrich_calls == []


def test_whitespace_only_difference_is_not_a_content_change():
    svc = Service(visibility="private", project_id=None, memory="the  text\n", workspace=None)
    out = _run(svc, content="the text", visibility="private")
    assert out["enriched"] is True


def test_unchanged_row_enriches_exactly_as_queued():
    svc = Service(visibility="shared", project_id="p", memory="the text", workspace=None)
    out = _run(svc, visibility="shared", project_id="p")
    assert out["enriched"] is True
    call = svc.enrich_calls[0]
    assert (call["visibility"], call["project_id"], call["user_id"]) == ("shared", "p", "alice")


# ── a change that lands DURING extraction ─────────────────────────────


class DuringService(Service):
    """get_memory returns the queued state first, then ``after`` (what the row became mid-extraction)."""

    def __init__(self, after, **row):
        super().__init__(**row)
        self.after = after
        self.reads = 0
        self.expired: list[tuple] = []

    def get_memory(self, memory_id):
        self.reads += 1
        return self.row if self.reads == 1 else self.after

    def _cascade_or_fallback_expire(self, mem, *, group_id=None, memory_id=""):
        self.expired.append((mem, group_id, memory_id))
        return {"resolved": True}


def _row_like(**kw):
    return SimpleNamespace(id="m1", occurred_at=None, **kw)


def test_memory_deleted_during_enrichment_has_its_episode_expired():
    svc = DuringService(None, visibility="private", project_id=None, memory="the text", workspace=None)
    out = _run(svc, visibility="private")
    assert out["skipped"] == "changed_during_enrichment" and out["enriched"] is False
    (mem, group, mid), = svc.expired
    assert group == "user--alice" and mid == "m1" and mem["memory"] == "the text"


def test_memory_narrowed_during_enrichment_has_the_shared_episode_expired():
    after = _row_like(visibility="private", project_id=None, memory="the text", workspace=None)
    svc = DuringService(after, visibility="shared", project_id=None, memory="the text", workspace=None)
    out = _run(svc, visibility="shared")
    assert out["skipped"] == "changed_during_enrichment"
    assert svc.expired[0][1] == "shared"          # the group the job wrote to, not the new one


def test_unchanged_row_after_enrichment_expires_nothing():
    after = _row_like(visibility="shared", project_id=None, memory="the text", workspace=None)
    svc = DuringService(after, visibility="shared", project_id=None, memory="the text", workspace=None)
    assert _run(svc, visibility="shared")["enriched"] is True
    assert svc.expired == []
