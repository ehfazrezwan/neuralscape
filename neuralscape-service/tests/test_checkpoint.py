"""Tests for the checkpoint batch save (roadmap C4).

Covers: per-item dedup verdicts (index-aligned, existing_id on hits), the
single-batch-job enqueue, the ≤25 item bound, the session-note shape
(task_context + meeting_outcome), the standard-tier write gate, the
storage-key derivation the verdicts must share with store_raw, and both
the REST route and MCP tool surfaces.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import checkpoint as checkpoint_mod
from checkpoint import (
    dedup_verdicts,
    effective_storage_key,
    prepare_checkpoint,
    render_session_note,
    session_note_item,
)
from schemas import CheckpointRequest, MemoryResponse, SessionNote


def _existing(mid: str = "existing-1") -> MemoryResponse:
    return MemoryResponse(id=mid, memory="dup", category="decision", source="vector")


# ──────────────────────────────────────────────
# Storage-key derivation (must mirror store_raw)
# ──────────────────────────────────────────────


class TestEffectiveStorageKey:
    def test_global_category_defaults_global(self):
        scope, vis, pid = effective_storage_key(
            {"content": "x", "category": "preference"}
        )
        assert (scope, vis, pid) == ("global", "private", None)

    def test_flexible_category_with_project_becomes_project(self):
        scope, vis, pid = effective_storage_key(
            {"content": "x", "category": "decision", "project_id": "p1"}
        )
        assert (scope, vis, pid) == ("project", "shared", "p1")

    def test_explicit_scope_wins(self):
        scope, _, _ = effective_storage_key(
            {"content": "x", "category": "decision", "scope": "global",
             "project_id": "p1"}
        )
        assert scope == "global"

    def test_standard_forces_global_no_project(self):
        scope, vis, pid = effective_storage_key(
            {"content": "x", "category": "convention", "project_id": "p1",
             "visibility": "standard"}
        )
        assert (scope, vis, pid) == ("global", "standard", None)


# ──────────────────────────────────────────────
# Dedup verdicts
# ──────────────────────────────────────────────


class TestDedupVerdicts:
    def test_verdicts_index_aligned_with_existing_ids(self):
        svc = MagicMock()
        svc._find_by_content_hash.side_effect = [None, _existing("dup-7"), None]
        items = [
            {"content": "a", "category": "decision", "user_id": "u"},
            {"content": "b", "category": "decision", "user_id": "u"},
            {"content": "c", "category": "decision", "user_id": "u"},
        ]
        verdicts = dedup_verdicts(svc, items)
        assert [v["verdict"] for v in verdicts] == ["new", "duplicate", "new"]
        assert [v["index"] for v in verdicts] == [0, 1, 2]
        assert verdicts[1]["existing_id"] == "dup-7"
        assert "existing_id" not in verdicts[0]

    def test_probe_uses_store_raw_dedup_key(self):
        """The pre-check must probe the same (user, hash, scope, visibility,
        project) key store_raw's dedup uses — or verdicts would lie."""
        import hashlib

        svc = MagicMock()
        svc._find_by_content_hash.return_value = None
        dedup_verdicts(svc, [{
            "content": "the fact", "category": "decision",
            "user_id": "alice", "project_id": "p1",
        }])
        kwargs = svc._find_by_content_hash.call_args.kwargs
        assert kwargs["user_id"] == "alice"
        assert kwargs["content_hash"] == hashlib.md5(b"the fact").hexdigest()
        assert kwargs["scope"] == "project"
        assert kwargs["project_id"] == "p1"
        assert kwargs["visibility"] == "shared"  # decision's category default


# ──────────────────────────────────────────────
# Session note
# ──────────────────────────────────────────────


class TestSessionNote:
    def test_render_includes_only_set_fields(self):
        text = render_session_note({
            "request": "Build the tiers", "learned": None,
            "completed": "Shipped C3", "next_steps": "  ",
        })
        assert "Request: Build the tiers" in text
        assert "Completed: Shipped C3" in text
        assert "Learned" not in text and "Next steps" not in text

    def test_item_shape(self):
        item = session_note_item(
            {"request": "r", "next_steps": "n"}, "alice", "proj-1"
        )
        assert item["category"] == "task_context"
        assert item["observation_type"] == "meeting_outcome"
        assert item["source_type"] == "explicit"
        assert item["tags"] == ["session_note"]
        assert item["user_id"] == "alice"
        assert item["scope"] == "project" and item["project_id"] == "proj-1"
        assert "Next steps: n" in item["content"]

    def test_item_without_project_is_global(self):
        item = session_note_item({"request": "r"}, "alice", None)
        assert item["scope"] == "global" and "project_id" not in item

    def test_schema_requires_at_least_one_field(self):
        with pytest.raises(ValueError):
            SessionNote()
        with pytest.raises(ValueError):
            SessionNote(request="   ")
        assert SessionNote(learned="x").learned == "x"

    def test_investigated_field_renders_in_narrative_order(self):
        # D2: the plugin's Stop summary sends all five structured fields.
        note = SessionNote(investigated="Read utils.ts and the hook manifest")
        assert note.investigated
        text = render_session_note({
            "request": "Fix the offset bug",
            "investigated": "Read utils.ts and the hook manifest",
            "next_steps": "Ship it",
        })
        assert "Investigated: Read utils.ts and the hook manifest" in text
        # Narrative order: Request before Investigated before Next steps.
        assert (
            text.index("Request:")
            < text.index("Investigated:")
            < text.index("Next steps:")
        )


# ──────────────────────────────────────────────
# prepare_checkpoint (shared REST/MCP core)
# ──────────────────────────────────────────────


class TestPrepareCheckpoint:
    def _svc(self, side_effect=None):
        svc = MagicMock()
        if side_effect is not None:
            svc._find_by_content_hash.side_effect = side_effect
        else:
            svc._find_by_content_hash.return_value = None
        return svc

    def test_dupes_excluded_from_enqueue_note_appended(self):
        svc = self._svc(side_effect=[None, _existing()])
        req = CheckpointRequest(
            memories=[
                {"content": "new fact", "category": "decision"},
                {"content": "dup fact", "category": "decision"},
            ],
            session_note={"request": "do things"},
            project_id="p1",
        )
        out = prepare_checkpoint(svc, req, "alice")
        assert [v["verdict"] for v in out["verdicts"]] == ["new", "duplicate"]
        assert out["duplicates"] == 1
        assert out["session_note_included"] is True
        contents = [d["content"] for d in out["to_enqueue"]]
        assert contents[0] == "new fact"
        assert contents[-1].startswith("Session note:")
        assert len(out["to_enqueue"]) == 2  # 1 new + note (dupe dropped)
        assert all(d["user_id"] == "alice" for d in out["to_enqueue"])

    def test_all_dupes_no_note_enqueues_nothing(self):
        svc = self._svc(side_effect=[_existing("e1"), _existing("e2")])
        req = CheckpointRequest(memories=[
            {"content": "a", "category": "decision"},
            {"content": "b", "category": "decision"},
        ])
        out = prepare_checkpoint(svc, req, "alice")
        assert out["to_enqueue"] == []
        assert out["duplicates"] == 2

    def test_item_user_mismatch_rejected(self):
        req = CheckpointRequest(memories=[
            {"content": "a", "category": "decision", "user_id": "mallory"},
        ])
        with pytest.raises(ValueError, match="does not match the caller"):
            prepare_checkpoint(self._svc(), req, "alice")

    def test_standard_gate_non_dictator_rejected(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "standards_enabled", True)
        monkeypatch.setattr(settings, "dictator_user_ids", "mark")
        req = CheckpointRequest(memories=[
            {"content": "a", "category": "convention", "visibility": "standard"},
        ])
        with pytest.raises(PermissionError, match="not authorized"):
            prepare_checkpoint(self._svc(), req, "alice")

    def test_standard_gate_dictator_allowed(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "standards_enabled", True)
        monkeypatch.setattr(settings, "dictator_user_ids", "mark")
        req = CheckpointRequest(memories=[
            {"content": "a", "category": "convention", "visibility": "standard"},
        ])
        out = prepare_checkpoint(self._svc(), req, "mark")
        assert out["to_enqueue"][0]["scope"] == "global"

    def test_expires_at_serialized_for_enqueue(self):
        req = CheckpointRequest(memories=[
            {"content": "a", "category": "task_context",
             "expires_at": "2026-12-01T00:00:00+00:00"},
        ])
        out = prepare_checkpoint(self._svc(), req, "alice")
        assert isinstance(out["to_enqueue"][0]["expires_at"], str)

    def test_derived_scope_stamped_on_payload(self):
        """The enqueued payload must carry the derived scope so the worker
        writes exactly the key the verdict probed."""
        req = CheckpointRequest(memories=[
            {"content": "a", "category": "decision", "project_id": "p1"},
        ])
        out = prepare_checkpoint(self._svc(), req, "alice")
        assert out["to_enqueue"][0]["scope"] == "project"


# ──────────────────────────────────────────────
# Schema bounds
# ──────────────────────────────────────────────


class TestCheckpointSchema:
    def test_over_25_items_rejected(self):
        with pytest.raises(ValueError):
            CheckpointRequest(memories=[
                {"content": f"m{i}", "category": "decision"} for i in range(26)
            ])

    def test_25_items_accepted(self):
        req = CheckpointRequest(memories=[
            {"content": f"m{i}", "category": "decision"} for i in range(25)
        ])
        assert len(req.memories) == 25

    def test_empty_checkpoint_rejected(self):
        with pytest.raises(ValueError, match="at least one memory or a session_note"):
            CheckpointRequest()

    def test_note_only_checkpoint_valid(self):
        req = CheckpointRequest(session_note={"completed": "shipped"})
        assert req.memories == []


# ──────────────────────────────────────────────
# REST route
# ──────────────────────────────────────────────


class TestCheckpointRoute:
    @pytest.fixture()
    def client(self, monkeypatch):
        import main

        svc = MagicMock()
        svc._find_by_content_hash.return_value = None
        monkeypatch.setattr(main, "_service", svc)
        tm = MagicMock()
        tm.enqueue_raw_batch = AsyncMock(return_value="task-cp-1")
        monkeypatch.setattr(main, "_task_manager", tm)
        client = TestClient(main.app, raise_server_exceptions=False)
        client._svc, client._tm = svc, tm
        return client

    def test_batch_enqueued_as_single_job_202(self, client):
        resp = client.post("/v1/checkpoint", json={
            "user_id": "alice",
            "memories": [
                {"content": "fact one", "category": "decision"},
                {"content": "fact two", "category": "decision"},
            ],
            "session_note": {"request": "r", "next_steps": "n"},
        })
        assert resp.status_code == 202
        body = resp.json()
        assert body["task_id"] == "task-cp-1"
        assert body["poll_url"].endswith("task-cp-1")
        assert body["enqueued"] == 3  # 2 new + session note
        assert body["session_note_included"] is True
        # ONE batch job for everything:
        client._tm.enqueue_raw_batch.assert_awaited_once()
        items = client._tm.enqueue_raw_batch.await_args.kwargs["items"]
        assert len(items) == 3

    def test_all_dupes_returns_200_null_task(self, client):
        client._svc._find_by_content_hash.return_value = _existing("old-1")
        resp = client.post("/v1/checkpoint", json={
            "user_id": "alice",
            "memories": [{"content": "dup", "category": "decision"}],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "task_id" not in body or body["task_id"] is None
        assert body["verdicts"][0]["verdict"] == "duplicate"
        assert body["verdicts"][0]["existing_id"] == "old-1"
        client._tm.enqueue_raw_batch.assert_not_awaited()

    def test_over_25_items_422(self, client):
        resp = client.post("/v1/checkpoint", json={
            "user_id": "alice",
            "memories": [
                {"content": f"m{i}", "category": "decision"} for i in range(26)
            ],
        })
        assert resp.status_code == 422

    def test_standard_gate_403(self, client, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "standards_enabled", True)
        monkeypatch.setattr(settings, "dictator_user_ids", "mark")
        resp = client.post("/v1/checkpoint", json={
            "user_id": "alice",
            "memories": [
                {"content": "rule", "category": "convention", "visibility": "standard"},
            ],
        })
        assert resp.status_code == 403
        client._tm.enqueue_raw_batch.assert_not_awaited()

    def test_redis_down_falls_back_to_sync(self, client):
        client._tm.enqueue_raw_batch = AsyncMock(side_effect=ConnectionError("down"))
        client._svc.store_raw_batch.return_value = []
        resp = client.post("/v1/checkpoint", json={
            "user_id": "alice",
            "memories": [{"content": "fact", "category": "decision"}],
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        client._svc.store_raw_batch.assert_called_once()


# ──────────────────────────────────────────────
# MCP tool
# ──────────────────────────────────────────────


class TestCheckpointMcpTool:
    @pytest.fixture(autouse=True)
    def _wire(self, monkeypatch):
        import mcp_server

        self.svc = MagicMock()
        self.svc._find_by_content_hash.return_value = None
        monkeypatch.setattr(mcp_server, "_service", self.svc)
        self.tm = MagicMock()
        self.tm.enqueue_raw_batch = AsyncMock(return_value="task-cp-mcp")
        monkeypatch.setattr(mcp_server, "_task_manager", self.tm)

    @pytest.mark.asyncio
    async def test_single_card_response_with_verdicts(self):
        import mcp_server

        self.svc._find_by_content_hash.side_effect = [None, _existing("dup-9")]
        result = await mcp_server.call_tool("checkpoint", {
            "user_id": "alice",
            "memories": [
                {"content": "new", "category": "decision"},
                {"content": "dup", "category": "decision"},
            ],
            "session_note": {"learned": "a lot"},
        })
        assert len(result) == 1  # one tool card
        data = json.loads(result[0].text)
        assert data["status"] == "accepted"
        assert data["task_id"] == "task-cp-mcp"
        assert [v["verdict"] for v in data["verdicts"]] == ["new", "duplicate"]
        assert data["verdicts"][1]["existing_id"] == "dup-9"
        assert data["session_note_included"] is True
        self.tm.enqueue_raw_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_over_25_rejected_before_enqueue(self):
        import mcp_server

        result = await mcp_server.call_tool("checkpoint", {
            "user_id": "alice",
            "memories": [
                {"content": f"m{i}", "category": "decision"} for i in range(26)
            ],
        })
        data = json.loads(result[0].text)
        assert data["status"] == "error"
        self.tm.enqueue_raw_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_dupes_returns_ok_without_enqueue(self):
        import mcp_server

        self.svc._find_by_content_hash.return_value = _existing()
        result = await mcp_server.call_tool("checkpoint", {
            "user_id": "alice",
            "memories": [{"content": "dup", "category": "decision"}],
        })
        data = json.loads(result[0].text)
        assert data["status"] == "ok" and data["task_id"] is None
        self.tm.enqueue_raw_batch.assert_not_awaited()
