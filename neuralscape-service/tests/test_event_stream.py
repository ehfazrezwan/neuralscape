"""Unit tests for E1 — the SSE live event stream (event_stream.py).

Covers the two visibility layers explicitly:

- **publish side** (authoritative): channel_for routes private events only
  to their owner's channel, shared/standard to the shared channel, and
  drops ownerless private events;
- **subscribe side** (defense in depth): visible_to + an integration-ish
  run of the SSE generator against a fake pubsub proving a mis-published
  private event never reaches another user's stream.

Plus: publish_event failure-safety and kill-switch, payload distillation,
heartbeats, disconnect teardown, and the extension-registry mirror hook.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import event_stream
from config import settings
from event_stream import (
    CHANNEL_PREFIX,
    SHARED_CHANNEL,
    build_event,
    channel_for,
    format_sse,
    publish_event,
    sse_event_stream,
    visible_to,
)


# ── Publish-time routing (the authoritative enforcement) ─────────────


class TestChannelFor:
    def test_private_routes_to_owner_channel_only(self):
        assert channel_for({"visibility": "private", "user_id": "alice"}) == f"{CHANNEL_PREFIX}alice"

    def test_owner_user_id_wins_over_user_id(self):
        event = {"visibility": "private", "owner_user_id": "owner", "user_id": "writer"}
        assert channel_for(event) == f"{CHANNEL_PREFIX}owner"

    def test_shared_routes_to_shared_channel(self):
        assert channel_for({"visibility": "shared", "user_id": "alice"}) == SHARED_CHANNEL

    def test_standard_routes_to_shared_channel(self):
        assert channel_for({"visibility": "standard"}) == SHARED_CHANNEL

    def test_missing_visibility_defaults_private(self):
        assert channel_for({"user_id": "bob"}) == f"{CHANNEL_PREFIX}bob"

    def test_ownerless_private_is_dropped(self):
        assert channel_for({"visibility": "private"}) is None


class TestVisibleTo:
    def test_own_private_event_visible(self):
        assert visible_to({"visibility": "private", "user_id": "alice"}, "alice")

    def test_other_users_private_event_never_visible(self):
        assert not visible_to({"visibility": "private", "user_id": "alice"}, "bob")

    def test_shared_visible_to_everyone(self):
        assert visible_to({"visibility": "shared", "user_id": "alice"}, "bob")

    def test_standard_visible_to_everyone(self):
        assert visible_to({"visibility": "standard"}, "anyone")

    def test_ownerless_private_visible_to_no_one(self):
        assert not visible_to({"visibility": "private"}, "alice")


# ── Publish side ─────────────────────────────────────────────────────


class FakeRedisPub:
    def __init__(self):
        self.published: list[tuple[str, str]] = []

    def publish(self, channel, message):
        self.published.append((channel, message))
        return 1


class TestPublishEvent:
    @pytest.fixture(autouse=True)
    def fake_redis(self, monkeypatch):
        fake = FakeRedisPub()
        monkeypatch.setattr(event_stream, "_get_redis", lambda: fake)
        return fake

    def test_private_event_published_to_owner_channel(self, fake_redis):
        ok = publish_event("memory_stored", {
            "user_id": "alice", "visibility": "private",
            "memory_id": "m1", "content": "secret fact",
        })
        assert ok
        [(channel, message)] = fake_redis.published
        assert channel == f"{CHANNEL_PREFIX}alice"
        event = json.loads(message)
        assert event["type"] == "memory_stored"
        assert event["memory_id"] == "m1"
        assert "ts" in event

    def test_shared_event_published_to_shared_channel(self, fake_redis):
        publish_event("memory_stored", {"user_id": "alice", "visibility": "shared"})
        assert fake_redis.published[0][0] == SHARED_CHANNEL

    def test_ownerless_private_never_published(self, fake_redis):
        assert not publish_event("memory_stored", {"visibility": "private"})
        assert fake_redis.published == []

    def test_kill_switch(self, fake_redis):
        saved = settings.event_stream_enabled
        settings.event_stream_enabled = False
        try:
            assert not publish_event("memory_stored", {"user_id": "a"})
            assert fake_redis.published == []
        finally:
            settings.event_stream_enabled = saved

    def test_redis_failure_swallowed(self, monkeypatch):
        def boom():
            raise ConnectionError("redis down")

        monkeypatch.setattr(event_stream, "_get_redis", boom)
        assert publish_event("memory_stored", {"user_id": "a"}) is False  # no raise

    def test_content_truncated_to_snippet(self, fake_redis):
        publish_event("memory_stored", {"user_id": "a", "content": "x" * 10_000})
        event = json.loads(fake_redis.published[0][1])
        assert len(event["content"]) == event_stream.CONTENT_SNIPPET_CHARS


class TestBuildEvent:
    def test_keeps_known_keys_drops_unknown_and_none(self):
        event = build_event("memory_stored", {
            "user_id": "u", "memory_id": "m", "category": None,
            "internal_gunk": {"big": "blob"},
        })
        assert event["user_id"] == "u"
        assert event["memory_id"] == "m"
        assert "category" not in event
        assert "internal_gunk" not in event
        assert event["type"] == "memory_stored"


# ── Subscribe side: SSE generator with a fake pubsub ─────────────────


class FakePubSub:
    """Delivers queued pub/sub messages, then None forever."""

    def __init__(self, events: list[dict]):
        self._messages = [
            {"type": "message", "data": json.dumps(e).encode()} for e in events
        ]

    async def get_message(self, ignore_subscribe_messages=True, timeout=None):
        if self._messages:
            return self._messages.pop(0)
        return None


async def _collect_frames(pubsub, caller, *, loops: int, heartbeat_seconds=3600.0):
    """Run the generator for a bounded number of poll loops."""
    remaining = loops

    async def is_disconnected():
        nonlocal remaining
        remaining -= 1
        return remaining < 0

    frames = []
    async for frame in sse_event_stream(
        pubsub, caller, is_disconnected,
        heartbeat_seconds=heartbeat_seconds, poll_timeout=0,
    ):
        frames.append(frame)
    return frames


class TestSseEventStream:
    @pytest.mark.asyncio
    async def test_own_and_shared_delivered_foreign_private_filtered(self):
        """Integration-ish: even a mis-published private event (already on
        the wire) is dropped by the subscribe-side re-check."""
        pubsub = FakePubSub([
            {"type": "memory_stored", "visibility": "private", "user_id": "alice", "memory_id": "own"},
            {"type": "memory_stored", "visibility": "private", "user_id": "mallory", "memory_id": "leak"},
            {"type": "insights_stored", "visibility": "shared", "count": 2},
        ])
        frames = await _collect_frames(pubsub, "alice", loops=5)
        joined = "".join(frames)
        assert '"memory_id": "own"' in joined
        assert "leak" not in joined
        assert "insights_stored" in joined

    @pytest.mark.asyncio
    async def test_connected_comment_first(self):
        frames = await _collect_frames(FakePubSub([]), "alice", loops=1)
        assert frames[0] == ": connected\n\n"

    @pytest.mark.asyncio
    async def test_heartbeat_comment_emitted(self):
        frames = await _collect_frames(
            FakePubSub([]), "alice", loops=3, heartbeat_seconds=0.0
        )
        assert ": keep-alive\n\n" in frames

    @pytest.mark.asyncio
    async def test_disconnect_stops_generator(self):
        async def gone():
            return True

        frames = []
        async for frame in sse_event_stream(FakePubSub([]), "a", gone, poll_timeout=0):
            frames.append(frame)
        # only the initial connected comment; loop exits on first poll
        assert frames == [": connected\n\n"]

    @pytest.mark.asyncio
    async def test_garbage_payload_skipped(self):
        pubsub = FakePubSub([])
        pubsub._messages = [{"type": "message", "data": b"not json"}]
        frames = await _collect_frames(pubsub, "alice", loops=3)
        assert all(f.startswith(":") for f in frames)  # comments only

    def test_format_sse_frame_shape(self):
        frame = format_sse({"type": "memory_stored", "memory_id": "m1"})
        assert frame.startswith("event: memory_stored\ndata: ")
        assert frame.endswith("\n\n")


# ── The registry mirror hook (worker emission point) ─────────────────


class TestRegistryMirror:
    @pytest.mark.asyncio
    async def test_memory_stored_emission_mirrors_to_stream(self):
        from extensions import ExtensionRegistry

        registry = ExtensionRegistry()
        payload = {"user_id": "alice", "memory_id": "m1", "visibility": "private"}
        with patch("event_stream.publish_event") as pub:
            await registry.emit_event("memory_stored", payload)
            # Audit 27 #11: the mirror is dispatched via the telemetry
            # executor (publish_event_bg), not called inline on the worker
            # loop — drain it before asserting.
            import telemetry

            telemetry.flush()
        pub.assert_called_once_with("memory_stored", payload)

    @pytest.mark.asyncio
    async def test_other_events_not_mirrored(self):
        from extensions import ExtensionRegistry

        registry = ExtensionRegistry()
        with patch("event_stream.publish_event") as pub:
            await registry.emit_event("session_start", {"user_id": "a", "session_id": "s"})
        pub.assert_not_called()

    @pytest.mark.asyncio
    async def test_mirror_failure_never_breaks_emit(self):
        from extensions import ExtensionRegistry

        registry = ExtensionRegistry()
        with patch("event_stream.publish_event", side_effect=RuntimeError("boom")):
            result = await registry.emit_event("memory_stored", {"user_id": "a"})
        assert result.notified_count == 0  # emit completed despite mirror failure
