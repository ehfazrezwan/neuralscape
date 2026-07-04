"""E1 — live memory-event stream: Redis pub/sub publishers + the SSE generator.

Real-time feed of memory events for GET /v1/stream: memory_stored, dream
actions applied, insights stored, checkpoint batches. Publishers are tiny
fire-and-forget hooks at the existing emission points (the extension
registry's memory_stored fan-out, the dreaming sweep's apply/store-insights
steps, the checkpoint endpoint); a publish failure never breaks a write.

Visibility contract — enforced at PUBLISH time, re-checked at SUBSCRIBE time:

- **Publish side (authoritative):** :func:`channel_for` routes each event by
  its visibility. Private events are published ONLY to their owner's channel
  ``ns:events:{user_id}``; shared/standard events go to the single shared
  channel ``ns:events:shared``. A private event with no resolvable owner is
  dropped, never broadcast.
- **Subscribe side (defense in depth):** the SSE generator re-runs
  :func:`visible_to` on every delivered message, so even a mis-published
  private event never reaches another user's stream.

The SSE endpoint subscribes to exactly two channels — the caller's own and
the shared one — emits a ``: keep-alive`` comment roughly every 20 s, and is
client-disconnect safe (the endpoint polls ``request.is_disconnected`` and
tears the pub/sub connection down in a ``finally``).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

CHANNEL_PREFIX = "ns:events:"
SHARED_CHANNEL = "ns:events:shared"
HEARTBEAT_SECONDS = 20.0

# Events stay lean: memory content is truncated to a snippet — the stream is
# a feed, not a payload channel (full payloads come from get_memories).
CONTENT_SNIPPET_CHARS = 240

# Payload keys worth streaming (identity, routing, and light context).
_EVENT_KEYS = (
    "user_id",
    "owner_user_id",
    "visibility",
    "memory_id",
    "memory_ids",
    "category",
    "scope",
    "project_id",
    "pool",
    "run_id",
    "source",
    "task_id",
    "applied",
    "reported",
    "action_types",
    "count",
    "enqueued",
    "duplicates",
)

_redis = None


def _get_redis():
    """Lazy module-level sync Redis client (publish side only)."""
    global _redis
    if _redis is None:
        import redis as redis_lib

        from config import settings

        _redis = redis_lib.Redis.from_url(
            settings.redis_url, socket_timeout=2, socket_connect_timeout=2
        )
    return _redis


# ── Visibility routing/filtering ────────────────────────────────────


def _owner_of(event: dict) -> str | None:
    return event.get("owner_user_id") or event.get("user_id") or None


def channel_for(event: dict) -> str | None:
    """Publish-time visibility enforcement: pick the one channel this event
    may appear on. Private → owner's channel only; shared/standard → the
    shared channel; unroutable private (no owner) → None (drop)."""
    visibility = (event.get("visibility") or "private").lower()
    if visibility in ("shared", "standard"):
        return SHARED_CHANNEL
    owner = _owner_of(event)
    if not owner:
        return None
    return f"{CHANNEL_PREFIX}{owner}"


def visible_to(event: dict, caller_user_id: str) -> bool:
    """Subscribe-time re-check: may this caller see this event?

    Shared/standard events are team-visible; private events only ever to
    their owner. Ownerless private events are visible to no one.
    """
    visibility = (event.get("visibility") or "private").lower()
    if visibility in ("shared", "standard"):
        return True
    owner = _owner_of(event)
    return bool(owner) and owner == caller_user_id


# ── Publish side ────────────────────────────────────────────────────


def build_event(event_type: str, payload: dict) -> dict:
    """Distill an emission payload into a lean, JSON-safe stream event."""
    event: dict = {"type": event_type, "ts": datetime.now(timezone.utc).isoformat()}
    for key in _EVENT_KEYS:
        value = payload.get(key)
        if value is not None:
            event[key] = value
    content = payload.get("content")
    if isinstance(content, str) and content:
        event["content"] = content[:CONTENT_SNIPPET_CHARS]
    return event


def publish_event_bg(event_type: str, payload: dict) -> None:
    """Publish without ever touching the caller's thread/event loop.

    Audit 27 #11: ``publish_event`` does sync Redis I/O (2s timeouts) — on
    the API event loop or the worker's per-fact fan-out that stalls real
    work. This variant hands the publish to the shared telemetry executor
    (bounded, single worker, drop-on-overflow) and returns immediately.
    Never raises.
    """
    try:
        import telemetry

        telemetry.submit(publish_event, event_type, payload)
    except Exception:
        logger.debug("event-stream bg publish dispatch failed (non-fatal)", exc_info=True)


def publish_event(event_type: str, payload: dict) -> bool:
    """Fire-and-forget publish onto the visibility-routed channel.

    Synchronous Redis I/O — call :func:`publish_event_bg` from any latency-
    sensitive path (API routes, worker fan-outs).

    Never raises — a down Redis or a serialization hiccup is logged at
    debug and swallowed (the stream is an observability surface, not a
    dependency of the write path). Returns True when a message was sent.
    """
    try:
        from config import settings

        if not settings.event_stream_enabled:
            return False
        event = build_event(event_type, payload)
        channel = channel_for(event)
        if channel is None:
            return False
        _get_redis().publish(channel, json.dumps(event, default=str))
        return True
    except Exception:
        logger.debug("event-stream publish failed (non-fatal)", exc_info=True)
        return False


# ── Subscribe side (SSE) ────────────────────────────────────────────


def format_sse(event: dict) -> str:
    """One SSE frame: `event:` = our event type, `data:` = the JSON body."""
    body = json.dumps(event, default=str, ensure_ascii=False)
    return f"event: {event.get('type') or 'message'}\ndata: {body}\n\n"


async def sse_event_stream(
    pubsub,
    caller_user_id: str,
    is_disconnected,
    *,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    poll_timeout: float = 1.0,
):
    """Async generator of SSE frames from an already-subscribed pub/sub.

    ``pubsub`` needs one async method: ``get_message(ignore_subscribe_messages,
    timeout)`` (redis.asyncio's PubSub — or a fake in tests). ``is_disconnected``
    is an async callable polled every loop so a vanished client tears the
    stream down within ~``poll_timeout`` seconds. Every delivered message is
    re-filtered with :func:`visible_to` before it reaches the client.
    """
    yield ": connected\n\n"
    last_beat = time.monotonic()
    while True:
        if await is_disconnected():
            return
        try:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=poll_timeout
            )
        except Exception:
            logger.debug("event-stream pubsub read failed; closing", exc_info=True)
            return
        if message is not None and message.get("type") == "message":
            event = _parse_message(message)
            if event is not None and visible_to(event, caller_user_id):
                yield format_sse(event)
        now = time.monotonic()
        if now - last_beat >= heartbeat_seconds:
            yield ": keep-alive\n\n"
            last_beat = now


def _parse_message(message: dict) -> dict | None:
    data = message.get("data")
    if isinstance(data, (bytes, bytearray)):
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        event = json.loads(data)
    except (TypeError, ValueError):
        return None
    return event if isinstance(event, dict) else None
