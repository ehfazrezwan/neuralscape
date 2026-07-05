"""Solo engine unit 5: in-process scheduler, SSE fan-out, extraction-settings
local store (docs/neuralscape/28-solo-engine.md §5.3 / §5.5).

Team-mode paths (Redis pub/sub, Redis settings keys, ARQ crons) are untouched;
everything here exercises the solo branches.
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest


# ── Scheduler ────────────────────────────────────────────────────────


class TestSchedulerDue:
    def _specs(self):
        return [
            ("expire", lambda: {3}, 15),
            ("dedup", lambda: {0, 6, 12, 18}, 0),
            ("compile", lambda: {18, 19, 20}, 30),
        ]

    def test_fires_inside_window_once_per_hour(self):
        from scheduler import _due

        fired: set = set()
        now = datetime(2026, 7, 6, 3, 15, tzinfo=timezone.utc)
        assert _due(now, self._specs(), fired) == ["expire"]
        # same hour, next poll — already fired
        now2 = datetime(2026, 7, 6, 3, 16, tzinfo=timezone.utc)
        assert _due(now2, self._specs(), fired) == []
        # next day, same hour — fires again
        now3 = datetime(2026, 7, 7, 3, 15, tzinfo=timezone.utc)
        assert _due(now3, self._specs(), fired) == ["expire"]

    def test_outside_hour_or_minute_window_skips(self):
        from scheduler import _due

        fired: set = set()
        assert _due(datetime(2026, 7, 6, 4, 15, tzinfo=timezone.utc), self._specs(), fired) == []
        assert _due(datetime(2026, 7, 6, 3, 18, tzinfo=timezone.utc), self._specs(), fired) == []

    def test_bad_hours_provider_skips_not_crashes(self):
        from scheduler import _due

        def boom():
            raise RuntimeError("no hours")

        fired: set = set()
        specs = [("broken", boom, 0), ("dedup", lambda: {5}, 0)]
        now = datetime(2026, 7, 6, 5, 0, tzinfo=timezone.utc)
        assert _due(now, specs, fired) == ["dedup"]

    def test_fire_runs_cron_under_slow_semaphore(self):
        from scheduler import _fire

        async def scenario():
            ran = {}

            async def cron(ctx):
                ran["ctx"] = ctx
                return {"ok": True}

            runner = SimpleNamespace(
                _worker=SimpleNamespace(mycron=cron),
                _sems={"slow": asyncio.Semaphore(1)},
                ctx={"service": "svc"},
            )
            await _fire(runner, "mycron")
            return ran

        ran = asyncio.run(scenario())
        assert ran["ctx"] == {"service": "svc"}

    def test_cron_specs_resolve_against_worker(self):
        """Every scheduled name must exist in worker.py (drift guard)."""
        import worker
        from scheduler import _cron_specs

        for name, hours, minute in _cron_specs():
            assert callable(getattr(worker, name)), name
            assert isinstance(set(hours()), set)
            assert 0 <= minute < 60


# ── SSE in-process fan-out ──────────────────────────────────────────


class TestInProcEventStream:
    @pytest.fixture(autouse=True)
    def _solo_mode(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "ns_mode", "solo")
        monkeypatch.setattr(settings, "event_stream_enabled", True)

    def test_publish_reaches_subscriber_and_respects_channels(self):
        import event_stream as es

        async def scenario():
            sub = es.InProcPubSub([es.SHARED_CHANNEL, f"{es.CHANNEL_PREFIX}alice"])
            es.publish_event(
                "memory_stored",
                {"user_id": "alice", "visibility": "private", "memory_id": "m1"},
            )
            es.publish_event(
                "memory_stored",
                {"user_id": "bob", "visibility": "private", "memory_id": "m2"},
            )
            got = []
            while True:
                msg = await sub.get_message(timeout=0.2)
                if msg is None:
                    break
                got.append(json.loads(msg["data"]))
            await sub.aclose()
            return got

        got = asyncio.run(scenario())
        # bob's private event routed to bob's channel only — never alice's
        assert [e["memory_id"] for e in got] == ["m1"]

    def test_shared_visibility_lands_on_shared_channel(self):
        import event_stream as es

        async def scenario():
            sub = es.InProcPubSub([es.SHARED_CHANNEL])
            es.publish_event(
                "memory_stored",
                {"user_id": "alice", "visibility": "shared", "memory_id": "m3"},
            )
            msg = await sub.get_message(timeout=0.5)
            await sub.aclose()
            return msg

        msg = asyncio.run(scenario())
        assert msg and json.loads(msg["data"])["memory_id"] == "m3"

    def test_cross_thread_publish_delivers(self):
        import event_stream as es

        async def scenario():
            sub = es.InProcPubSub([es.SHARED_CHANNEL])
            await asyncio.to_thread(
                es.publish_event,
                "checkpoint_saved",
                {"user_id": "alice", "visibility": "shared", "count": 3},
            )
            msg = await sub.get_message(timeout=1.0)
            await sub.aclose()
            return msg

        msg = asyncio.run(scenario())
        assert msg and json.loads(msg["data"])["type"] == "checkpoint_saved"

    def test_sse_generator_streams_from_inproc_pubsub(self):
        import event_stream as es

        async def scenario():
            sub = es.InProcPubSub([f"{es.CHANNEL_PREFIX}alice"])
            es.publish_event(
                "memory_stored",
                {"user_id": "alice", "visibility": "private", "memory_id": "m4"},
            )

            async def never_disconnected():
                return False

            frames = []
            gen = es.sse_event_stream(sub, "alice", never_disconnected, poll_timeout=0.2)
            async for frame in gen:
                frames.append(frame)
                if len(frames) >= 2:  # ": connected" + the event
                    break
            await gen.aclose()
            await sub.aclose()
            return frames

        frames = asyncio.run(scenario())
        assert frames[0].startswith(": connected")
        assert "m4" in frames[1] and frames[1].startswith("event: memory_stored")

    def test_unsubscribe_stops_delivery(self):
        import event_stream as es

        async def scenario():
            sub = es.InProcPubSub([es.SHARED_CHANNEL])
            await sub.unsubscribe()
            es.publish_event(
                "memory_stored", {"user_id": "x", "visibility": "shared"}
            )
            return await sub.get_message(timeout=0.2)

        assert asyncio.run(scenario()) is None


# ── Extraction settings local store ──────────────────────────────────


class TestExtractionSettingsLocalStore:
    @pytest.fixture(autouse=True)
    def _solo_store(self, monkeypatch, tmp_path):
        import extraction_settings as xs
        from config import settings

        monkeypatch.setattr(settings, "ns_mode", "solo")
        monkeypatch.setattr(settings, "extraction_instructions_enabled", True)
        monkeypatch.setattr(
            settings, "extraction_settings_path", str(tmp_path / "xs.json")
        )
        monkeypatch.setattr(xs, "_local_store", None)  # fresh store per test
        yield

    def test_set_get_resolve_roundtrip(self):
        import extraction_settings as xs

        rec = xs.set_instructions(
            user_id="solo", instructions="Prefer terse facts.", updated_by="solo"
        )
        assert rec["tokens"] > 0
        got = xs.get_instructions(user_id="solo")
        assert got["instructions"] == "Prefer terse facts."
        resolved = xs.resolve_instructions("solo", None)
        assert "Prefer terse facts." in resolved

    def test_clear_deletes_record_and_survives_restart(self, tmp_path):
        import extraction_settings as xs

        xs.set_instructions(user_id="solo", instructions="A rule.", updated_by="solo")
        # a "restarted daemon" — fresh store object over the same file
        xs._local_store = None
        assert xs.get_instructions(user_id="solo")["instructions"] == "A rule."
        xs.set_instructions(user_id="solo", instructions="", updated_by="solo")
        assert xs.get_instructions(user_id="solo") is None

    def test_project_and_user_scopes_compose(self):
        import extraction_settings as xs

        xs.set_instructions(project_id="proj", instructions="Project rule.", updated_by="solo")
        xs.set_instructions(user_id="solo", instructions="User rule.", updated_by="solo")
        resolved = xs.resolve_instructions("solo", "proj")
        assert resolved.index("Project rule.") < resolved.index("User rule.")
