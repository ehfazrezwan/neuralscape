"""Test ingest runner with mocked NS client."""

import json

import httpx
import pytest

from neuralscape_bench.client import NeuralscapeClient
from neuralscape_bench.accuracy.manifest import IngestManifest
from neuralscape_bench.accuracy.schema import Conversation, Session, SuiteData, Turn

from trackb.mem0_locomo.ingest import (
    ingest_conversation,
    ingest_suite,
    session_messages,
    trackb_user_id,
)


def test_trackb_user_id():
    """Test user ID generation for Track B."""
    assert trackb_user_id("conv1") == "trackb-mem0-locomo-conv1"
    assert trackb_user_id("sample_42") == "trackb-mem0-locomo-sample_42"


def test_session_messages_basic():
    """Test session to messages conversion."""
    session = Session(
        session_id="s1",
        turns=(
            Turn(role="user", content="Alice: Hello"),
            Turn(role="assistant", content="Bob: Hi"),
        ),
    )

    messages = session_messages(session)

    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "Alice: Hello"}
    assert messages[1] == {"role": "assistant", "content": "Bob: Hi"}


def test_session_messages_with_date():
    """Test date folding into first message."""
    session = Session(
        session_id="s1",
        date="2023-05-01",
        turns=(
            Turn(role="user", content="Alice: Hello"),
            Turn(role="assistant", content="Bob: Hi"),
        ),
    )

    messages = session_messages(session)

    assert "[This conversation session took place 2023-05-01.]" in messages[0]["content"]
    assert "Alice: Hello" in messages[0]["content"]
    assert messages[1]["content"] == "Bob: Hi"


class _FakeNS:
    """Minimal NS API: 202 writes, one-poll completion."""

    def __init__(self):
        self.extract_calls: list[dict] = []
        self._tasks = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path

        if path == "/v1/memories" and request.method == "POST":
            body = json.loads(request.content)
            self.extract_calls.append(body)
            self._tasks += 1
            return httpx.Response(202, json={"task_id": f"t{self._tasks}"})

        if path.startswith("/v1/memories/status/"):
            return httpx.Response(200, json={
                "task_id": path.rsplit("/", 1)[-1],
                "status": "completed",
            })

        return httpx.Response(404, json={"detail": "not found"})


@pytest.mark.asyncio
async def test_ingest_conversation_basic(tmp_path):
    """Test ingesting one conversation."""
    fake = _FakeNS()
    client = NeuralscapeClient(
        "http://test",
        http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler),
            base_url="http://test",
        ),
    )

    conv = Conversation(
        conv_id="c1",
        sessions=(
            Session(session_id="s1", turns=(
                Turn(role="user", content="Alice: Hi"),
            )),
            Session(session_id="s2", turns=(
                Turn(role="user", content="Alice: Bye"),
            )),
        ),
    )

    manifest = IngestManifest(tmp_path / "manifest.json")

    result = await ingest_conversation(
        client, conv,
        manifest=manifest,
        poll_timeout_s=5,
        poll_interval_s=0.01,
        log=lambda *_: None,
    )

    assert result["stored"] == 2
    assert result["skipped"] == 0
    assert result["failed"] == 0

    # Check extract calls
    assert len(fake.extract_calls) == 2
    assert fake.extract_calls[0]["user_id"] == "trackb-mem0-locomo-c1"
    assert fake.extract_calls[0]["run_id"] == "s1"
    assert fake.extract_calls[1]["run_id"] == "s2"

    await client.aclose()


@pytest.mark.asyncio
async def test_ingest_suite_with_concurrency(tmp_path):
    """Test ingesting multiple conversations with concurrency."""
    fake = _FakeNS()
    client = NeuralscapeClient(
        "http://test",
        http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler),
            base_url="http://test",
        ),
    )

    data = SuiteData(
        suite="test",
        conversations=[
            Conversation(conv_id="c1", sessions=(
                Session(session_id="s1", turns=(Turn(role="user", content="A: Hi"),)),
            )),
            Conversation(conv_id="c2", sessions=(
                Session(session_id="s1", turns=(Turn(role="user", content="B: Hello"),)),
            )),
        ],
    )

    manifest = IngestManifest(tmp_path / "manifest.json")

    summary = await ingest_suite(
        client, data,
        manifest=manifest,
        concurrency=2,
        poll_timeout_s=5,
        poll_interval_s=0.01,
        log=lambda *_: None,
    )

    assert summary["sessions_stored"] == 2
    assert summary["sessions_skipped"] == 0
    assert summary["sessions_failed"] == 0

    # Check both conversations got different user IDs
    user_ids = {c["user_id"] for c in fake.extract_calls}
    assert user_ids == {"trackb-mem0-locomo-c1", "trackb-mem0-locomo-c2"}

    await client.aclose()


@pytest.mark.asyncio
async def test_ingest_idempotent_resume(tmp_path):
    """Test idempotent resume from manifest."""
    fake = _FakeNS()
    client = NeuralscapeClient(
        "http://test",
        http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler),
            base_url="http://test",
        ),
    )

    conv = Conversation(
        conv_id="c1",
        sessions=(
            Session(session_id="s1", turns=(Turn(role="user", content="A: Hi"),)),
            Session(session_id="s2", turns=(Turn(role="user", content="A: Bye"),)),
        ),
    )

    manifest = IngestManifest(tmp_path / "manifest.json")

    # First run
    result1 = await ingest_conversation(
        client, conv,
        manifest=manifest,
        poll_timeout_s=5,
        poll_interval_s=0.01,
        log=lambda *_: None,
    )
    assert result1["stored"] == 2

    # Second run - should skip both
    result2 = await ingest_conversation(
        client, conv,
        manifest=manifest,
        poll_timeout_s=5,
        poll_interval_s=0.01,
        log=lambda *_: None,
    )
    assert result2["stored"] == 0
    assert result2["skipped"] == 2

    # Should not have made new extract calls
    assert len(fake.extract_calls) == 2

    await client.aclose()
