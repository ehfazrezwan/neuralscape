"""Tests for LongMemEval ingest protocol (mocked)."""

import json
from pathlib import Path

import httpx
import pytest

from neuralscape_bench.client import NeuralscapeClient
from neuralscape_bench.accuracy.schema import (
    Conversation,
    QAItem,
    Session,
    SuiteData,
    Turn,
)

from ..ingest import ingest_lme_haystacks


def _make_test_data() -> SuiteData:
    """Minimal LME-shaped data for testing."""
    conv = Conversation(
        conv_id="q_001",
        sessions=(
            Session(
                session_id="sess_a",
                date="2023/05/05 (Fri) 13:00",
                turns=(
                    Turn(role="user", content="I adopted a beagle named Kiwi.", has_answer=True),
                    Turn(role="assistant", content="Congratulations!"),
                ),
            ),
            Session(
                session_id="sess_b",
                turns=(
                    Turn(role="user", content="What's for dinner?"),
                ),
            ),
        ),
    )
    return SuiteData(
        suite="longmemeval_s",
        conversations=[conv],
        qa_items=[
            QAItem(
                qa_id="q_001",
                conv_id="q_001",
                question="What breed is my dog?",
                gold_answer="Beagle",
                qtype="single-session-user",
                evidence_session_ids=("sess_a",),
            )
        ],
    )


class _FakeNS:
    """Minimal NS API mock: 202 writes, one-poll completion."""

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
            return httpx.Response(
                200, json={"task_id": path.rsplit("/", 1)[-1], "status": "completed"}
            )
        return httpx.Response(404, json={"detail": "nope"})


@pytest.mark.asyncio
async def test_ingest_lme_haystacks(tmp_path):
    """Ingest protocol: dated sessions, one user per question."""
    fake = _FakeNS()
    client = NeuralscapeClient(
        "http://test",
        http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler), base_url="http://test"
        ),
    )
    data = _make_test_data()
    manifest_path = tmp_path / "ingest.json"

    summary = await ingest_lme_haystacks(
        client,
        data,
        manifest_path=manifest_path,
        concurrency=1,
        poll_timeout_s=5,
        log=lambda *_: None,
    )

    assert summary["sessions_stored"] == 2
    assert summary["sessions_failed"] == 0

    # Two extract calls (one per session)
    assert len(fake.extract_calls) == 2

    # User ID: bench-longmemeval_s-<conv_id>
    assert all(c["user_id"] == "bench-longmemeval_s-q_001" for c in fake.extract_calls)

    # Session run_ids
    run_ids = {c["run_id"] for c in fake.extract_calls}
    assert run_ids == {"sess_a", "sess_b"}

    # Date folded into first message of dated session
    dated = next(c for c in fake.extract_calls if c["run_id"] == "sess_a")
    assert dated["messages"][0]["content"].startswith(
        "[This conversation session took place 2023/05/05 (Fri) 13:00"
    )

    # Undated session has no date prefix
    undated = next(c for c in fake.extract_calls if c["run_id"] == "sess_b")
    assert not undated["messages"][0]["content"].startswith("[This conversation")

    await client.aclose()


@pytest.mark.asyncio
async def test_ingest_lme_resumable(tmp_path):
    """Ingest is resumable: already-stored sessions are skipped."""
    fake = _FakeNS()
    client = NeuralscapeClient(
        "http://test",
        http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler), base_url="http://test"
        ),
    )
    data = _make_test_data()
    manifest_path = tmp_path / "ingest.json"

    # First run
    await ingest_lme_haystacks(
        client, data, manifest_path=manifest_path, concurrency=1, poll_timeout_s=5, log=lambda *_: None
    )
    assert len(fake.extract_calls) == 2

    # Second run: same manifest → should skip both sessions
    summary2 = await ingest_lme_haystacks(
        client, data, manifest_path=manifest_path, concurrency=1, poll_timeout_s=5, log=lambda *_: None
    )
    assert summary2["sessions_stored"] == 0
    assert summary2["sessions_skipped"] == 2
    assert len(fake.extract_calls) == 2  # no new writes

    await client.aclose()
