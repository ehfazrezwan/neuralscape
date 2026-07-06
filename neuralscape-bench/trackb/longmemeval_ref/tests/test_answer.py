"""Tests for LongMemEval answer protocol (mocked)."""

import json
from pathlib import Path

import httpx
import pytest

from neuralscape_bench.client import NeuralscapeClient
from neuralscape_bench.accuracy.manifest import read_jsonl_records
from neuralscape_bench.accuracy.schema import (
    Conversation,
    QAItem,
    Session,
    SuiteData,
    Turn,
)

from ..answer import answer_lme_questions


def _make_test_data() -> SuiteData:
    """Minimal LME-shaped data for testing."""
    conv = Conversation(
        conv_id="q_001",
        sessions=(
            Session(
                session_id="sess_a",
                turns=(
                    Turn(role="user", content="I adopted a beagle named Kiwi.", has_answer=True),
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
    """Minimal NS API mock: search + ask."""

    def __init__(self):
        self.search_calls = 0
        self.ask_calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/search":
            self.search_calls += 1
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"id": "m1", "memory": "I adopted a beagle named Kiwi.", "score": 0.9}
                    ]
                },
            )
        if path == "/v1/ask":
            self.ask_calls += 1
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "answer": "The dog is a beagle.",
                    "citations": ["m1"],
                    "abstained": False,
                    "searches": ["q"],
                    "memories_considered": 1,
                },
            )
        return httpx.Response(404, json={"detail": "nope"})


@pytest.mark.asyncio
async def test_answer_lme_questions(tmp_path):
    """Answer protocol: retrieve + /v1/ask, compute R@k."""
    fake = _FakeNS()
    client = NeuralscapeClient(
        "http://test",
        http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler), base_url="http://test"
        ),
    )
    data = _make_test_data()
    out_path = tmp_path / "answers.jsonl"

    summary = await answer_lme_questions(
        client,
        data,
        out_path=out_path,
        k=5,
        reasoning_level="high",
        concurrency=1,
        log=lambda *_: None,
    )

    assert summary["answered"] == 1
    assert summary["skipped"] == 0

    # One search + one ask
    assert fake.search_calls == 1
    assert fake.ask_calls == 1

    # Record written
    [rec] = read_jsonl_records(out_path)
    assert rec["qa_id"] == "q_001"
    assert rec["answer"] == "The dog is a beagle."
    assert rec["abstained"] is False
    # retrieval_hit should be true (the memory attributes to sess_a, which is gold evidence)
    assert rec["retrieval_hit"] is True

    await client.aclose()


@pytest.mark.asyncio
async def test_answer_lme_resumable(tmp_path):
    """Answer is resumable: already-answered questions are skipped."""
    fake = _FakeNS()
    client = NeuralscapeClient(
        "http://test",
        http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler), base_url="http://test"
        ),
    )
    data = _make_test_data()
    out_path = tmp_path / "answers.jsonl"

    # First run
    await answer_lme_questions(
        client, data, out_path=out_path, k=5, reasoning_level="high", concurrency=1, log=lambda *_: None
    )
    assert fake.ask_calls == 1

    # Second run: should skip
    summary2 = await answer_lme_questions(
        client, data, out_path=out_path, k=5, reasoning_level="high", concurrency=1, log=lambda *_: None
    )
    assert summary2["answered"] == 0
    assert summary2["skipped"] == 1
    assert fake.ask_calls == 1  # no new calls

    await client.aclose()


@pytest.mark.asyncio
async def test_answer_abstention_item(tmp_path):
    """Abstention items: R@k is None (no gold evidence)."""
    fake = _FakeNS()
    client = NeuralscapeClient(
        "http://test",
        http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler), base_url="http://test"
        ),
    )

    # Abstention item: empty evidence_session_ids
    conv = Conversation(
        conv_id="q_002_abs",
        sessions=(Session(session_id="s1", turns=(Turn(role="user", content="I like cycling."),)),),
    )
    data = SuiteData(
        suite="longmemeval_s",
        conversations=[conv],
        qa_items=[
            QAItem(
                qa_id="q_002_abs",
                conv_id="q_002_abs",
                question="What color is my car?",
                gold_answer="The user never mentioned a car.",
                qtype="multi-session",
                evidence_session_ids=(),  # abstention: no gold evidence
                is_abstention=True,
            )
        ],
    )
    out_path = tmp_path / "answers_abs.jsonl"

    await answer_lme_questions(
        client, data, out_path=out_path, k=5, reasoning_level="high", concurrency=1, log=lambda *_: None
    )

    [rec] = read_jsonl_records(out_path)
    # retrieval_hit is None for abstention items
    assert rec["retrieval_hit"] is None

    await client.aclose()
