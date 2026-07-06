"""Test answer runner with mocked NS client."""

import json

import httpx
import pytest

from neuralscape_bench.client import NeuralscapeClient
from neuralscape_bench.accuracy.manifest import read_jsonl_records
from neuralscape_bench.accuracy.schema import Conversation, QAItem, Session, SuiteData, Turn

from trackb.mem0_locomo.answer import answer_one, answer_suite


class _FakeNS:
    """Minimal NS API: search + ask."""

    def __init__(self):
        self.search_calls = 0
        self.ask_calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path

        if path == "/v1/search":
            self.search_calls += 1
            return httpx.Response(200, json={
                "results": [
                    {"id": "m1", "memory": "Alice got a beagle named Kiwi", "score": 0.9},
                    {"id": "m2", "memory": "Bob likes pizza", "score": 0.3},
                ],
            })

        if path == "/v1/ask":
            self.ask_calls += 1
            return httpx.Response(200, json={
                "status": "ok",
                "answer": "The dog's name is Kiwi.",
                "citations": ["m1"],
                "abstained": False,
                "searches": ["dog name"],
                "memories_considered": 2,
            })

        return httpx.Response(404, json={"detail": "not found"})


def _test_suite_data() -> SuiteData:
    """Test suite with one conversation and one QA."""
    conv = Conversation(
        conv_id="c1",
        sessions=(
            Session(session_id="s1", turns=(
                Turn(role="user", content="Alice: I got a beagle named Kiwi."),
            )),
        ),
    )
    return SuiteData(
        suite="test",
        conversations=[conv],
        qa_items=[
            QAItem(
                qa_id="q1",
                conv_id="c1",
                question="What is the dog's name?",
                gold_answer="Kiwi",
                qtype="4-single-hop",
                evidence_session_ids=("s1",),
            ),
        ],
    )


@pytest.mark.asyncio
async def test_answer_one_basic():
    """Test answering one QA item."""
    fake = _FakeNS()
    client = NeuralscapeClient(
        "http://test",
        http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler),
            base_url="http://test",
        ),
    )

    data = _test_suite_data()
    qa = data.qa_items[0]

    record = await answer_one(client, data, qa, k=10, reasoning_level="high")

    # Check record fields
    assert record["qa_id"] == "q1"
    assert record["conv_id"] == "c1"
    assert record["qtype"] == "4-single-hop"
    assert record["answer"] == "The dog's name is Kiwi."
    assert record["abstained"] is False
    assert record["citations"] == 1
    assert record["retrieved"] == 2

    # Check calls
    assert fake.search_calls == 1
    assert fake.ask_calls == 1

    await client.aclose()


@pytest.mark.asyncio
async def test_answer_suite_resumable(tmp_path):
    """Test answer suite with resume."""
    fake = _FakeNS()
    client = NeuralscapeClient(
        "http://test",
        http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler),
            base_url="http://test",
        ),
    )

    data = _test_suite_data()
    out_path = tmp_path / "answers.jsonl"

    # First run
    summary1 = await answer_suite(
        client, data,
        out_path=out_path,
        k=5,
        reasoning_level="high",
        concurrency=1,
        log=lambda *_: None,
    )

    assert summary1["answered"] == 1
    assert summary1["skipped"] == 0

    # Second run - should skip
    summary2 = await answer_suite(
        client, data,
        out_path=out_path,
        k=5,
        reasoning_level="high",
        concurrency=1,
        log=lambda *_: None,
    )

    assert summary2["answered"] == 0
    assert summary2["skipped"] == 1

    # Should only have 1 record in file
    records = read_jsonl_records(out_path)
    assert len(records) == 1

    # Should only have made 1 ask call (second run skipped)
    assert fake.ask_calls == 1

    await client.aclose()


@pytest.mark.asyncio
async def test_answer_one_with_abstention():
    """Test answering an abstention (adversarial) question."""
    fake_abstain = type("FakeNS", (), {
        "handler": lambda _, req: httpx.Response(200, json={
            "results": [],
        }) if req.url.path == "/v1/search" else httpx.Response(200, json={
            "status": "ok",
            "answer": "Not mentioned in the conversation.",
            "citations": [],
            "abstained": True,
            "searches": [],
            "memories_considered": 0,
        }),
    })()

    client = NeuralscapeClient(
        "http://test",
        http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake_abstain.handler),
            base_url="http://test",
        ),
    )

    data = SuiteData(
        suite="test",
        conversations=[Conversation(conv_id="c1", sessions=(
            Session(session_id="s1", turns=(Turn(role="user", content="A: Hi"),)),
        ))],
        qa_items=[
            QAItem(
                qa_id="q_adv",
                conv_id="c1",
                question="What is Alice's favorite color?",
                gold_answer="Not mentioned in the conversation",
                qtype="5-adversarial",
                is_abstention=True,
            ),
        ],
    )

    record = await answer_one(client, data, data.qa_items[0], k=10, reasoning_level="high")

    assert record["abstained"] is True
    assert "not mentioned" in record["answer"].lower()

    await client.aclose()
