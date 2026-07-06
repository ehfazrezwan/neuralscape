"""Ingest + answer runners against a MockTransport-backed client."""

import json

import httpx
import pytest

from neuralscape_bench.client import NeuralscapeClient
from neuralscape_bench.accuracy.answer import answer_suite
from neuralscape_bench.accuracy.ingest import ingest_suite
from neuralscape_bench.accuracy.manifest import IngestManifest, read_jsonl_records
from neuralscape_bench.accuracy.schema import (
    Conversation, QAItem, Session, SuiteData, Turn,
)


def _suite_data() -> SuiteData:
    conv = Conversation(conv_id="c1", sessions=(
        Session(session_id="s1", date="5 May 2023", turns=(
            Turn(role="user", content="Ana: I adopted a beagle named Kiwi."),
            Turn(role="assistant", content="Ben: Congrats!"),
        )),
        Session(session_id="s2", turns=(
            Turn(role="user", content="Ana: Kiwi chewed my shoes."),
        )),
    ))
    return SuiteData(
        suite="testsuite",
        conversations=[conv],
        qa_items=[QAItem(qa_id="q1", conv_id="c1", question="What is the dog's name?",
                         gold_answer="Kiwi", qtype="single-hop",
                         evidence_session_ids=("s1",))],
    )


class _FakeNS:
    """Minimal NS API: 202 writes, one-poll completion, search, ask."""

    def __init__(self):
        self.extract_calls: list[dict] = []
        self.search_user_ids: list[str] = []
        self.ask_user_ids: list[str] = []
        self.search_calls = 0
        self.ask_calls = 0
        self._tasks = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/memories" and request.method == "POST":
            body = json.loads(request.content)
            self.extract_calls.append(body)
            self._tasks += 1
            return httpx.Response(202, json={"task_id": f"t{self._tasks}"})
        if path.startswith("/v1/memories/status/"):
            return httpx.Response(200, json={"task_id": path.rsplit("/", 1)[-1],
                                             "status": "completed"})
        if path == "/v1/search":
            self.search_calls += 1
            self.search_user_ids.append(json.loads(request.content).get("user_id"))
            return httpx.Response(200, json={"results": [
                {"id": "m1", "memory": "Ana adopted a beagle named Kiwi", "score": 0.9},
                {"id": "m2", "memory": "Totally unrelated quantum fact", "score": 0.2},
            ]})
        if path == "/v1/ask":
            self.ask_calls += 1
            self.ask_user_ids.append(json.loads(request.content).get("user_id"))
            return httpx.Response(200, json={
                "status": "ok", "reasoning_level": "high",
                "answer": "The dog's name is Kiwi.", "citations": ["m1"],
                "abstained": False, "searches": ["q"], "memories_considered": 2,
            })
        return httpx.Response(404, json={"detail": "nope"})


@pytest.mark.asyncio
async def test_ingest_polls_and_resumes(tmp_path):
    fake = _FakeNS()
    client = NeuralscapeClient(
        "http://test", http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler), base_url="http://test"))
    data = _suite_data()
    manifest = IngestManifest(tmp_path / "m.json")

    summary = await ingest_suite(client, data, manifest=manifest,
                                 concurrency=2, poll_timeout_s=5, poll_interval_s=0.01,
                                 log=lambda *_: None)
    assert summary["sessions_stored"] == 2
    assert summary["sessions_failed"] == 0
    # one extract call per session, session-scoped run_id + bench user id
    assert len(fake.extract_calls) == 2
    assert fake.extract_calls[0]["user_id"] == "bench-testsuite-c1"
    assert {c["run_id"] for c in fake.extract_calls} == {"s1", "s2"}
    # date folded into the first message of the dated session
    dated = next(c for c in fake.extract_calls if c["run_id"] == "s1")
    assert dated["messages"][0]["content"].startswith("[This conversation session took place")

    # Second run: everything skipped (idempotent — poll, never re-store).
    summary2 = await ingest_suite(client, data, manifest=IngestManifest(tmp_path / "m.json"),
                                  concurrency=2, poll_timeout_s=5, poll_interval_s=0.01,
                                  log=lambda *_: None)
    assert summary2["sessions_stored"] == 0
    assert summary2["sessions_skipped"] == 2
    assert len(fake.extract_calls) == 2  # no new writes
    await client.aclose()


@pytest.mark.asyncio
async def test_answer_records_retrieval_and_resume(tmp_path):
    fake = _FakeNS()
    client = NeuralscapeClient(
        "http://test", http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler), base_url="http://test"))
    data = _suite_data()
    out = tmp_path / "answers.jsonl"

    summary = await answer_suite(client, data, out_path=out, k=5,
                                 reasoning_level="high", concurrency=1,
                                 log=lambda *_: None)
    assert summary["answered"] == 1
    [rec] = read_jsonl_records(out)
    assert rec["qa_id"] == "q1"
    assert rec["answer"] == "The dog's name is Kiwi."
    # the on-topic memory attributes to s1 (gold) → retrieval hit
    assert rec["retrieval_hit"] is True
    assert "s1" in rec["attributed_sessions"]

    # Resume: already-answered questions are skipped.
    summary2 = await answer_suite(client, data, out_path=out, k=5,
                                  reasoning_level="high", concurrency=1,
                                  log=lambda *_: None)
    assert summary2["answered"] == 0 and summary2["skipped"] == 1
    assert fake.ask_calls == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_ingest_namespace_prefixes_user_ids(tmp_path):
    """--user-namespace isolates mini-ingest writes into a namespaced user."""
    fake = _FakeNS()
    client = NeuralscapeClient(
        "http://test", http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler), base_url="http://test"))
    data = _suite_data()
    await ingest_suite(client, data, manifest=IngestManifest(tmp_path / "m.json"),
                       concurrency=2, poll_timeout_s=5, poll_interval_s=0.01,
                       namespace="pr-t11", log=lambda *_: None)
    assert fake.extract_calls  # sanity
    assert all(c["user_id"] == "bench-pr-t11-testsuite-c1" for c in fake.extract_calls)
    await client.aclose()


@pytest.mark.asyncio
async def test_answer_namespace_matches_ingest(tmp_path):
    """Answer path queries the same namespaced user the ingest wrote to."""
    fake = _FakeNS()
    client = NeuralscapeClient(
        "http://test", http=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler), base_url="http://test"))
    data = _suite_data()
    out = tmp_path / "answers.jsonl"
    await answer_suite(client, data, out_path=out, k=5, reasoning_level="high",
                       concurrency=1, namespace="pr-t11", log=lambda *_: None)
    assert fake.search_user_ids == ["bench-pr-t11-testsuite-c1"]
    assert fake.ask_user_ids == ["bench-pr-t11-testsuite-c1"]
    await client.aclose()
