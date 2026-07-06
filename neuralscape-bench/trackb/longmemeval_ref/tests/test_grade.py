"""Tests for LongMemEval grading protocol."""

import json
from pathlib import Path

import httpx
import pytest

from neuralscape_bench.accuracy.judge import GeminiJudge, parse_verdict, render_judge_prompt
from neuralscape_bench.accuracy.manifest import append_jsonl, read_jsonl_records
from neuralscape_bench.accuracy.schema import QAItem, SuiteData

from ..grade import grade_lme_answers


def test_render_judge_prompt_standard():
    """Judge prompt rendering for standard QA item."""
    qa = QAItem(
        qa_id="q1",
        conv_id="c1",
        question="What breed is my dog?",
        gold_answer="Beagle",
        qtype="single-session-user",
    )
    prompt = render_judge_prompt(qa, "The dog is a beagle.")
    assert "What breed is my dog?" in prompt
    assert "Beagle" in prompt
    assert "The dog is a beagle." in prompt
    # No abstention note for standard items
    assert "not available / unknown" not in prompt


def test_render_judge_prompt_abstention():
    """Judge prompt rendering for abstention item."""
    qa = QAItem(
        qa_id="q2_abs",
        conv_id="c2",
        question="What color is my car?",
        gold_answer="Not mentioned",
        qtype="multi-session",
        is_abstention=True,
    )
    prompt = render_judge_prompt(qa, "I don't have that information.")
    assert "What color is my car?" in prompt
    assert "not available / unknown" in prompt  # abstention note


def test_parse_verdict_json():
    """Parse JSON verdict from judge response."""
    text = '{"correct": true, "reason": "Answer matches gold"}'
    correct, reason = parse_verdict(text)
    assert correct is True
    assert "Answer matches gold" in reason


def test_parse_verdict_json_false():
    """Parse false verdict."""
    text = '{"correct": false, "reason": "Missing key fact"}'
    correct, reason = parse_verdict(text)
    assert correct is False


def test_parse_verdict_fallback():
    """Parse fallback when JSON is messy."""
    text = 'Some preamble\n{"correct": true}\nSome postamble'
    correct, reason = parse_verdict(text)
    assert correct is True


def test_parse_verdict_unparseable():
    """Unparseable response returns None."""
    text = "This is not a valid verdict"
    correct, reason = parse_verdict(text)
    assert correct is None
    assert "unparseable" in reason


class _FakeGemini:
    """Minimal Gemini API mock."""

    def __init__(self, verdict: bool = True):
        self.verdict = verdict
        self.calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {"correct": self.verdict, "reason": "test reason"}
                                    )
                                }
                            ]
                        }
                    }
                ]
            },
        )


@pytest.mark.asyncio
async def test_grade_lme_answers(tmp_path, monkeypatch):
    """Grade answers using LLM judge."""
    fake = _FakeGemini(verdict=True)

    # Mock GeminiJudge to use our fake transport
    original_init = GeminiJudge.__init__

    def mock_init(self, api_key, *, model="gemini-3.1-flash-lite", timeout_s=60.0, max_retries=6, http=None):
        self.model = model
        self._key = api_key
        self._max_retries = max_retries
        self._http = httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler),
            base_url="https://generativelanguage.googleapis.com",
        )

    monkeypatch.setattr(GeminiJudge, "__init__", mock_init)

    # Prepare test data
    data = SuiteData(
        suite="longmemeval_s",
        qa_items=[
            QAItem(
                qa_id="q1",
                conv_id="c1",
                question="What breed?",
                gold_answer="Beagle",
                qtype="single-session-user",
            )
        ],
    )
    answers_path = tmp_path / "answers.jsonl"
    judged_path = tmp_path / "judged.jsonl"

    # Write fake answer
    append_jsonl(
        answers_path,
        {"qa_id": "q1", "qtype": "single-session-user", "answer": "Beagle"},
    )

    summary = await grade_lme_answers(
        data,
        answers_path=answers_path,
        judged_path=judged_path,
        judge_model="gemini-3.1-flash-lite",
        api_key="fake-key",
        concurrency=1,
        log=lambda *_: None,
    )

    assert summary["judged"] == 1
    # Unparseable check is lenient - parse_verdict handles various formats
    # Just verify the record was judged
    assert fake.calls == 1

    # Verdict written
    [rec] = read_jsonl_records(judged_path)
    assert rec["qa_id"] == "q1"
    assert rec["judge_model"] == "gemini-3.1-flash-lite"


@pytest.mark.asyncio
async def test_grade_lme_resumable(tmp_path, monkeypatch):
    """Grading is resumable: already-judged items are skipped."""
    fake = _FakeGemini(verdict=True)

    # Mock GeminiJudge to use our fake transport
    def mock_init(self, api_key, *, model="gemini-3.1-flash-lite", timeout_s=60.0, max_retries=6, http=None):
        self.model = model
        self._key = api_key
        self._max_retries = max_retries
        self._http = httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handler),
            base_url="https://generativelanguage.googleapis.com",
        )

    monkeypatch.setattr(GeminiJudge, "__init__", mock_init)

    data = SuiteData(
        suite="longmemeval_s",
        qa_items=[
            QAItem(
                qa_id="q1",
                conv_id="c1",
                question="What breed?",
                gold_answer="Beagle",
                qtype="single-session-user",
            )
        ],
    )
    answers_path = tmp_path / "answers.jsonl"
    judged_path = tmp_path / "judged.jsonl"

    append_jsonl(
        answers_path,
        {"qa_id": "q1", "qtype": "single-session-user", "answer": "Beagle"},
    )

    # First run
    await grade_lme_answers(
        data, answers_path=answers_path, judged_path=judged_path,
        judge_model="gemini-3.1-flash-lite", api_key="fake-key", concurrency=1,
        log=lambda *_: None,
    )
    assert fake.calls == 1

    # Second run: should skip
    summary2 = await grade_lme_answers(
        data, answers_path=answers_path, judged_path=judged_path,
        judge_model="gemini-3.1-flash-lite", api_key="fake-key", concurrency=1,
        log=lambda *_: None,
    )
    assert summary2["judged"] == 0
    assert summary2["skipped"] == 1
    assert fake.calls == 1  # no new calls
