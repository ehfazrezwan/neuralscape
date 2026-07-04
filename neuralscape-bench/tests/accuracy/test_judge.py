"""Judge prompt rendering + verdict parsing (pure) and the retry client."""

import httpx
import pytest

from neuralscape_bench.accuracy.judge import (
    GeminiJudge, parse_verdict, render_judge_prompt,
)
from neuralscape_bench.accuracy.schema import QAItem


def _qa(**kw) -> QAItem:
    base = dict(qa_id="q1", conv_id="c1", question="What breed is my dog?",
                gold_answer="Beagle", qtype="single-session-user")
    base.update(kw)
    return QAItem(**base)


def test_render_prompt_basic():
    p = render_judge_prompt(_qa(), "It's a beagle.")
    assert "What breed is my dog?" in p
    assert "Gold answer: Beagle" in p
    assert "Model answer: It's a beagle." in p
    assert "abstains" not in p  # no abstention note for normal questions
    assert "rubric" not in p.lower()


def test_render_prompt_abstention_and_rubric_and_date():
    qa = _qa(is_abstention=True, rubric=("Must say no info exists",),
             question_date="2023/05/30")
    p = render_judge_prompt(qa, "")
    assert "correct behavior for this question is to say the information is" in p
    assert "- Must say no info exists" in p
    assert "Question asked on: 2023/05/30" in p
    assert "(empty answer)" in p


@pytest.mark.parametrize("text,expected", [
    ('{"correct": true, "reason": "matches"}', True),
    ('{"correct": false, "reason": "wrong date"}', False),
    ('Sure! Here you go: {"correct": true, "reason": "ok"}', True),  # wrapped
    ("true", True),
    ("FALSE", False),
    ("", None),
    ("The answer seems plausible.", None),
    ('{"correct": "yes"}', None),  # non-boolean → unparseable
])
def test_parse_verdict(text, expected):
    correct, _reason = parse_verdict(text)
    assert correct is expected


def _gemini_response(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


@pytest.mark.asyncio
async def test_judge_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=_gemini_response('{"correct": true, "reason": "ok"}'))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    judge = GeminiJudge("test-key", model="test-model", http=http)
    # Shrink backoff for the test.
    judge._max_retries = 3
    import neuralscape_bench.accuracy.judge as judge_mod
    orig_sleep = judge_mod.asyncio.sleep

    async def fast_sleep(_s):
        await orig_sleep(0)

    judge_mod.asyncio.sleep, restore = fast_sleep, orig_sleep
    try:
        verdict = await judge.judge(_qa(), "a beagle")
    finally:
        judge_mod.asyncio.sleep = restore
        await judge.aclose()
    assert verdict["correct"] is True
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_judge_gives_none_after_exhausted_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "overloaded"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    judge = GeminiJudge("test-key", model="test-model", max_retries=2, http=http)
    import neuralscape_bench.accuracy.judge as judge_mod
    orig_sleep = judge_mod.asyncio.sleep

    async def fast_sleep(_s):
        await orig_sleep(0)

    judge_mod.asyncio.sleep = fast_sleep
    try:
        verdict = await judge.judge(_qa(), "a beagle")
    finally:
        judge_mod.asyncio.sleep = orig_sleep
        await judge.aclose()
    assert verdict["correct"] is None
    assert "judge failed" in verdict["reason"]
