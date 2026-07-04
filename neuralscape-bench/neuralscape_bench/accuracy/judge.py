"""LLM-judge correctness scoring (the standard for all six suites).

Prompt rendering and verdict parsing are pure (unit-tested); the network
call is a thin Gemini ``generateContent`` REST client (temperature 0) using
the same ``GOOGLE_API_KEY`` as the service, with exponential backoff on
429/5xx so rate limits throttle rather than fail the run.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re

import httpx

from neuralscape_bench.accuracy.schema import QAItem

DEFAULT_JUDGE_MODEL = os.environ.get("BENCH_JUDGE_MODEL", "gemini-2.5-flash")
_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

_PROMPT = """You are grading a memory system's answer against a gold reference answer.

Question: {question}
{question_date}Gold answer: {gold}
{rubric}{abstention}Model answer: {answer}

Grade the model answer as CORRECT if it conveys the same essential information
as the gold answer. Wording differences, extra correct detail, and formatting
do not matter. Dates must match the gold answer's meaning (allow equivalent
formats). Numeric answers must match. If the model answer is missing the key
fact, contradicts the gold answer, or hedges without answering, grade INCORRECT.

Respond with ONLY a JSON object: {{"correct": true or false, "reason": "<one short sentence>"}}"""

_ABSTENTION_NOTE = (
    "NOTE: the correct behavior for this question is to say the information is "
    "not available / unknown. Grade CORRECT if the model abstains or states there "
    "is no such information; grade INCORRECT if it fabricates an answer.\n"
)


def render_judge_prompt(qa: QAItem, model_answer: str) -> str:
    """Pure prompt construction (fixture-tested)."""
    rubric = ""
    if qa.rubric:
        rubric = "Grading rubric (the answer must satisfy these points):\n" + \
            "\n".join(f"- {r}" for r in qa.rubric) + "\n"
    qdate = f"Question asked on: {qa.question_date}\n" if qa.question_date else ""
    return _PROMPT.format(
        question=qa.question.strip(),
        question_date=qdate,
        gold=qa.gold_answer.strip() or "(no gold answer text)",
        rubric=rubric,
        abstention=_ABSTENTION_NOTE if qa.is_abstention else "",
        answer=(model_answer or "").strip() or "(empty answer)",
    )


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_verdict(text: str) -> tuple[bool | None, str]:
    """Parse the judge's JSON verdict → (correct, reason). None = unparseable."""
    if not text:
        return None, "empty judge response"
    for candidate in _JSON_RE.findall(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("correct"), bool):
            return obj["correct"], str(obj.get("reason", ""))
    # Fallback: bare true/false tokens.
    lowered = text.lower()
    if '"correct": true' in lowered or lowered.strip() in ("true", "correct"):
        return True, "fallback parse"
    if '"correct": false' in lowered or lowered.strip() in ("false", "incorrect"):
        return False, "fallback parse"
    return None, f"unparseable: {text[:120]}"


class GeminiJudge:
    """Minimal Gemini REST judge with retry/backoff (temperature 0)."""

    def __init__(self, api_key: str, *, model: str = DEFAULT_JUDGE_MODEL,
                 timeout_s: float = 60.0, max_retries: int = 6,
                 http: httpx.AsyncClient | None = None):
        self.model = model
        self._key = api_key
        self._max_retries = max_retries
        self._http = http or httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def judge(self, qa: QAItem, model_answer: str) -> dict:
        prompt = render_judge_prompt(qa, model_answer)
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            # thinkingBudget 0: on thinking-enabled Gemini models (2.5-flash+)
            # thoughts otherwise consume the output cap and truncate the JSON
            # verdict mid-object (observed: ~20% unparseable on DMR).
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024,
                                 "thinkingConfig": {"thinkingBudget": 0}},
        }
        url = _GEMINI_URL.format(model=self.model)
        delay = 2.0
        last_err = "unknown"
        for attempt in range(self._max_retries):
            try:
                r = await self._http.post(
                    url, json=body, headers={"x-goog-api-key": self._key},
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}"
                    raise httpx.HTTPStatusError(last_err, request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                text = "".join(
                    p.get("text", "")
                    for c in data.get("candidates", [])[:1]
                    for p in c.get("content", {}).get("parts", [])
                )
                correct, reason = parse_verdict(text)
                return {"correct": correct, "reason": reason, "judge_model": self.model}
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                last_err = str(e)
                if attempt == self._max_retries - 1:
                    break
                await asyncio.sleep(delay + random.uniform(0, delay / 2))
                delay = min(delay * 2, 60.0)
        return {"correct": None, "reason": f"judge failed: {last_err}", "judge_model": self.model}


async def judge_suite(judge: GeminiJudge, data, answers_path, judged_path, *,
                      concurrency: int = 4, log=print) -> dict:
    """Judge every answered-but-unjudged record (resumable via judged JSONL)."""
    from neuralscape_bench.accuracy.manifest import (  # noqa: PLC0415 — avoid cycle
        append_jsonl, read_jsonl_records,
    )

    qa_by_id: dict[str, QAItem] = {qa.qa_id: qa for qa in data.qa_items}
    answered = read_jsonl_records(answers_path)
    # Resume keys on (qa_id, qtype): plain qa_id is not unique in suites with
    # the known cross-category stem collision (ConvoMem), and keying on it
    # would silently skip unjudged records after an interrupted judge pass.
    done = {(r.get("qa_id"), r.get("qtype"))
            for r in read_jsonl_records(judged_path) if r.get("qa_id")}
    todo = [r for r in answered
            if r.get("qa_id") in qa_by_id and (r["qa_id"], r.get("qtype")) not in done]
    log(f"[judge] {data.suite}: {len(todo)} to judge ({len(done)} already judged)")

    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    counters = {"done": 0, "unparseable": 0}

    async def one(rec: dict):
        async with sem:
            qa = qa_by_id[rec["qa_id"]]
            verdict = await judge.judge(qa, rec.get("answer", ""))
            merged = {**rec, **verdict}
            async with lock:
                append_jsonl(judged_path, merged)
                counters["done"] += 1
                if verdict["correct"] is None:
                    counters["unparseable"] += 1
                if counters["done"] % 25 == 0 or counters["done"] == len(todo):
                    log(f"[judge] {data.suite}: {counters['done']}/{len(todo)}")

    await asyncio.gather(*(one(r) for r in todo))
    return {"judged": counters["done"], "skipped": len(done),
            "unparseable": counters["unparseable"], "out": str(judged_path)}
