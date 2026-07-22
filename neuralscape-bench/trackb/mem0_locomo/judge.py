"""LLM judge for mem0 LoCoMo answers (mirrors mem0's correctness judge).

Uses gemini-3.1-flash-lite @ temp=0 (locked standard model).
Judges answer correctness against gold answer.
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Callable

import httpx

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# Locked judge model (standard)
JUDGE_MODEL = "gemini-3.1-flash-lite"
JUDGE_TEMP = 0.0

# mem0's LoCoMo correctness judge prompt (faithfully reproduced)
JUDGE_PROMPT = """You are an expert judge evaluating whether a predicted answer correctly answers a question compared to a gold reference answer.

Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted_answer}

Evaluate whether the predicted answer is correct. Consider:
1. Does it convey the same core information as the gold answer?
2. For factual questions, are the key facts aligned?
3. For adversarial/unanswerable questions, does the predicted answer appropriately abstain (e.g., "not mentioned", "I don't know")?

Output ONLY "correct" or "incorrect" (lowercase, one word).
"""


def normalize_judgment(text: str) -> str | None:
    """Extract correct/incorrect from LLM output."""
    text = text.strip().lower()
    if "correct" in text and "incorrect" not in text:
        return "correct"
    if "incorrect" in text:
        return "incorrect"
    return None


async def judge_one(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    is_abstention: bool,
    *,
    api_key: str | None = None,
) -> dict:
    """Judge one answer pair. Returns {judgment: correct/incorrect/error, raw: ...}."""
    if not predicted_answer.strip():
        return {"judgment": "incorrect", "raw": "empty_prediction", "error": None}

    # Special handling for adversarial/abstention questions
    if is_abstention:
        # Check if prediction abstains appropriately
        pred_lower = predicted_answer.lower()
        abstain_signals = ["not mentioned", "don't know", "no information", "cannot answer"]
        if any(sig in pred_lower for sig in abstain_signals):
            return {"judgment": "correct", "raw": "abstention_matched", "error": None}

    # LLM judge
    if not HAS_GENAI:
        return {"judgment": "error", "raw": "", "error": "google-genai not installed"}

    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
    )

    try:
        client = genai.Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=JUDGE_TEMP,
                max_output_tokens=10,
            ),
        )

        raw = response.text.strip() if response.text else ""
        judgment = normalize_judgment(raw)
        if judgment is None:
            judgment = "error"

        return {"judgment": judgment, "raw": raw, "error": None}

    except Exception as e:
        return {"judgment": "error", "raw": "", "error": str(e)[:200]}


async def _with_backoff(coro_factory, *, max_retries: int = 5, base_delay: float = 2.0):
    """Exponential backoff for transient errors."""
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            # Gemini rate limit or transient error
            if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                await asyncio.sleep(delay + random.uniform(0, delay / 2))
                delay = min(delay * 2, 60.0)
            else:
                raise
    raise RuntimeError("unreachable")


async def judge_suite(
    records: list[dict],
    qa_items: list,
    *,
    concurrency: int = 5,
    api_key: str | None = None,
    log: Callable = print,
) -> list[dict]:
    """Judge all answer records. Returns records with 'judgment' field added.

    Args:
        records: Answer records from answer.py (JSONL loaded)
        qa_items: QAItem objects for gold answers
        concurrency: Parallel judge calls
        api_key: Google API key (defaults to GOOGLE_API_KEY env)
        log: Logging callable
    """
    qa_map = {qa.qa_id: qa for qa in qa_items}
    log(f"Judging {len(records)} answers (concurrency={concurrency})")

    sem = asyncio.Semaphore(concurrency)
    judged = 0

    async def _bounded(rec: dict) -> dict:
        nonlocal judged
        async with sem:
            qa_id = rec.get("qa_id")
            qa = qa_map.get(qa_id)
            if qa is None:
                rec["judgment"] = "error"
                rec["judge_error"] = "qa_item_not_found"
                return rec

            result = await _with_backoff(
                lambda: judge_one(
                    question=qa.question,
                    gold_answer=qa.gold_answer,
                    predicted_answer=rec.get("answer", ""),
                    is_abstention=qa.is_abstention,
                    api_key=api_key,
                )
            )

            rec["judgment"] = result["judgment"]
            rec["judge_raw"] = result["raw"]
            if result["error"]:
                rec["judge_error"] = result["error"]

            judged += 1
            if judged % 20 == 0:
                log(f"[judge] {judged}/{len(records)} complete")

            return rec

    results = await asyncio.gather(*[_bounded(rec) for rec in records])

    log(f"Judging complete: {judged} judged")
    return results
