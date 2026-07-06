"""Answer QA items using mem0 retrieval + LLM generation."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from neuralscape_bench.accuracy.schema import QAItem, SuiteData

logger = logging.getLogger(__name__)

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# Answer prompt mirroring NSBench's /ask endpoint logic
_ANSWER_PROMPT = """You are an AI assistant with access to memory about a user's past conversations.

Retrieved memories:
{memories}

User question: {question}

Based on the retrieved memories above, provide a concise, factual answer to the question. If the memories don't contain the information needed to answer, say "I don't have enough information to answer this question" rather than guessing.

Answer:"""


class GeminiAnswerer:
    """Minimal Gemini REST client for answer generation (mirrors NSBench backbone)."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        max_retries: int = 6,
        http: httpx.AsyncClient | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self._key = api_key
        self._max_retries = max_retries
        self._http = http or httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def generate(self, prompt: str) -> str:
        """Generate text with retry/backoff."""
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 1024,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        url = _GEMINI_URL.format(model=self.model)
        delay = 2.0
        last_err = "unknown"

        for attempt in range(self._max_retries):
            try:
                r = await self._http.post(
                    url,
                    json=body,
                    headers={"x-goog-api-key": self._key},
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
                return text.strip()
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                last_err = str(e)
                if attempt == self._max_retries - 1:
                    break
                await asyncio.sleep(delay + random.uniform(0, delay / 2))
                delay = min(delay * 2, 60.0)

        logger.warning(f"Answer generation failed after retries: {last_err}")
        return f"[generation failed: {last_err}]"


def _extract_memories(results) -> list[dict]:
    """Normalize mem0.search() output into a list of memory dicts.

    Vendored mem0 (v1.1+) returns ``{"results": [...]}``; older/other shapes
    may return a bare list. Handle both defensively.
    """
    if isinstance(results, dict):
        items = results.get("results", [])
        return items if isinstance(items, list) else []
    if isinstance(results, list):
        return results
    return []


def _render_answer_prompt(memories: list[dict], question: str) -> str:
    """Build the answer-generation prompt from retrieved memories."""
    mem_text = "\n".join(
        f"- {m.get('memory', m.get('text', str(m)))}"
        for m in memories
    )
    if not mem_text.strip():
        mem_text = "(no memories retrieved)"
    return _ANSWER_PROMPT.format(memories=mem_text, question=question)


async def answer_suite(
    memory_class,
    config_dict: dict,
    data: "SuiteData",
    answerer: GeminiAnswerer,
    *,
    out_path: Path,
    k: int = 10,
    concurrency: int = 4,
    log=print,
) -> dict:
    """Answer all QA items using mem0 retrieval + LLM generation.

    For each QA:
    1. Initialize Memory for the conversation's user_id
    2. Search mem0 with the question (top-k)
    3. Render answer prompt with retrieved memories
    4. Generate answer via Gemini
    5. Append to JSONL

    Args:
        memory_class: mem0.Memory class
        config_dict: mem0 config dict
        data: SuiteData with QA items
        answerer: GeminiAnswerer for generation
        out_path: Output JSONL path
        k: Retrieval depth
        concurrency: Parallel QA processing
        log: Logging function

    Returns:
        Summary dict
    """
    from neuralscape_bench.accuracy.manifest import append_jsonl  # noqa: PLC0415

    qa_by_id = {qa.qa_id: qa for qa in data.qa_items}
    log(f"[answer] {data.suite}: {len(data.qa_items)} QA items")

    # Truncate on every fresh answer run so re-runs never blend stale records
    # into the report (append-only writes below would otherwise mix old + new).
    # Cross-run isolation is additionally provided by the --run-label suffix on
    # the filename (see run.py._raw_paths).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    counters = {"done": 0}

    async def one(qa: "QAItem"):
        async with sem:
            user_id = f"{data.suite}-{qa.conv_id}"

            # Initialize Memory for this user (same config)
            memory = memory_class(config_dict)

            # Retrieve memories. Vendored mem0 signature is
            # search(query, *, top_k=20, filters=None, ...) — the entity id
            # MUST go inside filters (top-level user_id/limit kwargs are
            # rejected by the library), and results come back as {"results": [...]}.
            try:
                results = memory.search(
                    qa.question, top_k=k, filters={"user_id": user_id}
                )
                memories = _extract_memories(results)
            except Exception as e:
                logger.warning(f"mem0 search failed for {qa.qa_id}: {e}")
                memories = []

            # Generate answer
            prompt = _render_answer_prompt(memories, qa.question)
            answer = await answerer.generate(prompt)

            # Write result
            record = {
                "qa_id": qa.qa_id,
                "qtype": qa.qtype,
                "question": qa.question,
                "answer": answer,
                "retrieved_k": len(memories),
            }

            async with lock:
                append_jsonl(out_path, record)
                counters["done"] += 1
                if counters["done"] % 25 == 0 or counters["done"] == len(data.qa_items):
                    log(f"[answer] {data.suite}: {counters['done']}/{len(data.qa_items)}")

    await asyncio.gather(*(one(qa) for qa in data.qa_items))

    return {
        "answered": counters["done"],
        "out": str(out_path),
    }
