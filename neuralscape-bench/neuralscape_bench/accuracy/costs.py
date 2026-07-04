"""Analytic token/cost estimation per suite (pure — unit-tested).

No paid runs are needed: estimates extrapolate from dataset character
counts under DOCUMENTED assumptions. Every knob is a parameter with the
assumption stated next to its default; the report prints the assumptions
alongside the numbers so they can be re-derived or overridden.

Assumptions (defaults):

- 4 chars ≈ 1 token (English chat text).
- **Ingest**: each session is read by the NS extraction LLM once
  (prompt overhead ~600 tokens/call) and by the Graphiti enrichment
  pipeline ~2.5× more (entity extraction + resolution + edge passes over
  the episode) → ``ingest_llm_passes = 3.5`` input passes per session
  token. Extraction output ≈ 12% of input.
- **Ask (reasoning_level=high)**: ~5 retrieval passes × 25 hits × ~90
  tokens/hit context + prompt/thinking overhead ≈ 12k input tokens per
  question; output ≈ 500 tokens.
- **Judge**: ~450 input + ~40 output tokens per question.
- Pricing defaults are Gemini 2.5 Flash list prices (USD per 1M tokens);
  override to match the configured model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neuralscape_bench.accuracy.schema import SuiteData


@dataclass(frozen=True)
class CostModel:
    chars_per_token: float = 4.0
    ingest_llm_passes: float = 3.5          # NS extraction (1×) + Graphiti (~2.5×)
    ingest_prompt_overhead_tokens: int = 600  # per extraction call
    ingest_output_ratio: float = 0.12
    ask_input_tokens: int = 12_000          # per question at reasoning_level=high
    ask_output_tokens: int = 500
    judge_input_tokens: int = 450
    judge_output_tokens: int = 40
    usd_per_m_input: float = 0.30           # Gemini 2.5 Flash list price (assumed)
    usd_per_m_output: float = 2.50

    def assumptions(self) -> list[str]:
        return [
            f"{self.chars_per_token} chars/token",
            f"ingest: {self.ingest_llm_passes} LLM input passes/session "
            f"(+{self.ingest_prompt_overhead_tokens} prompt tokens/call), "
            f"output={self.ingest_output_ratio:.0%} of input",
            f"ask(high): {self.ask_input_tokens} in / {self.ask_output_tokens} out per question",
            f"judge: {self.judge_input_tokens} in / {self.judge_output_tokens} out per question",
            f"pricing: ${self.usd_per_m_input}/M in, ${self.usd_per_m_output}/M out "
            "(Gemini 2.5 Flash list — override for other models)",
        ]


@dataclass
class CostEstimate:
    suite: str
    sessions: int
    questions: int
    ingest_input_tokens: int
    ingest_output_tokens: int
    answer_input_tokens: int
    answer_output_tokens: int
    judge_input_tokens: int
    judge_output_tokens: int
    assumptions: list[str] = field(default_factory=list)

    @property
    def total_input(self) -> int:
        return self.ingest_input_tokens + self.answer_input_tokens + self.judge_input_tokens

    @property
    def total_output(self) -> int:
        return self.ingest_output_tokens + self.answer_output_tokens + self.judge_output_tokens

    def usd(self, model: CostModel) -> float:
        return round(
            self.total_input / 1e6 * model.usd_per_m_input
            + self.total_output / 1e6 * model.usd_per_m_output, 2)

    def to_dict(self, model: CostModel) -> dict:
        return {
            "suite": self.suite,
            "sessions": self.sessions,
            "questions": self.questions,
            "tokens": {
                "ingest_input": self.ingest_input_tokens,
                "ingest_output": self.ingest_output_tokens,
                "answer_input": self.answer_input_tokens,
                "answer_output": self.answer_output_tokens,
                "judge_input": self.judge_input_tokens,
                "judge_output": self.judge_output_tokens,
                "total_input": self.total_input,
                "total_output": self.total_output,
            },
            "estimated_usd": self.usd(model),
            "assumptions": self.assumptions,
        }


def estimate_suite_cost(data: SuiteData, *, model: CostModel = CostModel()) -> CostEstimate:
    stats = data.stats()
    conv_tokens = int(stats["conversation_chars"] / model.chars_per_token)
    n_sessions = stats["sessions"]
    n_questions = stats["qa_items"]

    ingest_in = int(conv_tokens * model.ingest_llm_passes
                    + n_sessions * model.ingest_prompt_overhead_tokens)
    ingest_out = int(conv_tokens * model.ingest_output_ratio)
    return CostEstimate(
        suite=data.suite,
        sessions=n_sessions,
        questions=n_questions,
        ingest_input_tokens=ingest_in,
        ingest_output_tokens=ingest_out,
        answer_input_tokens=n_questions * model.ask_input_tokens,
        answer_output_tokens=n_questions * model.ask_output_tokens,
        judge_input_tokens=n_questions * model.judge_input_tokens,
        judge_output_tokens=n_questions * model.judge_output_tokens,
        assumptions=model.assumptions(),
    )
