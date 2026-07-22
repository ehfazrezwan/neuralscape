"""
Capped ReAct-style agent loop, backend-agnostic.

The agent is handed a set of tools and a task prompt; it calls tools until it
submits an answer or hits the step cap (= "gave up"). The loop is deliberately
minimal — no scratchpad tricks, no ret/rewriting — so the token cost reflects the
raw work of navigating with the available tools. All token usage is accumulated
from the LLM turns (see :mod:`llm`).
"""

import json
import logging
from dataclasses import dataclass, field

from icebench.tokensave.llm import Usage
from icebench.tokensave.tools import ToolResult

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 12


@dataclass
class TraceStep:
    step: int
    tool: str
    args: dict
    observation: str
    raw: dict = field(default_factory=dict)
    ok: bool = True


@dataclass
class AgentResult:
    answer: dict  # {"location": "file:line"} | {"callers": [...]} | {"gave_up": True}
    usage: Usage
    n_tool_calls: int
    n_file_reads: int
    n_memory_calls: int
    steps: int
    stopped_reason: str  # "submitted" | "step_cap" | "no_action" | "error"
    trace: list[TraceStep] = field(default_factory=list)

    def first_memory_step(self) -> TraceStep | None:
        for s in self.trace:
            if s.tool == "code_memory":
                return s
        return None

    def first_tool(self) -> str | None:
        return self.trace[0].tool if self.trace else None


BASELINE_SYSTEM = (
    "You are a precise code-navigation agent working inside a source repository. "
    "You can ONLY inspect the repository through the provided file tools "
    "(list_dir, grep, read_file). Find the answer with as few tool calls as "
    "possible, then call submit_answer. Do not guess: verify a location by "
    "reading it before submitting. If, after a genuine effort, you cannot "
    "determine the answer, call submit_answer with gave_up=true."
)

MEMORY_SYSTEM = (
    "You are a precise code-navigation agent working inside a source repository. "
    "You have a code_memory index that can return a location directly WITHOUT "
    "reading files, plus file tools (list_dir, grep, read_file) as a fallback. "
    "ALWAYS try code_memory FIRST — it is far cheaper than reading files. Only "
    "read files if the memory result is empty or you must verify. Then call "
    "submit_answer. If you genuinely cannot determine the answer, call "
    "submit_answer with gave_up=true."
)


def _coerce_args(args: dict) -> dict:
    """google-genai sometimes returns numeric args as floats; normalize a bit."""
    out = {}
    for k, v in (args or {}).items():
        if isinstance(v, float) and v.is_integer():
            out[k] = int(v)
        else:
            out[k] = v
    return out


def run_agent(llm, dispatch, task_prompt: str, max_steps: int = DEFAULT_MAX_STEPS) -> AgentResult:
    """
    Run the capped agent loop.

    Args:
        llm: an LLM client exposing make_user_message(text), make_tool_response(
            name, result), and turn(contents) -> LLMTurn.
        dispatch: callable(name: str, args: dict) -> ToolResult for tool calls
            OTHER than submit_answer.
        task_prompt: the navigation question.
        max_steps: hard cap on model turns (a turn may issue several tool calls).

    Returns:
        AgentResult with the answer, accumulated token usage, and full trace.
    """
    contents = [llm.make_user_message(task_prompt)]
    usage = Usage()
    trace: list[TraceStep] = []
    n_file_reads = 0
    n_memory_calls = 0
    answer: dict | None = None
    stopped_reason = "step_cap"

    step = 0
    while step < max_steps:
        step += 1
        turn = llm.turn(contents)
        usage.add(turn.usage)
        contents.append(turn.model_content)

        if not turn.function_calls:
            # No tool call: nudge once, else stop. A well-behaved model calls
            # submit_answer; free-text with no action is treated as no_action.
            if step >= max_steps:
                stopped_reason = "no_action"
                break
            contents.append(
                llm.make_user_message(
                    "Please either call a tool to gather evidence or call "
                    "submit_answer with your final answer."
                )
            )
            continue

        submitted = False
        for fc in turn.function_calls:
            name = fc.name
            args = _coerce_args(fc.args)

            if name == "submit_answer":
                answer = {
                    k: v for k, v in args.items()
                    if k in ("location", "callers", "gave_up")
                }
                stopped_reason = "submitted"
                submitted = True
                break

            result: ToolResult = dispatch(name, args)
            if name == "read_file":
                n_file_reads += 1
            elif name == "code_memory":
                n_memory_calls += 1
            trace.append(
                TraceStep(
                    step=step, tool=name, args=args,
                    observation=result.observation, raw=result.raw, ok=result.ok,
                )
            )
            contents.append(llm.make_tool_response(name, result.observation))

        if submitted:
            break

    if answer is None:
        answer = {"gave_up": True}

    return AgentResult(
        answer=answer,
        usage=usage,
        n_tool_calls=len(trace),
        n_file_reads=n_file_reads,
        n_memory_calls=n_memory_calls,
        steps=step,
        stopped_reason=stopped_reason,
        trace=trace,
    )
