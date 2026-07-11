"""
Gemini agent backend (google-genai) with precise token accounting.

The agent is a small function-calling ReAct loop; this module owns the LLM
transport and normalizes each turn so :mod:`agent` stays backend-agnostic (and
trivially mockable in tests). Tokens are counted from the provider's own
``usage_metadata`` (``prompt_token_count`` + ``candidates_token_count`` +
``thoughts_token_count``) summed across every turn — i.e. the exact tokens the
LLM burned to solve the task, which is the North-Star metric.

Determinism: temperature 0, fixed seed, thinking disabled by default. An LLM is
never perfectly deterministic; this is stated as a v1 caveat in the report.
"""

import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("ICE_TOKENSAVE_MODEL", "gemini-2.5-flash")
DEFAULT_MAX_RETRIES = 6
DEFAULT_BASE_BACKOFF_S = 2.0


@dataclass
class Usage:
    """Token usage for one or many turns (additive)."""

    prompt_tokens: int = 0
    output_tokens: int = 0  # candidates
    thought_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.output_tokens += other.output_tokens
        self.thought_tokens += other.thought_tokens
        self.total_tokens += other.total_tokens
        self.calls += other.calls

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thought_tokens": self.thought_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


@dataclass
class FunctionCall:
    name: str
    args: dict


@dataclass
class LLMTurn:
    """One normalized model turn."""

    text: str | None
    function_calls: list[FunctionCall]
    usage: Usage
    model_content: object  # opaque content to append back to history
    finish_reason: str | None = None


class GeminiClient:
    """Thin google-genai wrapper exposing a backend-agnostic turn interface."""

    def __init__(
        self,
        tool_decls: list[dict],
        system_instruction: str,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        thinking_budget: int = 0,
        max_output_tokens: int = 8192,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        from google import genai
        from google.genai import types

        self._genai = genai
        self._types = types
        self.model = model
        self.max_retries = max_retries
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set (required for the Gemini agent)")
        self.client = genai.Client(api_key=api_key)

        # Build the tool + generation config once.
        fdecls = [types.FunctionDeclaration(**decl) for decl in tool_decls]
        tools = [types.Tool(function_declarations=fdecls)]
        cfg_kwargs = dict(
            temperature=temperature,
            seed=seed,
            # Realistic output cap: real agents don't emit tens of thousands of
            # output tokens. Prevents rare degenerate enumeration blow-ups (e.g.
            # listing every caller of a popular helper) from dominating token
            # accounting. Well above any legitimate answer length.
            max_output_tokens=max_output_tokens,
            tools=tools,
            system_instruction=system_instruction,
            # Force the model to actually call our tools rather than free-text.
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
        )
        # Thinking is only configurable on models that support it; guard it.
        if thinking_budget is not None:
            try:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )
            except Exception:  # pragma: no cover - SDK/model variance
                pass
        self.config = types.GenerateContentConfig(**cfg_kwargs)

    def close(self) -> None:
        """Best-effort close of the underlying genai/httpx client."""
        try:
            self.client.close()
        except Exception:  # pragma: no cover - SDK variance
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- content builders (own the provider content format) ----------------- #
    def make_user_message(self, text: str):
        types = self._types
        return types.Content(role="user", parts=[types.Part.from_text(text=text)])

    def make_tool_response(self, name: str, result: str):
        types = self._types
        return types.Content(
            role="tool",
            parts=[types.Part.from_function_response(name=name, response={"result": result})],
        )

    # -- the turn ----------------------------------------------------------- #
    def _extract_usage(self, resp) -> Usage:
        um = getattr(resp, "usage_metadata", None)
        if um is None:
            return Usage(calls=1)
        prompt = getattr(um, "prompt_token_count", 0) or 0
        output = getattr(um, "candidates_token_count", 0) or 0
        thoughts = getattr(um, "thoughts_token_count", 0) or 0
        total = getattr(um, "total_token_count", 0) or (prompt + output + thoughts)
        return Usage(
            prompt_tokens=prompt,
            output_tokens=output,
            thought_tokens=thoughts,
            total_tokens=total,
            calls=1,
        )

    def turn(self, contents: list) -> LLMTurn:
        resp = self._generate_with_retry(contents)
        usage = self._extract_usage(resp)

        text_parts: list[str] = []
        calls: list[FunctionCall] = []
        model_content = None
        finish_reason = None

        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            cand = candidates[0]
            finish_reason = str(getattr(cand, "finish_reason", "") or "")
            model_content = getattr(cand, "content", None)
            parts = getattr(model_content, "parts", None) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    args = dict(getattr(fc, "args", {}) or {})
                    calls.append(FunctionCall(name=fc.name, args=args))
                txt = getattr(part, "text", None)
                if txt:
                    text_parts.append(txt)

        if model_content is None:
            # Synthesize an empty model content so history stays well-formed.
            model_content = self._types.Content(role="model", parts=[])

        return LLMTurn(
            text="\n".join(text_parts) if text_parts else None,
            function_calls=calls,
            usage=usage,
            model_content=model_content,
            finish_reason=finish_reason,
        )

    def _generate_with_retry(self, contents: list):
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=contents, config=self.config
                )
            except Exception as e:  # noqa: BLE001 - normalize provider errors
                last_exc = e
                msg = str(e).lower()
                retryable = any(
                    tok in msg
                    for tok in ("429", "resource_exhausted", "rate", "503", "500",
                                "unavailable", "internal", "deadline", "timeout")
                )
                if not retryable or attempt == self.max_retries - 1:
                    raise
                backoff = DEFAULT_BASE_BACKOFF_S * (2 ** attempt)
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s; backing off %.1fs",
                    attempt + 1, self.max_retries, e, backoff,
                )
                time.sleep(backoff)
        raise last_exc  # pragma: no cover
