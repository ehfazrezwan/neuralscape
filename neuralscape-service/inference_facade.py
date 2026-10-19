"""Recipient-aware inference facade over the existing Gemini and gateway clients.

Implements ``DecisionPort`` and ``GenerationPort`` (and a plain text completion
used by the extraction write path) with these guarantees:

* **Every attempt is authorized first.**  Before any content leaves, the target
  recipient is checked against the request's ``ProcessingPolicy`` *and* current
  processing authority (``validate_plaintext_dispatch``), for the primary and
  for every fallback and retry.  A denied recipient is never called and raises
  ``RecipientDenied``; a denied fallback is not silently replaced by another
  recipient, and an unreadable authority fails closed.
* **One absolute deadline** is computed once and shared by the primary, its
  retries and all fallbacks.  Each provider call runs in a worker thread so the
  facade can stop waiting.  The declared cancellation guarantee is therefore
  *stops waiting and fences publication*: the provider call itself is not
  interrupted.  A response that arrives after the deadline is recorded as late
  usage and is never applied.
* **Per-attempt usage** (completed, failed, cancelled and late) is handed to the
  service usage recorder as content-free ``UsageEvent`` values naming the
  recipient.  Recording is best-effort and never changes the outcome.
* **Generation output is schema validated** through ``contracts_inference``; an
  invalid result is an ``ERROR`` outcome with no proposals.  A valid empty
  outcome is distinct from an abstention.
* Outstanding provider calls are bounded; when the bound is reached the call is
  refused rather than queued.

Nothing here is imported by the service unless ``INFERENCE_FACADE_ENABLED`` is
set; ``memory/write.py`` routes its two extraction calls through
``complete_text`` only then.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass
from typing import Any, Callable, Collection, NamedTuple, Optional, Protocol

from contracts_errors import InternalErrorCode
from contracts_inference import (
    AttemptUsage,
    DecisionHeadResult,
    DecisionOutcome,
    DecisionRequest,
    DecisionResult,
    GenerationOutcome,
    GenerationProposal,
    GenerationRequest,
    GenerationResult,
    validate_decision_result,
    validate_generation_result,
)
from contracts_processing import (
    ExecutionLocation,
    PlaintextDispatchRejected,
    ProcessingPolicy,
    validate_plaintext_dispatch,
)
from contracts_references import ReferenceHandle
from contracts_usage import AttributionSnapshot, TokenUsage, UsageEvent
from engine_ports import CONTRACT_VERSION, OperationContext, PortFailure
from contracts_engines import (
    CapabilityHealth,
    CapabilityManifest,
    CapabilityOperationState,
    EngineAdapterKind,
)

logger = logging.getLogger(__name__)

FACADE_VERSION = "inference-facade-v1"
_UNAVAILABLE = InternalErrorCode.UNAVAILABLE_DEPENDENCY


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class RecipientDenied(PortFailure):
    """No content was sent: policy or current authority refused the recipient."""

    def __init__(self, recipient_id: str, reason: str, *, is_fallback: bool) -> None:
        super().__init__(InternalErrorCode.DENIED, reason)
        self.args = (f"denied: {reason}",)
        self.recipient_id = recipient_id
        self.is_fallback = is_fallback


class DeadlineExceeded(PortFailure):
    """The shared deadline was spent; later attempts were not made."""

    def __init__(self, detail: str = "deadline exceeded") -> None:
        super().__init__(_UNAVAILABLE, detail)
        self.args = (f"unavailable: {detail}",)


class ProviderFailed(PortFailure):
    """Every permitted attempt failed.

    Provider and driver error text can carry credentials and prompt content, so
    only the *type* of the last error is kept; the error itself is never chained
    or stored on this exception.
    """

    def __init__(self, detail: str, cause: Optional[BaseException] = None) -> None:
        super().__init__(_UNAVAILABLE, detail)
        self.args = (f"unavailable: {detail}",)
        self.cause = None
        self.cause_type = type(cause).__name__ if cause is not None else None


# --------------------------------------------------------------------------
# Providers and authority
# --------------------------------------------------------------------------


class ProviderResponse(NamedTuple):
    text: str
    usage: Optional[TokenUsage]
    model: Optional[str]


class Provider(Protocol):
    def call(self, prompt: str, *, model: str, timeout_s: float) -> ProviderResponse: ...


class GeminiProvider:
    """Plain text completion through a ``google.genai`` client."""

    provider_label = "google"

    def __init__(self, client: Any) -> None:
        self._client = client

    def call(self, prompt: str, *, model: str, timeout_s: float) -> ProviderResponse:
        from google.genai.types import GenerateContentConfig, HttpOptions

        from usage_capture import normalize_gemini_usage

        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=GenerateContentConfig(
                http_options=HttpOptions(timeout=max(1, int(timeout_s * 1000)))
            ),
        )
        reported = getattr(response, "model_version", None)
        return ProviderResponse(
            text=response.text or "",
            usage=normalize_gemini_usage(getattr(response, "usage_metadata", None)),
            model=reported if isinstance(reported, str) and reported else None,
        )


class OpenAICompatibleProvider:
    """Chat completion through an OpenAI-compatible gateway client."""

    provider_label = "openai_compatible"

    def __init__(self, client: Any) -> None:
        self._client = client

    def call(self, prompt: str, *, model: str, timeout_s: float) -> ProviderResponse:
        from usage_capture import normalize_openai_usage

        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout_s,
        )
        choices = getattr(response, "choices", None) or []
        text = ""
        if choices:
            text = getattr(getattr(choices[0], "message", None), "content", "") or ""
        usage = getattr(response, "usage", None)
        reported = getattr(response, "model", None)
        return ProviderResponse(
            text=text,
            usage=normalize_openai_usage(usage) if usage is not None else None,
            model=reported if isinstance(reported, str) and reported else None,
        )


@dataclass(frozen=True)
class Recipient:
    """One routable destination: the first is primary, the rest are fallbacks."""

    recipient_id: str
    execution_location: ExecutionLocation
    provider: Any
    model: str

    def __post_init__(self) -> None:
        # ``endpoint`` means the client's own device.  A facade that calls a
        # provider is never that, and the location label is never trusted to
        # skip recipient approval, so it cannot be claimed here.
        if self.execution_location == "endpoint":
            raise ValueError(
                "a provider-backed recipient cannot declare the endpoint execution location"
            )


class ProcessingAuthority(Protocol):
    """Current processing authority, read fresh for every attempt."""

    def current_policy_epoch(self, tenant_id: str) -> int: ...

    def authorized_recipient_ids(self, tenant_id: str) -> Collection[str]: ...


class AuthorityUnavailable(Exception):
    """Authority could not be read; the facade fails closed."""


class StaticProcessingAuthority:
    """Fixed authority for tests and for config-driven deployments.

    It cannot deny anything its own configuration allows, so the service only
    uses it when ``INFERENCE_STATIC_AUTHORITY_ENABLED`` is set explicitly; a real
    authority source must be supplied through ``configure`` before relying on
    the dispatch check.
    """

    def __init__(self, epoch: int, recipient_ids: Collection[str]) -> None:
        self._epoch = epoch
        self._recipients = frozenset(recipient_ids)

    def current_policy_epoch(self, tenant_id: str) -> int:
        return self._epoch

    def authorized_recipient_ids(self, tenant_id: str) -> Collection[str]:
        return self._recipients


class UnconfiguredProcessingAuthority:
    """Placeholder installed when no authority source exists: every read fails closed."""

    def current_policy_epoch(self, tenant_id: str) -> int:
        raise AuthorityUnavailable("no processing authority is configured")

    def authorized_recipient_ids(self, tenant_id: str) -> Collection[str]:
        raise AuthorityUnavailable("no processing authority is configured")


# --------------------------------------------------------------------------
# Facade
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _AttemptRecord:
    attempt_id: str
    recipient_id: str
    model: str
    input_bytes: int
    output_bytes: int
    outcome: str  # completed | failed | cancelled
    late: bool = False


class _Outcome(NamedTuple):
    text: Optional[str]
    attempts: tuple[_AttemptRecord, ...]
    failure: Optional[str]  # None | deadline_exceeded | provider_error | provider_unavailable | attempt_limit
    error: Optional[BaseException]
    policy_epoch: int


def _default_is_transient(error: BaseException) -> bool:
    from memory.retry import _is_transient

    return _is_transient(error)  # type: ignore[arg-type]


def _utf8(text: str) -> int:
    return len(text.encode("utf-8", errors="replace"))


class InferenceFacade:
    """Decision and generation ports with recipient authorization and one deadline."""

    def __init__(
        self,
        recipients: tuple[Recipient, ...],
        *,
        authority: ProcessingAuthority,
        resolve_policy: Callable[[ReferenceHandle], ProcessingPolicy],
        recorder: Any = None,
        attempt_context: Any = None,
        implementation_id: str = "inference-facade",
        max_retries: int = 2,
        base_delay_s: float = 1.0,
        max_outstanding: int = 8,
        is_transient: Optional[Callable[[BaseException], bool]] = None,
        health_probe: Optional[Callable[[], bool]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not recipients:
            raise ValueError("at least one recipient is required")
        if len({r.recipient_id for r in recipients}) != len(recipients):
            raise ValueError("recipient ids must be unique")
        self._recipients = recipients
        self._authority = authority
        self._resolve_policy = resolve_policy
        self._recorder = recorder
        self._attempt_context = attempt_context
        self._implementation_id = implementation_id
        self._max_retries = max(0, max_retries)
        self._base_delay_s = base_delay_s
        self._is_transient = is_transient or _default_is_transient
        self._health_probe = health_probe
        self._clock = clock
        self._sleep = sleep
        self._slots = threading.BoundedSemaphore(max(1, max_outstanding))
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_outstanding), thread_name_prefix="ns-inference"
        )
        self._counter = 0
        self._counter_lock = threading.Lock()

    # -- capability ---------------------------------------------------------

    def manifest(self) -> CapabilityManifest:
        health = CapabilityHealth.UNKNOWN
        if self._health_probe is not None:
            try:
                health = (
                    CapabilityHealth.HEALTHY if self._health_probe() else CapabilityHealth.UNHEALTHY
                )
            except Exception:
                health = CapabilityHealth.UNHEALTHY
        return CapabilityManifest(
            schema_version=CONTRACT_VERSION,
            implementation_id=self._implementation_id,
            implementation_version=FACADE_VERSION,
            adapter_kind=EngineAdapterKind.INFERENCE,
            contract_schema_versions=(CONTRACT_VERSION,),
            operations=tuple(
                CapabilityOperationState(
                    operation=name,
                    supported=True,
                    configured=True,
                    health=health,
                    qualification=None,
                )
                for name in ("decide", "generate")
            ),
        )

    # -- usage ----------------------------------------------------------------

    def _next_attempt_id(self, base: str) -> str:
        with self._counter_lock:
            self._counter += 1
            return f"{base}-a{self._counter}"

    def _emit(
        self,
        *,
        attempt_id: str,
        operation: str,
        recipient: Recipient,
        usage: Optional[TokenUsage],
        reported_model: Optional[str],
        outcome: str,
        sink: tuple[Any, Any],
        late: bool = False,
        predecessor: Optional[str] = None,
        pending: bool = False,
    ) -> Optional[str]:
        """Hand one content-free event to the recorder and return its id; never raises.

        ``pending`` writes the "call is about to be made" event (durable
        recorders only); the call's outcome event then names it as predecessor.
        """
        recorder, context = sink
        if recorder is None or context is None:
            return None
        if pending and not getattr(recorder, "records_pending", False):
            return None
        try:
            missing = "not_yet_reported" if outcome == "cancelled" else "provider_did_not_report"
            provider = getattr(recipient.provider, "provider_label", None)
            model = reported_model or recipient.model
            event = UsageEvent(
                schema_version=CONTRACT_VERSION,
                event_id=uuid.uuid4().hex,
                tenant_id=context.tenant_id,
                task_id=context.task_id,
                session_id=context.session_id,
                intent_id=context.intent_id,
                attempt_id=attempt_id[:256],
                ledger="service",
                operation=operation[:256],
                predecessor_event_id=predecessor,
                status="pending" if pending else ("final" if usage is not None else "unavailable"),
                attempt_outcome="in_progress" if pending else outcome,
                late_after_cancellation=late,
                attribution=AttributionSnapshot(
                    producer=f"{operation}.facade"[:256],
                    bill_owner=context.bill_owner,
                    bill_owner_missing_reason=None if context.bill_owner else "dependency_unavailable",
                    execution_location=recipient.execution_location,
                    provider=provider,
                    provider_missing_reason=None if provider else "provider_did_not_report",
                    model=model[:256] if model else None,
                    model_missing_reason=None if model else "provider_did_not_report",
                    recipients=(recipient.recipient_id,),
                    recipients_missing_reason=None,
                ),
                usage=usage,
                usage_missing_reason=None if (usage is not None or pending) else missing,
            )
            # The recorder may run on a provider thread (late responses); give it
            # the bound attempt context so the ledger attempt travels with the event.
            import usage_capture

            with usage_capture.attempt_context(context):
                stored = recorder.record(event)
            return None if stored is False else event.event_id
        except Exception:
            logger.warning("inference usage capture failed for %s", operation, exc_info=True)
            return None

    # -- dispatch guard -------------------------------------------------------

    def _authorize(
        self, tenant_id: str, policy: ProcessingPolicy, recipient: Recipient, *, is_fallback: bool
    ) -> int:
        """Raise ``RecipientDenied`` unless policy and current authority agree."""
        try:
            epoch = self._authority.current_policy_epoch(tenant_id)
            authorized = self._authority.authorized_recipient_ids(tenant_id)
        except Exception as error:
            raise RecipientDenied(
                recipient.recipient_id, "processing authority is unavailable", is_fallback=is_fallback
            ) from error
        try:
            validate_plaintext_dispatch(
                policy,
                execution_location=recipient.execution_location,
                recipient_id=recipient.recipient_id,
                current_policy_epoch=epoch,
                currently_authorized_recipient_ids=authorized,
                is_fallback=is_fallback,
            )
        except PlaintextDispatchRejected as error:
            raise RecipientDenied(recipient.recipient_id, str(error), is_fallback=is_fallback) from error
        return epoch

    # -- core attempt loop ------------------------------------------------------

    def _run(
        self,
        *,
        tenant_id: str,
        policy: ProcessingPolicy,
        prompt: str,
        operation: str,
        deadline_ms: int,
        max_attempts: Optional[int],
        max_input_bytes: Optional[int],
        attempt_base: str,
        sink: Optional[tuple[Any, Any]] = None,
    ) -> _Outcome:
        sink = sink if sink is not None else (self._recorder, self._attempt_context)
        deadline = self._clock() + deadline_ms / 1000.0  # one absolute deadline
        input_bytes = _utf8(prompt)
        if max_input_bytes is not None and input_bytes > max_input_bytes:
            return _Outcome(None, (), "input_limit_exceeded", None, policy.policy_epoch)
        attempts: list[_AttemptRecord] = []
        last_error: Optional[BaseException] = None
        epoch = policy.policy_epoch
        for index, recipient in enumerate(self._recipients):
            is_fallback = index > 0
            for retry in range(self._max_retries + 1):
                if max_attempts is not None and len(attempts) >= max_attempts:
                    return _Outcome(None, tuple(attempts), "attempt_limit", last_error, epoch)
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return _Outcome(None, tuple(attempts), "deadline_exceeded", last_error, epoch)
                # Authorization is evaluated immediately before each send.
                epoch = self._authorize(tenant_id, policy, recipient, is_fallback=is_fallback)
                if not self._slots.acquire(blocking=False):
                    raise ProviderFailed("too many outstanding provider calls")
                attempt_id = self._next_attempt_id(attempt_base)
                pending_id = self._emit(
                    attempt_id=attempt_id, operation=operation, recipient=recipient,
                    usage=None, reported_model=None, outcome="in_progress", sink=sink, pending=True,
                )
                future: Future = self._executor.submit(
                    recipient.provider.call, prompt, model=recipient.model, timeout_s=remaining
                )
                done, _pending = futures_wait([future], timeout=remaining)
                if not done:
                    record = _AttemptRecord(
                        attempt_id, recipient.recipient_id, recipient.model, input_bytes, 0, "cancelled"
                    )
                    attempts.append(record)
                    cancelled_id = self._emit(
                        attempt_id=attempt_id, operation=operation, recipient=recipient,
                        usage=None, reported_model=None, outcome="cancelled", sink=sink,
                        predecessor=pending_id,
                    )
                    # The call keeps running; its late response is only recorded.
                    future.add_done_callback(
                        self._late_handler(attempt_id, operation, recipient, sink, cancelled_id)
                    )
                    return _Outcome(None, tuple(attempts), "deadline_exceeded", last_error, epoch)
                try:
                    response: ProviderResponse = future.result()
                except Exception as error:
                    self._slots.release()
                    last_error = error
                    attempts.append(
                        _AttemptRecord(
                            attempt_id, recipient.recipient_id, recipient.model, input_bytes, 0, "failed"
                        )
                    )
                    self._emit(
                        attempt_id=attempt_id, operation=operation, recipient=recipient,
                        usage=None, reported_model=None, outcome="failed", sink=sink,
                        predecessor=pending_id,
                    )
                    if not self._is_transient(error):
                        return _Outcome(None, tuple(attempts), "provider_error", error, epoch)
                    delay = min(self._base_delay_s * (2**retry), max(0.0, deadline - self._clock()))
                    if retry < self._max_retries and delay > 0:
                        self._sleep(delay)
                    continue
                self._slots.release()
                attempts.append(
                    _AttemptRecord(
                        attempt_id, recipient.recipient_id, recipient.model,
                        input_bytes, _utf8(response.text), "completed",
                    )
                )
                self._emit(
                    attempt_id=attempt_id, operation=operation, recipient=recipient,
                    usage=response.usage, reported_model=response.model, outcome="completed", sink=sink,
                    predecessor=pending_id,
                )
                return _Outcome(response.text, tuple(attempts), None, None, epoch)
        return _Outcome(None, tuple(attempts), "provider_unavailable", last_error, epoch)

    def _late_handler(
        self,
        attempt_id: str,
        operation: str,
        recipient: Recipient,
        sink: tuple[Any, Any],
        predecessor: Optional[str] = None,
    ) -> Callable[[Future], None]:
        def done(future: Future) -> None:
            self._slots.release()
            try:
                response = future.result()
            except BaseException:
                return  # nothing was reported; the cancelled event already stands
            if response.usage is None:
                return
            # Late usage is retained for reconciliation; the text is discarded.
            self._emit(
                attempt_id=attempt_id, operation=operation, recipient=recipient,
                usage=response.usage, reported_model=response.model,
                outcome="cancelled", sink=sink, late=True, predecessor=predecessor,
            )

        return done

    # -- text completion (service extraction path) -----------------------------

    def complete_text(
        self,
        *,
        tenant_id: str,
        prompt: str,
        policy: ProcessingPolicy,
        operation: str,
        deadline_s: float,
        attempt_base: Optional[str] = None,
        sink: Optional[tuple[Any, Any]] = None,
    ) -> str:
        """Return the model text, or raise a typed ``PortFailure``."""
        outcome = self._run(
            tenant_id=tenant_id,
            policy=policy,
            prompt=prompt,
            operation=operation,
            deadline_ms=max(1, int(deadline_s * 1000)),
            max_attempts=None,
            max_input_bytes=None,
            attempt_base=attempt_base or uuid.uuid4().hex,
            sink=sink,
        )
        if outcome.text is not None:
            return outcome.text
        if outcome.failure == "deadline_exceeded":
            raise DeadlineExceeded()
        raise ProviderFailed(outcome.failure or "provider_unavailable", outcome.error) from None

    # -- ports ------------------------------------------------------------------

    @staticmethod
    def _attempt_usages(outcome: _Outcome, policy_version: str) -> tuple[AttemptUsage, ...]:
        return tuple(
            AttemptUsage(
                attempt_id=record.attempt_id[:256],
                model_version=record.model[:256],
                policy_version=policy_version,
                input_utf8_bytes=record.input_bytes,
                output_utf8_bytes=record.output_bytes,
                late=record.outcome == "cancelled",
            )
            for record in outcome.attempts
        )

    def decide(self, ctx: OperationContext, request: DecisionRequest) -> DecisionResult:
        request = DecisionRequest.model_validate(request)
        policy = self._resolve(request.processing_policy)
        spans = {span.span_id: span for span in request.spans}
        sources = {e.source_version: e.text for e in request.sources}
        lines = []
        for span_id in request.head.span_ids:
            span = spans[span_id]
            raw = sources[span.source_version].encode("utf-8")
            lines.append(f"[{span_id}] " + raw[span.start_utf8_byte : span.end_utf8_byte].decode("utf-8", "ignore"))
        prompt = (
            f"Task: {request.operation}\nSource spans:\n" + "\n".join(lines) + "\nCandidates: "
            + json.dumps(list(request.head.candidate_ids)) + "\n"
            'Reply with JSON only: {"outcome": "selected"|"empty"|"abstained", '
            '"selected": [candidate ids], "reason": "short code when abstained"}. '
            "Use empty when no candidate applies and abstained when you cannot decide."
        )
        outcome = self._run(
            tenant_id=ctx.tenant_id, policy=policy, prompt=prompt, operation=request.operation,
            deadline_ms=request.deadline_ms,
            max_attempts=request.resource_limits.max_attempts,
            max_input_bytes=request.resource_limits.max_total_input_utf8_bytes,
            attempt_base=request.request_id,
        )
        head_id = request.head.head_id
        failure, selected, kind = self._decision_head(outcome, request)
        if failure is not None:
            head = DecisionHeadResult(
                head_id=head_id, outcome=DecisionOutcome.ERROR, selected_candidate_ids=(),
                distribution=None, reason_code=failure,
            )
        elif kind == "selected":
            head = DecisionHeadResult(
                head_id=head_id, outcome=DecisionOutcome.SELECTED, selected_candidate_ids=selected,
                distribution=None, reason_code=None,
            )
        elif kind == "empty":
            head = DecisionHeadResult(
                head_id=head_id, outcome=DecisionOutcome.EMPTY, selected_candidate_ids=(),
                distribution=None, reason_code=None,
            )
        else:
            head = DecisionHeadResult(
                head_id=head_id, outcome=DecisionOutcome.ABSTAINED, selected_candidate_ids=(),
                distribution=None, reason_code=selected_reason(selected),
            )
        result = DecisionResult(
            schema_version=CONTRACT_VERSION,
            request_id=request.request_id,
            head_result=head,
            attempts=self._attempt_usages(outcome, f"epoch-{outcome.policy_epoch}"),
        )
        validate_decision_result(request, result)
        return result

    @staticmethod
    def _decision_head(outcome: _Outcome, request: DecisionRequest):
        if outcome.text is None:
            return (outcome.failure or "provider_unavailable"), (), None
        parsed = _parse_json_object(outcome.text)
        if parsed is None:
            return "invalid_output", (), None
        kind = parsed.get("outcome")
        if kind == "selected":
            chosen = parsed.get("selected")
            if (
                not isinstance(chosen, list)
                or not chosen
                or not all(isinstance(c, str) for c in chosen)
                or len(set(chosen)) != len(chosen)
                or not set(chosen) <= set(request.head.candidate_ids)
            ):
                return "invalid_candidate", (), None  # never publish a non-candidate
            return None, tuple(chosen), "selected"
        if kind == "empty":
            return None, (), "empty"
        if kind == "abstained":
            return None, (parsed.get("reason"),), "abstained"
        return "invalid_output", (), None

    def generate(self, ctx: OperationContext, request: GenerationRequest) -> GenerationResult:
        request = GenerationRequest.model_validate(request)
        policy = self._resolve(request.processing_policy)
        texts = {e.source_version: e.text for e in request.sources}
        lines = []
        for span in request.spans:
            raw = texts[span.source_version].encode("utf-8")
            lines.append(f"[{span.span_id}] " + raw[span.start_utf8_byte : span.end_utf8_byte].decode("utf-8", "ignore"))
        prompt = (
            f"Task: {request.operation}\nSource spans:\n" + "\n".join(lines) + "\n"
            "Reply with JSON only: {\"outcome\": \"proposed\"|\"empty\"|\"abstained\", "
            "\"outputs\": [objects matching this JSON schema], \"reason\": \"short code when abstained\"}.\n"
            "Schema: " + json.dumps(request.output_schema) + "\n"
            "Use empty when there is nothing to propose and abstained when you cannot do the task."
        )
        outcome = self._run(
            tenant_id=ctx.tenant_id, policy=policy, prompt=prompt, operation=request.operation,
            deadline_ms=request.deadline_ms,
            max_attempts=request.resource_limits.max_attempts,
            max_input_bytes=request.resource_limits.max_total_input_utf8_bytes,
            attempt_base=request.request_id,
        )
        attempts = self._attempt_usages(outcome, f"epoch-{outcome.policy_epoch}")

        def fail(reason: str) -> GenerationResult:
            return GenerationResult(
                schema_version=CONTRACT_VERSION, request_id=request.request_id,
                outcome=GenerationOutcome.ERROR, proposals=(), reason_code=reason, attempts=attempts,
            )

        if outcome.text is None:
            return fail(outcome.failure or "provider_unavailable")
        if _utf8(outcome.text) > request.resource_limits.max_total_output_utf8_bytes:
            return fail("output_limit_exceeded")
        parsed = _parse_json_object(outcome.text)
        if parsed is None:
            return fail("invalid_output")
        kind = parsed.get("outcome")
        if kind == "empty":
            result = GenerationResult(
                schema_version=CONTRACT_VERSION, request_id=request.request_id,
                outcome=GenerationOutcome.EMPTY, proposals=(), reason_code=None, attempts=attempts,
            )
        elif kind == "abstained":
            result = GenerationResult(
                schema_version=CONTRACT_VERSION, request_id=request.request_id,
                outcome=GenerationOutcome.ABSTAINED, proposals=(),
                reason_code=selected_reason((parsed.get("reason"),)), attempts=attempts,
            )
        elif kind == "proposed":
            outputs = parsed.get("outputs")
            if not isinstance(outputs, list) or not outputs:
                return fail("invalid_output")
            support = tuple(span.span_id for span in request.spans)
            try:
                proposals = tuple(
                    GenerationProposal(
                        proposal_id=f"{request.request_id}-p{i + 1}", output=output,
                        support_span_ids=support, machine_authored=True,
                    )
                    for i, output in enumerate(outputs)
                )
                result = GenerationResult(
                    schema_version=CONTRACT_VERSION, request_id=request.request_id,
                    outcome=GenerationOutcome.PROPOSED, proposals=proposals, reason_code=None,
                    attempts=attempts,
                )
            except ValueError:
                return fail("invalid_output")
        else:
            return fail("invalid_output")
        try:
            validate_generation_result(request, result)
        except ValueError:
            return fail("schema_invalid")  # rejected: no proposal is returned
        return result

    def _resolve(self, handle: ReferenceHandle) -> ProcessingPolicy:
        try:
            return ProcessingPolicy.model_validate(self._resolve_policy(handle))
        except Exception as error:
            raise RecipientDenied("-", "processing policy could not be resolved", is_fallback=False) from error

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def selected_reason(values: tuple[Any, ...]) -> str:
    """Bounded opaque reason code for an abstention."""
    reason = values[0] if values else None
    if isinstance(reason, str) and reason.strip():
        cleaned = "".join(c if c.isalnum() or c in "_-." else "_" for c in reason.strip())
        return cleaned[:64] or "model_abstained"
    return "model_abstained"


def _parse_json_object(text: str) -> Optional[dict]:
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`")
        if body.lower().startswith("json"):
            body = body[4:]
    try:
        value = json.loads(body)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


# --------------------------------------------------------------------------
# Service wiring (only used when INFERENCE_FACADE_ENABLED)
# --------------------------------------------------------------------------

_service_authority: Optional[ProcessingAuthority] = None
_service_facade: Optional[InferenceFacade] = None
_service_facade_key: Optional[tuple] = None
_service_lock = threading.Lock()


def configure(authority: Optional[ProcessingAuthority]) -> None:
    """Install the processing-authority source; resets the cached service facade."""
    global _service_authority, _service_facade, _service_facade_key
    with _service_lock:
        _service_authority = authority
        _service_facade = None
        _service_facade_key = None


def gemini_endpoint(settings: Any) -> Optional[str]:
    """The configured Gemini base URL, or ``None`` for the SDK default."""
    value = getattr(settings, "ns_gemini_base_url", "")
    return value if type(value) is str and value else None


def build_gemini_client(settings: Any, genai_module: Any = None) -> Any:
    """``genai.Client`` honoring ``NS_GEMINI_BASE_URL``; unchanged when unset."""
    if genai_module is None:
        from google import genai as genai_module
    endpoint = gemini_endpoint(settings)
    if endpoint is None:
        return genai_module.Client(api_key=settings.google_api_key)
    from google.genai.types import HttpOptions

    return genai_module.Client(
        api_key=settings.google_api_key, http_options=HttpOptions(base_url=endpoint)
    )


def gemini_recipient_id(settings: Any, model: str) -> str:
    """Recipient id for a Gemini model; names the host when the endpoint is overridden."""
    endpoint = gemini_endpoint(settings)
    if endpoint is None:
        return f"gemini:{model}"
    from urllib.parse import urlsplit

    return f"gemini@{urlsplit(endpoint).netloc}:{model}"


def _configured_recipient_ids(settings: Any) -> tuple[str, ...]:
    configured = tuple(settings.inference_approved_recipient_ids)
    if configured:
        return configured
    ids = [gemini_recipient_id(settings, settings.gemini_llm_model)]
    if settings.gemini_llm_fallback_model != settings.gemini_llm_model:
        ids.append(gemini_recipient_id(settings, settings.gemini_llm_fallback_model))
    return tuple(ids)


def service_policy(settings: Any) -> ProcessingPolicy:
    """The processing policy implied by configuration (operator-trusted routing)."""
    recipients = _configured_recipient_ids(settings)
    return ProcessingPolicy(
        schema_version=CONTRACT_VERSION,
        mode="operator_trusted",
        allowed_execution_locations=("external_provider",),
        approved_recipient_ids=recipients,
        fallback_policy=settings.inference_fallback_policy,
        policy_epoch=settings.inference_policy_epoch,
    )


_static_authority_warned = False


def _service_key(settings: Any) -> tuple:
    """Everything the cached service facade is built from; a change rebuilds it."""
    return (
        settings.gemini_llm_model,
        settings.gemini_llm_fallback_model,
        gemini_endpoint(settings),
        _configured_recipient_ids(settings),
        settings.inference_policy_epoch,
        settings.inference_static_authority_enabled is True,
        settings.llm_max_retries,
        settings.llm_retry_base_delay,
    )


def _service_authority_for(settings: Any) -> ProcessingAuthority:
    """The installed authority, an explicit static one, or one that denies everything."""
    global _static_authority_warned
    if _service_authority is not None:
        return _service_authority
    if settings.inference_static_authority_enabled is True:
        if not _static_authority_warned:
            _static_authority_warned = True
            logger.warning(
                "Using the static processing authority built from configuration: it cannot "
                "revoke or deny anything the processing policy allows. Install a real "
                "authority source with inference_facade.configure()."
            )
        return StaticProcessingAuthority(settings.inference_policy_epoch, _configured_recipient_ids(settings))
    return UnconfiguredProcessingAuthority()


def service_complete_text(
    *, client: Any, tenant_id: str, prompt: str, operation: str, settings: Any
) -> str:
    """Complete ``prompt`` for the extraction write path through the facade."""
    global _service_facade, _service_facade_key
    policy = service_policy(settings)
    key = _service_key(settings)
    with _service_lock:
        facade = _service_facade
        if facade is None or _service_facade_key != key:
            if facade is not None:
                facade.close()
            # The execution location is a property of how the provider is
            # reached, not a caller label: the Gemini client (default or an
            # overridden base URL) always talks to an external provider.
            primary = Recipient(
                gemini_recipient_id(settings, settings.gemini_llm_model), "external_provider",
                GeminiProvider(client), settings.gemini_llm_model,
            )
            recipients = [primary]
            if settings.gemini_llm_fallback_model != settings.gemini_llm_model:
                recipients.append(
                    Recipient(
                        gemini_recipient_id(settings, settings.gemini_llm_fallback_model), "external_provider",
                        GeminiProvider(client), settings.gemini_llm_fallback_model,
                    )
                )
            facade = InferenceFacade(
                tuple(recipients),
                authority=_service_authority_for(settings),
                resolve_policy=lambda _handle: policy,
                max_retries=settings.llm_max_retries,
                base_delay_s=settings.llm_retry_base_delay,
            )
            _service_facade = facade
            _service_facade_key = key
    import usage_capture

    state = usage_capture._attempt.get()  # bound by the caller; None disables capture
    # The sink is per call: concurrent extractions never share attribution.
    sink = (usage_capture.get_usage_recorder(), state.context if state is not None else None)
    return facade.complete_text(
        tenant_id=tenant_id,
        prompt=prompt,
        policy=policy,
        operation=operation,
        deadline_s=settings.inference_deadline_seconds,
        # Provider attempts are named from the bound ledger attempt (one id per
        # call, "-aN" per retry/fallback attempt) so they correlate exactly.
        attempt_base=state.next_attempt_id() if state is not None else None,
        sink=sink,
    )


__all__ = [
    "DeadlineExceeded",
    "GeminiProvider",
    "build_gemini_client",
    "gemini_endpoint",
    "gemini_recipient_id",
    "InferenceFacade",
    "OpenAICompatibleProvider",
    "ProcessingAuthority",
    "ProviderFailed",
    "ProviderResponse",
    "Recipient",
    "RecipientDenied",
    "StaticProcessingAuthority",
    "UnconfiguredProcessingAuthority",
    "configure",
    "service_complete_text",
    "service_policy",
]
