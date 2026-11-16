"""Service-ledger usage capture seam for provider generation responses.

Normalizes provider-reported token counters into :class:`TokenUsage` and hands
a content-free :class:`UsageEvent` to an optional recorder.  Nothing is
recorded unless a recorder is installed *and* the caller has bound an
:class:`AttemptContext`, so existing behavior is unchanged by default.

Capture is strictly best-effort: no failure here ever propagates into the
calling write path.  An absent or malformed counter is reported as missing
(``provider_did_not_report``), never as zero; only an integer the provider
actually supplied becomes a measured value (including a genuine ``0``).

Gemini conventions: ``prompt_token_count`` already includes cached tokens,
``candidates_token_count`` excludes thought tokens (which are additional to
output), and implicit caching reports no cache-write count.  The raw counter
names are retained in :data:`GEMINI_COUNTER_NAMES`.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, Protocol

from contracts_common import MAX_SAFE_INTEGER
from contracts_usage import (
    AttributionSnapshot,
    TokenQuantity,
    TokenUsage,
    UsageEvent,
)

logger = logging.getLogger(__name__)

GEMINI_COUNTER_NAMES = {
    "input_tokens": "prompt_token_count",
    "output_tokens": "candidates_token_count",
    "cache_read_tokens": "cached_content_token_count",
    "reasoning_tokens": "thoughts_token_count",
}


@dataclass(frozen=True)
class AttemptContext:
    """Caller-supplied correlation for one attempt (opaque, content-free)."""

    tenant_id: str
    task_id: str
    attempt_id: str
    session_id: str | None = None
    intent_id: str | None = None
    bill_owner: str | None = None
    ledger_attempt: int | None = None


class UsageRecorder(Protocol):
    def record(self, event: UsageEvent) -> None: ...


class InMemoryUsageRecorder:
    """Test/diagnostic recorder; keeps events in arrival order."""

    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    def record(self, event: UsageEvent) -> None:
        self.events.append(event)


_recorder: UsageRecorder | None = None


class _AttemptState:
    """The bound context plus a counter minting one attempt id per provider call.

    Threads started from the bound context share this object, so concurrent
    calls still receive distinct ids.
    """

    def __init__(self, context: AttemptContext) -> None:
        self.context = context
        self._counter = itertools.count(1)

    def next_attempt_id(self) -> str:
        return _bounded_id(f"{self.context.attempt_id}.{next(self._counter)}")


_attempt: ContextVar[_AttemptState | None] = ContextVar("usage_attempt", default=None)

_ID_LIMIT = 256
_ID_PREFIX = 200


def _bounded_id(value: str) -> str:
    """Return ``value`` if it fits an OpaqueId, else a deterministic bounded form.

    The bounded form keeps a readable prefix of the raw value plus a digest of
    the whole value, so distinct long values stay distinct.
    """
    if value and len(value) <= _ID_LIMIT:
        return value
    digest = hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:16]
    return f"{value[:_ID_PREFIX]}~{digest}"


_operation: ContextVar[str | None] = ContextVar("usage_operation", default=None)

ProviderCallKind = Literal["generation", "embedding"]


def current_attempt_context() -> AttemptContext | None:
    """The context bound in the calling task/thread, if any."""
    state = _attempt.get()
    return state.context if state is not None else None


def get_usage_recorder() -> UsageRecorder | None:
    return _recorder


@contextmanager
def operation_scope(operation: str) -> Iterator[None]:
    """Label provider calls made in this context (threads started from it inherit it)."""
    token = _operation.set(operation)
    try:
        yield
    finally:
        _operation.reset(token)


def current_operation(default: str) -> str:
    return _operation.get() or default


def set_usage_recorder(recorder: UsageRecorder | None) -> UsageRecorder | None:
    """Install ``recorder`` (``None`` disables capture); return the previous one."""
    global _recorder
    previous, _recorder = _recorder, recorder
    return previous


@contextmanager
def attempt_context(context: AttemptContext) -> Iterator[None]:
    token = _attempt.set(_AttemptState(context))
    try:
        yield
    finally:
        _attempt.reset(token)


UNATTRIBUTED_TENANT = "unattributed"


@contextmanager
def ensure_attempt(
    tenant_id: str | None, operation: str, *, task_id: str | None = None, bill_owner: str | None = None
) -> Iterator[None]:
    """Bind an attempt context for a service entry point unless the caller already bound one.

    The service's own entry points (search, ask, an ARQ job) call this so the provider calls made under them
    are attributed to the person they serve.  A no-op unless a recorder is installed, and a context a ledger
    stage bound already wins (it carries the ledger attempt).  The ids are opaque and content-free; a missing
    person is recorded under :data:`UNATTRIBUTED_TENANT` rather than dropped.  Never raises.
    """
    token = None
    try:
        if _recorder is not None and _attempt.get() is None:
            task = _bounded_id(task_id or uuid.uuid4().hex)
            context = AttemptContext(
                tenant_id=_bounded_id(tenant_id or UNATTRIBUTED_TENANT),
                task_id=task,
                attempt_id=_bounded_id(f"{task}:{operation}"),
                bill_owner=bill_owner,
            )
            token = _attempt.set(_AttemptState(context))
    except Exception:
        logger.warning("usage attempt context could not be bound for %s", operation, exc_info=True)
    try:
        yield
    finally:
        if token is not None:
            _attempt.reset(token)


def attributed(operation: str, user_arg: str = "user_id") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator form of :func:`ensure_attempt` for a synchronous service method taking ``user_arg``.

    With no recorder installed the wrapper is one ``None`` check and a call.
    """
    import functools
    import inspect

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _recorder is None:
                return fn(*args, **kwargs)
            tenant = None
            try:
                value = signature.bind_partial(*args, **kwargs).arguments.get(user_arg)
                tenant = value if isinstance(value, str) else None
            except Exception:
                pass
            with ensure_attempt(tenant, operation):
                return fn(*args, **kwargs)

        return wrapper

    return decorate


def bind_job_attempt(tenant_id: str | None, task_id: str, try_number: int | None = None) -> bool:
    """Bind an attempt context to the calling task for the rest of its life (an ARQ job); never raises.

    Used from the worker's ``on_job_start`` hook, which runs in the job's own asyncio task; pair it with
    :func:`unbind_job_attempt` in ``after_job_end``.  Returns whether a context was bound.
    """
    try:
        if _recorder is None or not task_id:
            return False
        task = _bounded_id(str(task_id))
        context = AttemptContext(
            tenant_id=_bounded_id(tenant_id or UNATTRIBUTED_TENANT),
            task_id=task,
            attempt_id=_bounded_id(f"{task}:try{try_number or 1}"),
        )
        _attempt.set(_AttemptState(context))
        return True
    except Exception:
        logger.warning("usage job context could not be bound for %s", task_id, exc_info=True)
        return False


def unbind_job_attempt() -> None:
    _attempt.set(None)


def _counter(value: Any) -> TokenQuantity:
    if type(value) is int and 0 <= value <= MAX_SAFE_INTEGER:
        return TokenQuantity(value=value, missing_reason=None)
    return TokenQuantity(value=None, missing_reason="provider_did_not_report")


def normalize_gemini_usage(metadata: Any) -> TokenUsage | None:
    """Map Gemini ``usage_metadata`` to :class:`TokenUsage`; ``None`` if absent."""
    if metadata is None:
        return None
    fields = {
        key: _counter(getattr(metadata, raw, None))
        for key, raw in GEMINI_COUNTER_NAMES.items()
    }
    return TokenUsage(
        measurement="actual",
        cache_write_tokens=TokenQuantity(value=None, missing_reason="not_applicable"),
        cache_read_inclusion="included_in_input",
        cache_write_inclusion="not_applicable",
        reasoning_inclusion="additional_to_output",
        **fields,
    )


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def normalize_openai_usage(usage: Any, kind: ProviderCallKind = "generation") -> TokenUsage | None:
    """Map OpenAI/gateway-shaped ``usage`` to :class:`TokenUsage`; ``None`` if absent.

    Chat shape (``prompt_tokens``/``completion_tokens``) and responses shape
    (``input_tokens``/``output_tokens``) are accepted.  Cached tokens are a
    subset of input and reasoning tokens a subset of output, so neither is
    added again; ``total_tokens`` is deliberately not used.  Embeddings have
    no output, cache, or reasoning categories.
    """
    if usage is None:
        return None
    na = TokenQuantity(value=None, missing_reason="not_applicable")
    input_value = _field(usage, "prompt_tokens")
    if input_value is None:
        input_value = _field(usage, "input_tokens")
    if kind == "embedding":
        return TokenUsage(
            measurement="actual",
            input_tokens=_counter(input_value),
            output_tokens=na,
            cache_read_tokens=na,
            cache_write_tokens=na,
            reasoning_tokens=na,
            cache_read_inclusion="not_applicable",
            cache_write_inclusion="not_applicable",
            reasoning_inclusion="not_applicable",
        )
    output_value = _field(usage, "completion_tokens")
    if output_value is None:
        output_value = _field(usage, "output_tokens")
    prompt_details = _field(usage, "prompt_tokens_details")
    if prompt_details is None:
        prompt_details = _field(usage, "input_tokens_details")
    completion_details = _field(usage, "completion_tokens_details")
    if completion_details is None:
        completion_details = _field(usage, "output_tokens_details")
    return TokenUsage(
        measurement="actual",
        input_tokens=_counter(input_value),
        output_tokens=_counter(output_value),
        cache_read_tokens=_counter(_field(prompt_details, "cached_tokens")),
        cache_write_tokens=na,
        reasoning_tokens=_counter(_field(completion_details, "reasoning_tokens")),
        cache_read_inclusion="included_in_input",
        cache_write_inclusion="not_applicable",
        reasoning_inclusion="included_in_output",
    )


def _known(quantity: TokenQuantity) -> int | None:
    return quantity.value


def _reconcile_counters(usage: TokenUsage | None, kind: ProviderCallKind) -> TokenUsage | None:
    """Drop all-unreported usage and mark internally inconsistent subcounts missing."""
    if usage is None:
        return None
    base = (usage.input_tokens, usage.output_tokens) if kind == "generation" else (usage.input_tokens,)
    categories = base + (
        (usage.cache_read_tokens, usage.reasoning_tokens) if kind == "generation" else ()
    )
    if all(_known(q) is None for q in categories):
        return None
    if kind != "generation":
        return usage
    unreported = TokenQuantity(value=None, missing_reason="provider_did_not_report")
    updates: dict[str, TokenQuantity] = {}
    cache, inp = _known(usage.cache_read_tokens), _known(usage.input_tokens)
    if (
        usage.cache_read_inclusion == "included_in_input"
        and cache is not None
        and inp is not None
        and cache > inp
    ):
        updates["cache_read_tokens"] = unreported
    reasoning, out = _known(usage.reasoning_tokens), _known(usage.output_tokens)
    if (
        usage.reasoning_inclusion == "included_in_output"
        and reasoning is not None
        and out is not None
        and reasoning > out
    ):
        updates["reasoning_tokens"] = unreported
    return usage.model_copy(update=updates) if updates else usage


def _normalize_response(
    response: Any, kind: ProviderCallKind
) -> tuple[TokenUsage | None, str, str | None]:
    """Return (usage, provider label, provider-reported model) for either shape."""
    reported = getattr(response, "model_version", None) or getattr(response, "model", None)
    reported = reported if isinstance(reported, str) and reported else None
    if getattr(response, "usage_metadata", None) is not None:
        usage = normalize_gemini_usage(response.usage_metadata)
        if kind == "embedding" and usage is not None:
            usage = _embedding_usage_from_gemini(usage)
        return _reconcile_counters(usage, kind), "google", reported
    openai_usage = _field(response, "usage")
    if openai_usage is not None:
        usage = normalize_openai_usage(openai_usage, kind)
        return _reconcile_counters(usage, kind), "openai_compatible", reported
    gemini_shaped = any(
        hasattr(response, name) for name in ("model_version", "candidates", "embeddings")
    )
    provider = "google" if gemini_shaped else "openai_compatible"
    return None, provider, reported


def _embedding_usage_from_gemini(usage: TokenUsage) -> TokenUsage:
    na = TokenQuantity(value=None, missing_reason="not_applicable")
    return TokenUsage(
        measurement="actual",
        input_tokens=usage.input_tokens,
        output_tokens=na,
        cache_read_tokens=na,
        cache_write_tokens=na,
        reasoning_tokens=na,
        cache_read_inclusion="not_applicable",
        cache_write_inclusion="not_applicable",
        reasoning_inclusion="not_applicable",
    )


def _attribution(
    producer: str,
    provider: str,
    model: str | None,
    context: AttemptContext,
) -> AttributionSnapshot:
    model_value = _bounded_id(model) if isinstance(model, str) and model else None
    return AttributionSnapshot(
        producer=_bounded_id(producer),
        bill_owner=context.bill_owner,
        bill_owner_missing_reason=(
            None if context.bill_owner else "dependency_unavailable"
        ),
        execution_location="provider_hosted",
        provider=_bounded_id(provider),
        provider_missing_reason=None,
        model=model_value,
        model_missing_reason=None if model_value else "provider_did_not_report",
        recipients=None,
        recipients_missing_reason="dependency_unavailable",
    )


def _event(
    state: _AttemptState,
    operation: str,
    attribution: AttributionSnapshot,
    usage: TokenUsage | None,
    outcome: str,
    missing_reason: str,
    *,
    attempt_id: str | None = None,
    predecessor: str | None = None,
    pending: bool = False,
) -> UsageEvent:
    context = state.context
    return UsageEvent(
        schema_version="candidate-v1",
        event_id=uuid.uuid4().hex,
        tenant_id=context.tenant_id,
        task_id=context.task_id,
        session_id=context.session_id,
        intent_id=context.intent_id,
        attempt_id=attempt_id or state.next_attempt_id(),
        ledger="service",
        operation=_bounded_id(operation),
        predecessor_event_id=predecessor,
        status="pending" if pending else ("final" if usage is not None else "unavailable"),
        attempt_outcome="in_progress" if pending else outcome,
        late_after_cancellation=False,
        attribution=attribution,
        usage=usage,
        usage_missing_reason=None if (usage is not None or pending) else missing_reason,
    )


def _emit(
    *,
    operation: str,
    model: str | None,
    kind: ProviderCallKind,
    response: Any = None,
    outcome: str = "completed",
    fallback: bool = False,
    attempt_id: str | None = None,
    predecessor: str | None = None,
    pending: bool = False,
) -> str | None:
    """Build and record one event and return its id; best-effort, never raises."""
    recorder = _recorder
    if recorder is None:
        return None
    try:
        state = _attempt.get()
        if state is None:
            return None
        producer = f"{operation}.fallback" if fallback else operation
        missing = "not_yet_reported" if outcome == "cancelled" else "provider_did_not_report"
        try:
            usage, provider, reported_model = (
                _normalize_response(response, kind) if response is not None else (None, "unknown", None)
            )
            event = _event(
                state,
                operation,
                _attribution(producer, provider, reported_model or model, state.context),
                usage,
                outcome,
                missing,
                attempt_id=attempt_id,
                predecessor=predecessor,
                pending=pending,
            )
        except Exception:
            # Provider-supplied values must never drop the attempt: record it
            # with no usage and no model rather than losing it.
            logger.warning("usage event degraded for %s", operation, exc_info=True)
            event = _event(
                state,
                operation,
                _attribution(producer, "unknown", None, state.context),
                None,
                outcome,
                missing,
                attempt_id=attempt_id,
                predecessor=predecessor,
                pending=pending,
            )
        if recorder.record(event) is False:
            return None  # not durable: nothing may name it as a predecessor
        return event.event_id
    except Exception:
        logger.warning("usage capture failed for %s", operation, exc_info=True)
        return None


# One-slot memory of the response already recorded per-attempt by the retry
# seam, so the caller-side hook does not record it a second time.
_attempt_recorded: ContextVar[Any] = ContextVar("usage_attempt_recorded", default=None)


def record_provider_usage(
    response: Any,
    *,
    operation: str,
    model: str | None,
    kind: ProviderCallKind = "generation",
) -> None:
    """Record one completed provider call (generation or embedding); never raises."""
    if response is not None and _attempt_recorded.get() is response:
        _attempt_recorded.set(None)  # the duplicate is consumed: do not keep the response alive
        return
    _emit(operation=operation, model=model, kind=kind, response=response)


def record_failed_provider_attempt(
    error: BaseException,
    *,
    operation: str,
    model: str | None,
    kind: ProviderCallKind = "generation",
) -> None:
    """Record one provider call that raised (failed or cancelled); never raises."""
    cancelled = isinstance(error, asyncio.CancelledError)
    _emit(
        operation=operation,
        model=model,
        kind=kind,
        outcome="cancelled" if cancelled else "failed",
    )


def record_generation_usage(response: Any, *, operation: str, model: str | None) -> None:
    """Record one completed generation attempt; never raises."""
    record_provider_usage(response, operation=operation, model=model, kind="generation")


_PROVIDER_CALL_NAMES = frozenset({"generate_content"})


def _operation_label(text: str) -> str:
    label = "".join(c if c.isalnum() else "_" for c in text.lower())
    return "_".join(part for part in label.split("_") if part) or "provider_call"


class ProviderAttempt:
    """Observes one attempt made by the retry helper; methods never raise."""

    def __init__(self, operation: str, model: str | None, fallback: bool) -> None:
        self._operation = operation
        self._model = model
        self._fallback = fallback
        self._attempt_id: str | None = None
        self._pending_id: str | None = None

    def begin(self) -> None:
        """Durably note that the call is about to be made (durable recorders only).

        If the process dies during the provider call the pending event is all
        that survives: the attempt stays visible with unknown usage instead of
        vanishing, and the retry is a separate attempt.
        """

        try:
            recorder = _recorder
            state = _attempt.get()
            if recorder is None or state is None or not getattr(recorder, "records_pending", False):
                return
            self._attempt_id = state.next_attempt_id()
            self._pending_id = _emit(
                operation=self._operation,
                model=self._model,
                kind="generation",
                fallback=self._fallback,
                attempt_id=self._attempt_id,
                pending=True,
            )
            if self._pending_id is None:
                self._attempt_id = None
        except Exception:
            self._attempt_id = self._pending_id = None

    def succeeded(self, response: Any) -> None:
        _emit(
            operation=self._operation,
            model=self._model,
            kind="generation",
            response=response,
            fallback=self._fallback,
            attempt_id=self._attempt_id,
            predecessor=self._pending_id,
        )
        try:
            _attempt_recorded.set(response)
        except Exception:
            pass

    def failed(self, error: BaseException) -> None:
        cancelled = isinstance(error, asyncio.CancelledError)
        _emit(
            operation=self._operation,
            model=self._model,
            kind="generation",
            outcome="cancelled" if cancelled else "failed",
            fallback=self._fallback,
            attempt_id=self._attempt_id,
            predecessor=self._pending_id,
        )


def start_provider_attempt(
    fn: Callable[..., Any], operation: str, model: Any, fallback: bool
) -> ProviderAttempt | None:
    """Return an observer for one retry-helper attempt, or ``None`` when disabled.

    Disabled unless a recorder is installed, an attempt context is bound and
    ``fn`` is a provider generation call; so graph and other retried work is
    never reported as model usage.
    """
    try:
        if _recorder is None or _attempt.get() is None:
            return None
        if getattr(fn, "__name__", None) not in _PROVIDER_CALL_NAMES:
            return None
        attempt = ProviderAttempt(
            _operation_label(operation), model if isinstance(model, str) else None, fallback
        )
        attempt.begin()  # a pending event first, for durable recorders only
        return attempt
    except Exception:
        return None


class DurableUsageRecorder:
    """Recorder that persists each event to a usage store.

    ``ledger_attempt`` (the lifecycle ledger's claim attempt) is taken from the
    bound :class:`AttemptContext` when supplied.  Writes are fail-open (a store
    error never reaches the calling path) but not silent:

    * a failed write is retried a few times (a transient store error is the
      common case) and then reported as not stored by returning ``False``, so
      capture seams never name an event that is not in the store as a
      predecessor;
    * every event that could not be stored is counted per task and persisted as
      a gap by :meth:`flush_gaps`, so a rollup reports the task as incomplete
      rather than complete or missing.  Gaps wait in memory until the store takes
      them: they are lost if the process dies first.
    """

    #: Tells capture seams to write a pending event before each provider call.
    records_pending = True

    #: Most (tenant, task) gap counters held in memory while the store is down; more are only counted.
    MAX_UNRECORDED_KEYS = 1_000

    #: Longest the retries of one event may take (the call runs on the provider-call thread).
    RECORD_BUDGET_S = 2.5

    def __init__(
        self,
        store: Any,
        *,
        retries: int = 2,
        retry_delay: float = 0.05,
        sleep: Any = None,
        max_unrecorded_keys: int | None = None,
        record_budget_s: float | None = None,
        monotonic: Any = None,
    ) -> None:
        import threading
        import time

        self._store = store
        self._retries = max(0, int(retries))
        self._retry_delay = retry_delay
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic
        self._max_keys = self.MAX_UNRECORDED_KEYS if max_unrecorded_keys is None else int(max_unrecorded_keys)
        self._budget_s = self.RECORD_BUDGET_S if record_budget_s is None else float(record_budget_s)
        self._lock = threading.Lock()
        self._unrecorded: dict[tuple[str, str], int] = {}
        self.failures = 0
        #: Events that failed to store whose task could not even be remembered (the counter map was full).
        self.gaps_overflowed = 0

    @property
    def unrecorded(self) -> dict[tuple[str, str], int]:
        """Events not yet persisted, nor persisted as a gap, by (tenant, task)."""
        with self._lock:
            return dict(self._unrecorded)

    def record(self, event: UsageEvent) -> bool:
        """Store ``event``; ``False`` when it could not be stored (never raises)."""
        from usage_store import UsageEventConflict

        context = current_attempt_context()
        ledger_attempt = context.ledger_attempt if context else None
        started = self._monotonic()
        for attempt in range(self._retries + 1):
            try:
                self._store.append(event, ledger_attempt=ledger_attempt)
            except (UsageEventConflict, ValueError, TypeError):
                break  # not transient: retrying cannot help
            except Exception:
                # The provider-call thread never waits longer than the budget on a sick store.
                if attempt < self._retries and self._monotonic() - started < self._budget_s:
                    self._sleep(self._retry_delay * (attempt + 1))
                    continue
                break
            else:
                if self._unrecorded:
                    self.flush_gaps()  # the store is taking writes again
                return True
        self.failures += 1
        key = (event.tenant_id, event.task_id)
        with self._lock:
            if key in self._unrecorded or len(self._unrecorded) < self._max_keys:
                self._unrecorded[key] = self._unrecorded.get(key, 0) + 1
            else:
                # The store has been down long enough to fill the map: keep the memory bounded and
                # count the drop (an operator sees ``gaps_overflowed`` and the error logs).
                self.gaps_overflowed += 1
        logger.error("durable usage write failed for %s", event.event_id, exc_info=True)
        return False

    def flush_gaps(self) -> None:
        """Persist the per-task counts of events that could not be stored (best effort)."""
        with self._lock:
            # Claim the counts so a concurrent flush cannot persist them twice.
            pending = self._unrecorded
            self._unrecorded = {}
        for (tenant_id, task_id), count in pending.items():
            try:
                self._store.record_gap(tenant_id, task_id, count)
            except Exception:
                logger.error("usage gap for task %s could not be stored", task_id, exc_info=True)
                with self._lock:
                    key = (tenant_id, task_id)
                    if key in self._unrecorded or len(self._unrecorded) < self._max_keys:
                        self._unrecorded[key] = self._unrecorded.get(key, 0) + count
                    else:
                        self.gaps_overflowed += count
