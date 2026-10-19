"""Recipient-aware inference facade: authorization, one deadline, usage, validation."""

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import inference_facade
import usage_capture as uc
from contracts_engines import CapabilityHealth, CapabilityRequirement, EngineAdapterKind
from contracts_errors import InternalErrorCode
from contracts_inference import (
    DecisionCandidate,
    DecisionHead,
    DecisionOutcome,
    DecisionRequest,
    GenerationOutcome,
    GenerationRequest,
    InferenceResourceLimits,
    JSON_SCHEMA_DIALECT,
    SourceEvidence,
    SourceSpan,
)
from contracts_processing import ProcessingPolicy
from contracts_references import ReferenceHandle, SourceVersion
from contracts_usage import TokenQuantity, TokenUsage
from engine_ports import (
    ActivationRejected,
    AdapterRegistry,
    OperationContext,
    PortFailure,
    run_decision_conformance,
    run_generation_conformance,
)
from inference_facade import (
    DeadlineExceeded,
    InferenceFacade,
    ProviderFailed,
    ProviderResponse,
    Recipient,
    RecipientDenied,
    StaticProcessingAuthority,
)

V = "candidate-v1"
CTX = OperationContext(tenant_id="tenant-a", correlation_id="c1", deadline_ms=1000)
ATTEMPT = uc.AttemptContext(tenant_id="tenant-a", task_id="task-1", attempt_id="att")
HANDLE = ReferenceHandle(kind="artifact", id="policy-1", tenant_id="tenant-a", resolver="policy")


def policy(*, fallback="within_approved_recipients", epoch=3, recipients=("r1", "r2")):
    return ProcessingPolicy(
        schema_version=V,
        mode="operator_trusted",
        allowed_execution_locations=("external_provider",),
        approved_recipient_ids=tuple(recipients),
        fallback_policy=fallback,
        policy_epoch=epoch,
    )


def usage(input_tokens=10, output_tokens=5):
    na = TokenQuantity(value=None, missing_reason="not_applicable")
    unknown = TokenQuantity(value=None, missing_reason="provider_did_not_report")
    return TokenUsage(
        measurement="actual",
        input_tokens=TokenQuantity(value=input_tokens, missing_reason=None),
        output_tokens=TokenQuantity(value=output_tokens, missing_reason=None),
        cache_read_tokens=unknown,
        cache_write_tokens=na,
        reasoning_tokens=unknown,
        cache_read_inclusion="included_in_input",
        cache_write_inclusion="not_applicable",
        reasoning_inclusion="additional_to_output",
    )


class TransientError(Exception):
    pass


class FakeProvider:
    """Scripted provider; each script item is text, an Exception, or a gate."""

    provider_label = "fake"

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []
        self.lock = threading.Lock()

    def call(self, prompt, *, model, timeout_s):
        with self.lock:
            self.calls.append({"prompt": prompt, "model": model, "timeout_s": timeout_s})
            item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):  # (event, text): block until released
            item[0].wait(5)
            item = item[1]
        return ProviderResponse(text=item, usage=usage(), model="reported-model")


def make(
    *providers,
    authority=None,
    resolve=None,
    recorder=None,
    with_context=True,
    max_retries=0,
    clock=time.monotonic,
    sleep=lambda s: None,
    max_outstanding=8,
    pol=None,
):
    recipients = tuple(
        Recipient(f"r{i + 1}", "external_provider", provider, f"model-{i + 1}")
        for i, provider in enumerate(providers)
    )
    pol = pol or policy()
    return InferenceFacade(
        recipients,
        authority=authority or StaticProcessingAuthority(3, ("r1", "r2")),
        resolve_policy=resolve or (lambda handle: pol),
        recorder=recorder,
        attempt_context=ATTEMPT if with_context else None,
        max_retries=max_retries,
        base_delay_s=1.0,
        is_transient=lambda e: isinstance(e, TransientError),
        clock=clock,
        sleep=sleep,
        max_outstanding=max_outstanding,
    )


def source_parts(text):
    version = SourceVersion(record_id="src-1", content_revision=1, policy_epoch=1)
    length = len(text.encode("utf-8"))
    return (
        (SourceEvidence(source_version=version, text=text),),
        (SourceSpan(span_id="span-1", source_version=version, start_utf8_byte=0, end_utf8_byte=length),),
    )


def limits(**over):
    values = dict(max_total_input_utf8_bytes=100_000, max_total_output_utf8_bytes=100_000, max_attempts=5)
    values.update(over)
    return InferenceResourceLimits(**values)


SCHEMA = {
    "$schema": JSON_SCHEMA_DIALECT,
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


def gen_request(text="prefers dark mode", *, deadline_ms=1000, **over):
    sources, spans = source_parts(text)
    return GenerationRequest(
        schema_version=V, request_id="gen-1", operation="summarize", sources=sources, spans=spans,
        processing_policy=HANDLE, output_schema_dialect=JSON_SCHEMA_DIALECT, output_schema=SCHEMA,
        resource_limits=limits(**over), deadline_ms=deadline_ms,
    )


def dec_request(text="the alpha team", candidates=("alpha", "beta"), *, deadline_ms=1000, **over):
    sources, spans = source_parts(text)
    return DecisionRequest(
        schema_version=V, request_id="dec-1", operation="select", sources=sources, spans=spans,
        candidates=tuple(DecisionCandidate(candidate_id=c) for c in candidates), processing_policy=HANDLE,
        resource_limits=limits(**over), deadline_ms=deadline_ms,
        head=DecisionHead(head_id="head-1", span_ids=("span-1",), candidate_ids=candidates, required=True),
    )


def proposed(summary="dark mode"):
    return json.dumps({"outcome": "proposed", "outputs": [{"summary": summary}]})


@pytest.fixture()
def recorder():
    return uc.InMemoryUsageRecorder()


# ---- authorization ---------------------------------------------------------


def test_authorized_primary_succeeds_and_records_one_event_naming_the_recipient(recorder):
    provider = FakeProvider(proposed())
    result = make(provider, recorder=recorder).generate(CTX, gen_request())
    assert result.outcome is GenerationOutcome.PROPOSED
    assert result.proposals[0].output == {"summary": "dark mode"}
    (event,) = recorder.events
    assert event.attempt_outcome == "completed" and event.status == "final"
    assert event.attribution.recipients == ("r1",)
    assert event.usage.input_tokens.value == 10


def test_unauthorized_primary_is_never_called_and_nothing_is_recorded(recorder):
    provider = FakeProvider(proposed())
    facade = make(provider, authority=StaticProcessingAuthority(3, ("r2",)), recorder=recorder)
    with pytest.raises(RecipientDenied) as raised:
        facade.generate(CTX, gen_request())
    assert provider.calls == [] and recorder.events == []
    assert raised.value.code is InternalErrorCode.DENIED and not raised.value.is_fallback


def test_policy_that_does_not_approve_the_recipient_blocks_the_call():
    provider = FakeProvider(proposed())
    facade = make(provider, pol=policy(recipients=("other",)), authority=StaticProcessingAuthority(3, ("r1",)))
    with pytest.raises(RecipientDenied):
        facade.generate(CTX, gen_request())
    assert provider.calls == []


def test_unauthorized_fallback_is_never_called_and_failure_is_explicit(recorder):
    primary = FakeProvider(TransientError("503 unavailable"))
    fallback = FakeProvider(proposed())
    authority = StaticProcessingAuthority(3, ("r1",))  # r2 approved by policy, not by authority
    facade = make(primary, fallback, authority=authority, recorder=recorder)
    with pytest.raises(RecipientDenied) as raised:
        facade.generate(CTX, gen_request())
    assert raised.value.is_fallback and raised.value.recipient_id == "r2"
    assert len(primary.calls) == 1 and fallback.calls == []
    assert [e.attempt_outcome for e in recorder.events] == ["failed"]


def test_policy_denying_fallback_blocks_it():
    primary = FakeProvider(TransientError("503"))
    fallback = FakeProvider(proposed())
    facade = make(primary, fallback, pol=policy(fallback="deny"))
    with pytest.raises(RecipientDenied, match="fallback is denied"):
        facade.generate(CTX, gen_request())
    assert fallback.calls == []


def test_unreadable_authority_fails_closed():
    class Broken:
        def current_policy_epoch(self, tenant_id):
            raise RuntimeError("store down")

        def authorized_recipient_ids(self, tenant_id):
            return ("r1",)

    provider = FakeProvider(proposed())
    with pytest.raises(RecipientDenied, match="unavailable"):
        make(provider, authority=Broken()).generate(CTX, gen_request())
    assert provider.calls == []


def test_stale_policy_epoch_is_denied():
    provider = FakeProvider(proposed())
    with pytest.raises(RecipientDenied, match="not current"):
        make(provider, authority=StaticProcessingAuthority(4, ("r1", "r2"))).generate(CTX, gen_request())
    assert provider.calls == []


def test_authority_is_rechecked_before_every_retry():
    class Revoking(StaticProcessingAuthority):
        def __init__(self):
            super().__init__(3, ("r1", "r2"))
            self.reads = 0

        def authorized_recipient_ids(self, tenant_id):
            self.reads += 1
            return ("r1", "r2") if self.reads == 1 else ()

    provider = FakeProvider(TransientError("503"), proposed())
    facade = make(provider, authority=Revoking(), max_retries=2)
    with pytest.raises(RecipientDenied):
        facade.generate(CTX, gen_request())
    assert len(provider.calls) == 1  # the retry never left


def test_unresolvable_policy_is_a_denial():
    def resolve(handle):
        raise KeyError("missing")

    provider = FakeProvider(proposed())
    with pytest.raises(RecipientDenied, match="could not be resolved"):
        make(provider, resolve=resolve).generate(CTX, gen_request())
    assert provider.calls == []


# ---- fallback, retries, deadline -------------------------------------------


def test_authorized_fallback_after_transient_failure_records_both_attempts(recorder):
    primary = FakeProvider(TransientError("503 unavailable"))
    fallback = FakeProvider(proposed())
    result = make(primary, fallback, recorder=recorder).generate(CTX, gen_request())
    assert result.outcome is GenerationOutcome.PROPOSED
    assert [e.attempt_outcome for e in recorder.events] == ["failed", "completed"]
    assert [e.attribution.recipients for e in recorder.events] == [("r1",), ("r2",)]
    assert len({e.attempt_id for e in recorder.events}) == 2
    assert recorder.events[0].status == "unavailable"


def test_non_transient_error_is_raised_without_fallback(recorder):
    primary = FakeProvider(ValueError("bad request"))
    fallback = FakeProvider(proposed())
    result = make(primary, fallback, recorder=recorder).generate(CTX, gen_request())
    assert result.outcome is GenerationOutcome.ERROR and result.reason_code == "provider_error"
    assert fallback.calls == []
    with pytest.raises(ProviderFailed) as raised:  # typed; the provider's own error is never re-raised
        make(FakeProvider(ValueError("bad request")), fallback).complete_text(
            tenant_id="tenant-a", prompt="p", policy=policy(), operation="op", deadline_s=1
        )
    assert raised.value.cause_type == "ValueError" and "bad request" not in str(raised.value)


def test_one_deadline_bounds_retries_and_fallbacks():
    now = [0.0]
    primary = FakeProvider(TransientError("503"))
    fallback = FakeProvider(TransientError("503"))

    def sleep(seconds):
        now[0] += seconds

    facade = make(primary, fallback, max_retries=5, clock=lambda: now[0], sleep=sleep)
    with pytest.raises(DeadlineExceeded):
        facade.complete_text(tenant_id="tenant-a", prompt="p", policy=policy(), operation="op", deadline_s=2.5)
    timeouts = [c["timeout_s"] for c in primary.calls + fallback.calls]
    assert max(timeouts) <= 2.5
    assert timeouts == sorted(timeouts, reverse=True)  # the remaining budget only shrinks
    assert len(primary.calls) + len(fallback.calls) < 12  # not 6 attempts per recipient


def test_deadline_exhaustion_stops_waiting_never_calls_fallback_and_never_applies_late_text(recorder):
    gate = threading.Event()
    primary = FakeProvider((gate, proposed("LATE")))
    fallback = FakeProvider(proposed("fallback"))
    facade = make(primary, fallback, recorder=recorder)
    result = facade.generate(CTX, gen_request(deadline_ms=100))
    assert result.outcome is GenerationOutcome.ERROR and result.reason_code == "deadline_exceeded"
    assert result.proposals == ()
    assert fallback.calls == []
    assert [a.late for a in result.attempts] == [True]
    assert [e.attempt_outcome for e in recorder.events] == ["cancelled"]
    assert recorder.events[0].late_after_cancellation is False  # nothing reported yet

    gate.set()  # the provider answers after the deadline
    deadline = time.time() + 5
    while len(recorder.events) < 2 and time.time() < deadline:
        time.sleep(0.01)
    late = recorder.events[1]
    assert late.late_after_cancellation is True and late.status == "final"
    assert late.attempt_outcome == "cancelled" and late.usage.input_tokens.value == 10
    assert late.attempt_id == recorder.events[0].attempt_id
    assert result.proposals == ()  # the late text was never applied


def test_exhausted_outstanding_slots_refuse_instead_of_queueing():
    gate = threading.Event()
    try:
        provider = FakeProvider((gate, proposed()))
        facade = make(provider, max_outstanding=1)
        facade.generate(CTX, gen_request(deadline_ms=50))  # times out, call still running
        calls_before = len(provider.calls)
        with pytest.raises(ProviderFailed, match="outstanding"):
            facade.generate(CTX, gen_request(deadline_ms=1000))
        assert len(provider.calls) == calls_before
    finally:
        gate.set()


def test_attempt_limit_and_input_limit_stop_before_sending(recorder):
    primary = FakeProvider(TransientError("503"))
    fallback = FakeProvider(proposed())
    result = make(primary, fallback, recorder=recorder).generate(CTX, gen_request(max_attempts=1))
    assert result.reason_code == "attempt_limit" and fallback.calls == []
    tiny = make(FakeProvider(proposed()))
    result = tiny.generate(CTX, gen_request(max_total_input_utf8_bytes=5))
    assert result.reason_code == "input_limit_exceeded" and result.attempts == ()


# ---- generation validation and empty vs abstain ----------------------------


def test_schema_invalid_output_is_rejected_with_no_proposal():
    bad = json.dumps({"outcome": "proposed", "outputs": [{"summary": 7}]})
    result = make(FakeProvider(bad)).generate(CTX, gen_request())
    assert result.outcome is GenerationOutcome.ERROR and result.reason_code == "schema_invalid"
    assert result.proposals == ()


@pytest.mark.parametrize(
    "text",
    ["not json", "[1, 2]", json.dumps({"outcome": "proposed", "outputs": []}), json.dumps({"outcome": "wat"})],
)
def test_malformed_model_output_is_an_error(text):
    result = make(FakeProvider(text)).generate(CTX, gen_request())
    assert result.outcome is GenerationOutcome.ERROR and result.reason_code == "invalid_output"


def test_output_over_limit_is_an_error():
    result = make(FakeProvider(proposed("x" * 500))).generate(CTX, gen_request(max_total_output_utf8_bytes=50))
    assert result.reason_code == "output_limit_exceeded" and result.proposals == ()


def test_generation_empty_is_distinct_from_abstained():
    empty = make(FakeProvider(json.dumps({"outcome": "empty"}))).generate(CTX, gen_request())
    abstained = make(FakeProvider(json.dumps({"outcome": "abstained", "reason": "unsupported language"}))).generate(
        CTX, gen_request()
    )
    assert empty.outcome is GenerationOutcome.EMPTY and empty.reason_code is None
    assert abstained.outcome is GenerationOutcome.ABSTAINED and abstained.reason_code == "unsupported_language"


def test_code_fenced_json_is_accepted():
    result = make(FakeProvider("```json\n" + proposed() + "\n```")).generate(CTX, gen_request())
    assert result.outcome is GenerationOutcome.PROPOSED


# ---- decisions ---------------------------------------------------------------


def test_decision_selects_only_supplied_candidates():
    ok = make(FakeProvider(json.dumps({"outcome": "selected", "selected": ["alpha"]}))).decide(CTX, dec_request())
    assert ok.head_result.outcome is DecisionOutcome.SELECTED and ok.head_result.selected_candidate_ids == ("alpha",)
    foreign = make(FakeProvider(json.dumps({"outcome": "selected", "selected": ["gamma"]}))).decide(CTX, dec_request())
    assert foreign.head_result.outcome is DecisionOutcome.ERROR
    assert foreign.head_result.reason_code == "invalid_candidate"
    assert foreign.head_result.selected_candidate_ids == ()


def test_decision_empty_is_distinct_from_abstained():
    empty = make(FakeProvider(json.dumps({"outcome": "empty"}))).decide(CTX, dec_request())
    abstained = make(FakeProvider(json.dumps({"outcome": "abstained", "reason": "cannot tell"}))).decide(
        CTX, dec_request()
    )
    assert empty.head_result.outcome is DecisionOutcome.EMPTY and empty.head_result.reason_code is None
    assert abstained.head_result.outcome is DecisionOutcome.ABSTAINED
    assert abstained.head_result.reason_code == "cannot_tell"


def test_decision_denial_and_deadline_are_typed():
    with pytest.raises(RecipientDenied):
        make(FakeProvider("{}"), authority=StaticProcessingAuthority(3, ())).decide(CTX, dec_request())
    gate = threading.Event()
    try:
        result = make(FakeProvider((gate, "{}"))).decide(CTX, dec_request(deadline_ms=50))
        assert result.head_result.outcome is DecisionOutcome.ERROR
        assert result.head_result.reason_code == "deadline_exceeded"
    finally:
        gate.set()


# ---- usage seam --------------------------------------------------------------


def test_no_context_or_recorder_means_no_events_and_no_failure():
    assert make(FakeProvider(proposed()), with_context=False, recorder=uc.InMemoryUsageRecorder()).generate(
        CTX, gen_request()
    ).outcome is GenerationOutcome.PROPOSED
    assert make(FakeProvider(proposed()), recorder=None).generate(CTX, gen_request()).outcome is GenerationOutcome.PROPOSED


def test_a_failing_recorder_never_changes_the_outcome():
    broken = SimpleNamespace(record=MagicMock(side_effect=RuntimeError("ledger down")))
    result = make(FakeProvider(proposed()), recorder=broken).generate(CTX, gen_request())
    assert result.outcome is GenerationOutcome.PROPOSED
    broken.record.assert_called_once()


# ---- ports, registry, conformance ---------------------------------------------


class RuleProvider:
    """Deterministic stand-in for a model that follows the reference rules."""

    provider_label = "rules"

    def call(self, prompt, *, model, timeout_s):
        body = prompt.split("[span-1] ", 1)[1]
        if "Candidates:" in prompt:
            text = body.split("\nCandidates:", 1)[0]
            candidates = json.loads(prompt.split("Candidates: ", 1)[1].split("\n", 1)[0])
            if not text.strip():
                out = {"outcome": "abstained", "reason": "unsupported_input"}
            else:
                chosen = [c for c in candidates if c.lower() in text.lower()]
                out = {"outcome": "selected", "selected": chosen} if chosen else {"outcome": "empty"}
        else:
            text = body.split("\nReply with JSON only", 1)[0]
            if not text.strip():
                out = {"outcome": "abstained", "reason": "unsupported_input"}
            elif text.strip().startswith("#"):
                out = {"outcome": "empty"}
            else:
                out = {"outcome": "proposed", "outputs": [{"summary": text.strip()}]}
        return ProviderResponse(text=json.dumps(out), usage=usage(), model="m")


def test_facade_passes_the_decision_and_generation_conformance_suites():
    assert run_decision_conformance(lambda: make(RuleProvider())) == ()
    assert run_generation_conformance(lambda: make(RuleProvider())) == ()


def test_facade_registers_and_activates_only_with_a_live_health_probe():
    registry = AdapterRegistry()
    unprobed = make(RuleProvider())
    adapter_id = registry.register(unprobed, EngineAdapterKind.INFERENCE)
    requirements = tuple(
        CapabilityRequirement(
            operation=name, contract_schema_version=V, require_configured=True, require_healthy=True,
            qualification_profile_reference=None, qualification_profile_version=None,
        )
        for name in ("decide", "generate")
    )
    with pytest.raises(ActivationRejected) as raised:
        registry.activate(adapter_id, requirements)  # health is unknown, never assumed
    assert {v.fact for v in raised.value.violations} == {"healthy"}

    probed = make(RuleProvider())
    probed._implementation_id = "probed"
    probed._health_probe = lambda: True
    registry.register(probed, EngineAdapterKind.INFERENCE)
    assert registry.activate("probed", requirements).adapter_id == "probed"
    assert all(s.qualification is None for s in probed.manifest().operations)


# ---- service wiring ------------------------------------------------------------


class FakeGenaiClient:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.models = SimpleNamespace(generate_content=self._generate)
        self.calls = []

    def _generate(self, model=None, contents=None, config=None):
        self.calls.append(model)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        meta = SimpleNamespace(
            prompt_token_count=11, candidates_token_count=4,
            cached_content_token_count=None, thoughts_token_count=None,
        )
        return SimpleNamespace(text=outcome, usage_metadata=meta, model_version="m-1")


@pytest.fixture()
def service(monkeypatch):
    import config

    inference_facade.configure(None)
    monkeypatch.setattr(config.settings, "inference_facade_enabled", True)
    monkeypatch.setattr(config.settings, "inference_static_authority_enabled", True)
    monkeypatch.setattr(config.settings, "llm_max_retries", 0)
    previous = uc.set_usage_recorder(uc.InMemoryUsageRecorder())
    yield uc.get_usage_recorder()
    uc.set_usage_recorder(previous)
    inference_facade.configure(None)


def _memory_service(client):
    from memory_service import MemoryService

    svc = MemoryService.__new__(MemoryService)
    svc._genai_model = client
    return svc


def test_write_path_uses_the_facade_only_when_enabled(service):
    client = FakeGenaiClient('{"facts": []}')
    with uc.attempt_context(ATTEMPT):
        assert _memory_service(client).extract_facts_only("doc", user_id="u1") == []
    (event,) = service.events
    assert event.operation == "llm_extraction_ingest"
    assert event.attribution.recipients is not None and event.attribution.recipients[0].startswith("gemini:")
    assert event.usage.input_tokens.value == 11


def test_write_path_default_is_unchanged(monkeypatch):
    import config

    assert config.Settings().inference_facade_enabled is False
    calls = []
    monkeypatch.setattr(inference_facade, "service_complete_text", lambda **kw: calls.append(kw))
    client = FakeGenaiClient('{"facts": []}')
    previous = uc.set_usage_recorder(uc.InMemoryUsageRecorder())
    try:
        with uc.attempt_context(ATTEMPT):
            _memory_service(client).extract_facts_only("doc", user_id="u1")
        events = uc.get_usage_recorder().events
    finally:
        uc.set_usage_recorder(previous)
    assert calls == [] and len(client.calls) == 1
    assert [e.operation for e in events] == ["llm_extraction_ingest"]
    assert events[0].attribution.recipients is None  # the legacy path never named one


def test_write_path_denial_is_explicit_and_sends_nothing(service):
    inference_facade.configure(StaticProcessingAuthority(0, ()))
    client = FakeGenaiClient('{"facts": []}')
    with uc.attempt_context(ATTEMPT):
        # Ingestion stays fail-open for provider errors, but a denial is explicit.
        with pytest.raises(RecipientDenied):
            _memory_service(client).extract_facts_only("doc", user_id="u1")
    assert client.calls == []
    assert service.events == []


def test_conversation_path_denial_propagates_as_a_loud_failure(service):
    inference_facade.configure(StaticProcessingAuthority(0, ()))
    client = FakeGenaiClient('{"facts": []}')
    svc = _memory_service(client)
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    with pytest.raises(RecipientDenied):
        svc.extract_and_store([{"role": "user", "content": "hi"}], user_id="u1")
    assert client.calls == []


def test_service_facade_falls_back_to_the_configured_fallback_model(service):
    import config

    client = FakeGenaiClient(Exception("503 UNAVAILABLE: model overloaded"), '{"facts": []}')
    with uc.attempt_context(ATTEMPT):
        text = inference_facade.service_complete_text(
            client=client, tenant_id="u1", prompt="p", operation="llm_extraction", settings=config.settings
        )
    assert text == '{"facts": []}'
    assert client.calls == [config.settings.gemini_llm_model, config.settings.gemini_llm_fallback_model]
    assert [e.attempt_outcome for e in service.events] == ["failed", "completed"]
    assert [e.attribution.recipients for e in service.events] == [
        (f"gemini:{config.settings.gemini_llm_model}",),
        (f"gemini:{config.settings.gemini_llm_fallback_model}",),
    ]


def test_service_fallback_denied_by_policy_is_not_called(service, monkeypatch):
    import config

    monkeypatch.setattr(config.settings, "inference_fallback_policy", "deny")
    client = FakeGenaiClient(Exception("503 UNAVAILABLE"), '{"facts": []}')
    with uc.attempt_context(ATTEMPT):
        with pytest.raises(RecipientDenied, match="fallback is denied"):
            inference_facade.service_complete_text(
                client=client, tenant_id="u1", prompt="p", operation="llm_extraction", settings=config.settings
            )
    assert client.calls == [config.settings.gemini_llm_model]


def test_concurrent_service_calls_keep_their_own_usage_attribution(service):
    import config

    client = FakeGenaiClient('{"facts": []}')
    contexts = [uc.AttemptContext(tenant_id=f"t{i}", task_id=f"task{i}", attempt_id=f"a{i}") for i in range(6)]

    def run(context):
        with uc.attempt_context(context):
            inference_facade.service_complete_text(
                client=client, tenant_id=context.tenant_id, prompt="p", operation="op", settings=config.settings
            )

    threads = [threading.Thread(target=run, args=(c,)) for c in contexts]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(e.tenant_id for e in service.events) == sorted(c.tenant_id for c in contexts)
    assert sorted(e.task_id for e in service.events) == sorted(c.task_id for c in contexts)


# ---- review fixes: authority, endpoint labelling, provider error text -----------


def _service_client():
    return FakeGenaiClient('{"facts": []}')


def test_config_built_facade_denies_without_an_installed_authority(monkeypatch):
    import config

    inference_facade.configure(None)
    monkeypatch.setattr(config.settings, "inference_facade_enabled", True)
    monkeypatch.setattr(config.settings, "llm_max_retries", 0)
    assert config.settings.inference_static_authority_enabled is False
    client = _service_client()
    with pytest.raises(RecipientDenied, match="authority is unavailable"):
        inference_facade.service_complete_text(
            client=client, tenant_id="u1", prompt="p", operation="op", settings=config.settings
        )
    assert client.calls == []
    inference_facade.configure(None)


def test_installed_authority_can_revoke_a_recipient_the_config_approves(monkeypatch):
    import config

    monkeypatch.setattr(config.settings, "llm_max_retries", 0)
    ids = inference_facade._configured_recipient_ids(config.settings)
    inference_facade.configure(StaticProcessingAuthority(config.settings.inference_policy_epoch, ids))
    client = _service_client()
    inference_facade.service_complete_text(
        client=client, tenant_id="u1", prompt="p", operation="op", settings=config.settings
    )
    assert len(client.calls) == 1
    inference_facade.configure(StaticProcessingAuthority(config.settings.inference_policy_epoch, ()))
    with pytest.raises(RecipientDenied):
        inference_facade.service_complete_text(
            client=client, tenant_id="u1", prompt="p", operation="op", settings=config.settings
        )
    assert len(client.calls) == 1
    inference_facade.configure(None)


def test_static_authority_is_an_explicit_opt_in_and_warns_once(monkeypatch, caplog):
    import config

    inference_facade.configure(None)
    monkeypatch.setattr(inference_facade, "_static_authority_warned", False)
    monkeypatch.setattr(config.settings, "inference_static_authority_enabled", True)
    monkeypatch.setattr(config.settings, "llm_max_retries", 0)
    client = _service_client()
    with caplog.at_level("WARNING"):
        for _ in range(2):
            inference_facade.service_complete_text(
                client=client, tenant_id="u1", prompt="p", operation="op", settings=config.settings
            )
    assert len(client.calls) == 2
    assert sum("static processing authority" in r.getMessage() for r in caplog.records) == 1
    inference_facade.configure(None)


def test_changed_settings_rebuild_the_cached_service_facade(monkeypatch):
    import config

    monkeypatch.setattr(config.settings, "inference_static_authority_enabled", True)
    monkeypatch.setattr(config.settings, "llm_max_retries", 0)
    inference_facade.configure(None)
    client = _service_client()
    inference_facade.service_complete_text(
        client=client, tenant_id="u1", prompt="p", operation="op", settings=config.settings
    )
    monkeypatch.setattr(config.settings, "inference_approved_recipient_ids", ["gemini:somewhere-else"])
    with pytest.raises(RecipientDenied):  # the policy now names a recipient the facade does not route to
        inference_facade.service_complete_text(
            client=client, tenant_id="u1", prompt="p", operation="op", settings=config.settings
        )
    inference_facade.configure(None)


def test_endpoint_location_is_rejected_for_any_provider_backed_recipient():
    with pytest.raises(ValueError, match="endpoint"):
        Recipient("r1", "endpoint", FakeProvider("x"), "m")  # type: ignore[arg-type]


def test_a_remote_endpoint_override_never_relabels_the_execution_location(monkeypatch):
    import config

    seen = []
    original = inference_facade.InferenceFacade

    class Spy(original):
        def __init__(self, recipients, **kw):
            seen.extend(r.execution_location for r in recipients)
            super().__init__(recipients, **kw)

    monkeypatch.setattr(inference_facade, "InferenceFacade", Spy)
    monkeypatch.setattr(config.settings, "ns_gemini_base_url", "http://127.0.0.1:9/gemini")
    monkeypatch.setattr(config.settings, "inference_static_authority_enabled", True)
    monkeypatch.setattr(config.settings, "llm_max_retries", 0)
    inference_facade.configure(None)
    inference_facade.service_complete_text(
        client=_service_client(), tenant_id="u1", prompt="p", operation="op", settings=config.settings
    )
    assert seen and set(seen) == {"external_provider"}
    inference_facade.configure(None)


def test_non_transient_provider_error_text_never_reaches_the_caller():
    secret = "sk-LIVE-API-KEY-0123456789"
    provider = FakeProvider(RuntimeError(f"401 for key {secret} while sending PROMPT-BODY-TEXT"))
    facade = make(provider)
    with pytest.raises(ProviderFailed) as raised:
        facade.complete_text(tenant_id="tenant-a", prompt="PROMPT-BODY-TEXT", policy=policy(), operation="op", deadline_s=2)
    error = raised.value
    assert secret not in str(error) and "PROMPT-BODY" not in str(error)
    assert error.__cause__ is None and error.__context__ is None
    assert getattr(error, "cause", None) is None
    assert error.cause_type == "RuntimeError"
