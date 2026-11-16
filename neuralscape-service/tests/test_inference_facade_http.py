"""The inference facade against a provider endpoint reached over real HTTP.

A local stand-in speaks the provider's REST shape (``generateContent``) on a loopback socket, and the facade's
``GeminiProvider`` uses the real ``google.genai`` client pointed at it, so recipient authorization, the single
operation deadline, late usage after cancellation and the empty/abstained distinction are exercised through a
real network client, its own timeout and the facade's real transient-error classification.  The deadline is
shown twice: with a client timeout that outlasts it (only the facade can stop, and late usage is reconciled) and
with the shipped provider (the client is cut off at the deadline, so no late usage can arrive).  The stand-in
records every request it receives, so a recipient that must never be contacted is checked at the wire.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import usage_capture as uc
from contracts_inference import DecisionOutcome, GenerationOutcome
from inference_facade import GeminiProvider, InferenceFacade, Recipient, RecipientDenied, StaticProcessingAuthority
from tests.test_inference_facade import CTX, ATTEMPT, dec_request, gen_request, policy, proposed


class StandIn:
    """Answers ``generateContent`` per model with a scripted behavior; records every request."""

    def __init__(self) -> None:
        self.behaviors: dict[str, dict] = {}
        self.requests: list[str] = []
        self.finished: list[str] = []
        self.lock = threading.Lock()
        stand_in = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                model = self.path.split("/models/")[-1].split(":")[0]
                with stand_in.lock:
                    stand_in.requests.append(model)
                behavior = stand_in.behaviors.get(model, {"text": proposed()})
                time.sleep(behavior.get("delay", 0))
                if "status" in behavior:
                    body = json.dumps({"error": {"code": behavior["status"], "message": "unavailable",
                                                 "status": "UNAVAILABLE"}}).encode()
                    self.send_response(behavior["status"])
                else:
                    body = json.dumps({
                        "candidates": [{"content": {"role": "model", "parts": [{"text": behavior["text"]}]},
                                        "finishReason": "STOP"}],
                        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18},
                        "modelVersion": model,
                    }).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                with stand_in.lock:
                    stand_in.finished.append(model)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stand_in():
    server = StandIn()
    yield server
    server.close()


def http_facade(stand_in, *, authorized=("r1", "r2"), recorder=None, pol=None, provider_class=GeminiProvider):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key="stand-in-key", http_options=types.HttpOptions(base_url=stand_in.url))
    provider = provider_class(client)
    recipients = (
        Recipient("r1", "external_provider", provider, "model-1"),
        Recipient("r2", "external_provider", provider, "model-2"),
    )
    pol = pol or policy()
    return InferenceFacade(
        recipients,
        authority=StaticProcessingAuthority(3, authorized),
        resolve_policy=lambda handle: pol,
        recorder=recorder,
        attempt_context=ATTEMPT if recorder is not None else None,
        max_retries=0,
        base_delay_s=0.01,
    )


def test_an_authorized_primary_answers_over_http_and_usage_is_recorded(stand_in):
    recorder = uc.InMemoryUsageRecorder()
    result = http_facade(stand_in, recorder=recorder).generate(CTX, gen_request())
    assert result.outcome is GenerationOutcome.PROPOSED and result.proposals[0].output == {"summary": "dark mode"}
    assert stand_in.requests == ["model-1"]
    (event,) = recorder.events
    assert event.attribution.recipients == ("r1",) and event.usage.input_tokens.value == 11


def test_a_transient_http_failure_falls_back_only_to_an_authorized_recipient(stand_in):
    stand_in.behaviors["model-1"] = {"status": 503}
    recorder = uc.InMemoryUsageRecorder()
    result = http_facade(stand_in, recorder=recorder).generate(CTX, gen_request())
    assert result.outcome is GenerationOutcome.PROPOSED
    assert stand_in.requests == ["model-1", "model-2"]
    assert [e.attribution.recipients for e in recorder.events] == [("r1",), ("r2",)]

    stand_in.requests.clear()
    with pytest.raises(RecipientDenied) as denied:  # approved by policy, not by the current authority
        http_facade(stand_in, authorized=("r1",)).generate(CTX, gen_request())
    assert denied.value.is_fallback and denied.value.recipient_id == "r2"
    assert stand_in.requests == ["model-1"]  # the unauthorized fallback never received a request

    stand_in.requests.clear()
    with pytest.raises(RecipientDenied, match="fallback is denied"):
        http_facade(stand_in, pol=policy(fallback="deny")).generate(CTX, gen_request())
    assert stand_in.requests == ["model-1"]


class LongTimeoutGeminiProvider(GeminiProvider):
    """The real provider and client, but the client's own HTTP timeout outlasts the operation deadline.

    With the shipped provider the client gives up at the remaining budget, so whether the facade stops at the
    deadline, or the client does, cannot be told apart.  With a 10 s client timeout only the facade can stop the
    wait, and the provider's late answer can reach the facade after the deadline.
    """

    def call(self, prompt: str, *, model: str, timeout_s: float):
        return super().call(prompt, model=model, timeout_s=10.0)


def _wait_for(predicate, seconds=8.0):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    return predicate()


def _deadline_call(stand_in, recorder, *, provider_class=GeminiProvider, delay=3.0):
    stand_in.behaviors["model-1"] = {"text": proposed("LATE"), "delay": delay}
    facade = http_facade(stand_in, recorder=recorder, provider_class=provider_class)
    started = time.monotonic()
    result = facade.generate(CTX, gen_request(deadline_ms=300))
    elapsed = time.monotonic() - started
    assert result.outcome is GenerationOutcome.ERROR and result.reason_code == "deadline_exceeded"
    assert result.proposals == ()  # nothing late is applied
    assert stand_in.requests == ["model-1"]  # no fallback after the deadline
    return elapsed


def test_the_facade_itself_stops_at_the_one_deadline_when_the_client_would_wait_longer(stand_in):
    """A 300 ms deadline against a 3 s provider and a 10 s client timeout: only the facade's own wait can stop."""

    elapsed = _deadline_call(stand_in, uc.InMemoryUsageRecorder(), provider_class=LongTimeoutGeminiProvider)
    assert elapsed < 1.5, elapsed  # the provider answers at 3 s and the client would wait 10 s


def test_late_usage_after_the_deadline_is_reconciled_over_a_real_client(stand_in):
    """The cancelled attempt is recorded at the deadline with usage not yet reported; when the provider's late
    answer arrives (the client still waiting), its real usage is recorded as a successor event marked late after
    cancellation, and its text is discarded."""

    recorder = uc.InMemoryUsageRecorder()
    _deadline_call(stand_in, recorder, provider_class=LongTimeoutGeminiProvider)
    assert _wait_for(lambda: len(recorder.events) >= 2), recorder.events
    cancelled, late = recorder.events
    assert (cancelled.attempt_outcome, cancelled.status, cancelled.usage) == ("cancelled", "unavailable", None)
    assert cancelled.usage_missing_reason == "not_yet_reported" and cancelled.late_after_cancellation is False
    assert (late.attempt_outcome, late.status, late.late_after_cancellation) == ("cancelled", "final", True)
    assert late.predecessor_event_id == cancelled.event_id and late.attempt_id == cancelled.attempt_id
    assert late.usage.input_tokens.value == 11  # the provider's reported usage, reconciled late
    assert stand_in.finished == ["model-1"]


def test_with_the_shipped_provider_the_client_is_cut_off_and_no_late_usage_arrives(stand_in):
    """Pinned gap: the shipped provider sets the client timeout to the remaining budget, so the client discards
    the late answer although the provider finished the work; the attempt stays cancelled with usage not reported
    (never a fabricated zero) and no late event can follow."""

    recorder = uc.InMemoryUsageRecorder()
    elapsed = _deadline_call(stand_in, recorder)
    assert elapsed < 1.5, elapsed
    assert _wait_for(lambda: "model-1" in stand_in.finished)  # the provider did the work
    time.sleep(0.5)  # time for a late event, if one could arrive
    (event,) = recorder.events  # exactly one: no late event follows
    assert (event.status, event.usage, event.late_after_cancellation) == ("unavailable", None, False)
    # The client's timeout and the facade's wait expire together, so which one ends the attempt is a race:
    # the facade records "cancelled / not yet reported" when its wait expires first, "failed / provider did
    # not report" when the client's timeout error arrives first (about one run in six on a loopback host).
    assert (event.attempt_outcome, event.usage_missing_reason) in (
        ("cancelled", "not_yet_reported"), ("failed", "provider_did_not_report"),
    )


def test_valid_empty_and_abstained_stay_distinct_over_http(stand_in):
    stand_in.behaviors["model-1"] = {"text": json.dumps({"outcome": "empty"})}
    empty = http_facade(stand_in).generate(CTX, gen_request())
    stand_in.behaviors["model-1"] = {"text": json.dumps({"outcome": "abstained", "reason": "unsupported language"})}
    abstained = http_facade(stand_in).generate(CTX, gen_request())
    assert empty.outcome is GenerationOutcome.EMPTY and empty.reason_code is None
    assert abstained.outcome is GenerationOutcome.ABSTAINED and abstained.reason_code == "unsupported_language"

    stand_in.behaviors["model-1"] = {"text": json.dumps({"outcome": "empty"})}
    d_empty = http_facade(stand_in).decide(CTX, dec_request())
    stand_in.behaviors["model-1"] = {"text": json.dumps({"outcome": "abstained", "reason": "cannot tell"})}
    d_abstained = http_facade(stand_in).decide(CTX, dec_request())
    assert d_empty.head_result.outcome is DecisionOutcome.EMPTY
    assert d_abstained.head_result.outcome is DecisionOutcome.ABSTAINED
