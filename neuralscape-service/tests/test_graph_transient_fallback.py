"""Regression tests for the Gemini transient-error fallback in the graphiti subtree.

Root cause we guard against: ``GeminiClient._generate_response`` used to re-raise
with ``raise Exception from e``, producing a *message-less* wrapper whose 503/
overload text survived only in ``__cause__``. ``_is_transient_error`` inspected
only ``str(exc)`` (empty), so it returned False and the fallback-model failover
to the GA model never fired — a transient 503 silently dropped graph enrichment.

The fix is twofold: preserve the message on re-raise, AND have
``_is_transient_error`` walk the cause/context chain as defense-in-depth.
"""

from graphiti_core.llm_client.gemini_client import _TRANSIENT_PATTERNS, _is_transient_error


def test_detects_503_in_top_level_message():
    assert _is_transient_error(Exception("503 UNAVAILABLE. high demand"))


def test_detects_503_buried_in_cause_when_wrapper_message_empty():
    """The exact original failure: empty wrapper, 503 only in __cause__."""
    try:
        try:
            raise Exception(
                "503 UNAVAILABLE. {'error': {'message': 'high demand'}}"
            )
        except Exception as e:
            raise Exception() from e  # message-less wrapper (the old bug shape)
    except Exception as wrapped:
        assert str(wrapped) == ""  # confirm the wrapper itself carries no text
        assert _is_transient_error(wrapped)  # chain-walk still catches it


def test_detects_transient_in_context_chain():
    """Implicit chaining (__context__) is walked too."""
    try:
        try:
            raise Exception("model is overloaded")
        except Exception:
            raise RuntimeError()  # no explicit `from`, sets __context__
    except RuntimeError as wrapped:
        assert _is_transient_error(wrapped)


def test_non_transient_error_not_flagged():
    assert not _is_transient_error(Exception("400 invalid request: bad schema"))


def test_chain_walk_terminates_on_cycle():
    """A self-referential cause must not loop forever."""
    e = Exception("nothing transient here")
    e.__cause__ = e
    assert _is_transient_error(e) is False


def test_patterns_cover_known_transient_signals():
    for p in ("503", "unavailable", "overloaded", "high demand"):
        assert p in _TRANSIENT_PATTERNS
