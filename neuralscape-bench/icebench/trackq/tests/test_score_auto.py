"""AR4: ns-auto Track-Q scoring — dispatch each op to the SERVED engine's parser.

The `ns-auto` pseudo-system auto-selects the best engine per op, so a single run's
answers come from different engines. The scorer reads the served engine from each
answer's `served_by` (AR3 attribution) and delegates to that engine's format
parser. These tests pin that dispatch — including the health-fallback case where
the served engine is NOT the measured winner.
"""

from icebench.trackq.score import (
    _resolve_auto_system,
    _parse_symbol_set,
    normalize_answer,
)


def test_resolve_auto_system_from_served_by():
    ans = {"served_by": "code-graphify-lib", "status": "ok"}
    assert _resolve_auto_system("ns-auto", ans) == "ns-graphify-lib"
    ans = {"served_by": "code-native", "status": "ok"}
    assert _resolve_auto_system("ns-auto", ans) == "ns-native"


def test_resolve_auto_system_from_envelope_system_field():
    """Falls back to the response envelope's `system` field when served_by absent."""
    ans = {"text": '{"result":"x","system":"code-native"}', "status": "ok"}
    assert _resolve_auto_system("ns-auto", ans) == "ns-native"


def test_resolve_auto_system_noop_for_real_systems():
    ans = {"served_by": "code-native"}
    assert _resolve_auto_system("ns-native", ans) == "ns-native"
    assert _resolve_auto_system("ns-cbm", ans) == "ns-cbm"


def test_auto_symbol_lookup_parses_via_native():
    """auto symbol_lookup served by native → native's symbol parser applies."""
    answer = {
        "text": '{"result":"Code graph search results for: get_best_encoding\\n\\n'
        'src.click._compat.get_best_encoding (function) in src/click/_compat.py:42\\n",'
        '"system":"code-native","routed_by":"auto"}',
        "status": "ok",
        "served_by": "code-native",
    }
    assert normalize_answer(answer, system="ns-auto", for_symbol_lookup=True) == (
        "src/click/_compat.py",
        "get_best_encoding",
    )


def test_auto_neighbors_parses_via_graphify_lib():
    """auto neighbors served by graphify-lib → graphify-lib's neighbor parser."""
    answer = {
        "text": (
            '{"result":"Neighbors of _NonClosingTextIOWrapper:\\n'
            "  -- _compat.py [contains] [EXTRACTED]\\n"
            "  -- _make_text_stream() [calls] [EXTRACTED]\\n"
            '  -- BinaryIO [references] [EXTRACTED]",'
            '"system":"code-graphify-lib","routed_by":"auto"}'
        ),
        "status": "ok",
        "served_by": "code-graphify-lib",
    }
    symbols = _parse_symbol_set(answer, system="ns-auto")
    assert "_compat.py" not in symbols  # containment edge dropped
    assert "_make_text_stream" in symbols
    assert "binaryio" in symbols


def test_auto_health_fallback_parses_via_fallback_engine():
    """Health fallback: neighbors' top engine (graphify) down → cbm served.

    The scorer must parse cbm's FQN-arrow rendering, NOT graphify's node-label
    form — driven purely by the attribution, so the fallback is scored honestly.
    """
    answer = {
        "text": (
            '{"result":"Neighbors of \'click.core.Command\':\\n'
            '  --> click.core.Command.invoke [CALLS]","system":"code-cbm","routed_by":"auto"}'
        ),
        "status": "ok",
        "served_by": "code-cbm",
    }
    symbols = _parse_symbol_set(answer, system="ns-auto")
    # cbm uses the ns-ice/ns-cbm arrow parser → the invoke callee is picked up.
    assert any("invoke" in s for s in symbols)


def test_auto_error_answer_is_empty():
    answer = {"error": "boom", "status": "error", "served_by": "code-native"}
    assert normalize_answer(answer, system="ns-auto", for_symbol_lookup=True) is None
    assert _parse_symbol_set(answer, system="ns-auto") == set()
