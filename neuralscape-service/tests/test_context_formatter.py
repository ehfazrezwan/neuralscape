"""Tests for the authoritative-standards injection block in context_formatter.

The server-side formatter (used by GET /v1/context/inject) mirrors the plugin's
session-start rendering: standards are prepended as a BINDING block and exempt
from the ordinary max_chars budget so they are never truncated away.
"""

from context_formatter import (
    format_context_for_injection,
    format_standards_block,
)
from schemas import MemoryResponse


def _mem(text: str) -> MemoryResponse:
    return MemoryResponse(id=text[:8], memory=text)


class TestFormatStandardsBlock:
    def test_empty_when_no_standards(self):
        assert format_standards_block([]) == ""

    def test_renders_binding_header_and_items(self):
        block = format_standards_block([_mem("Always use the Opti deck template")])
        assert "AUTHORITATIVE Standards (binding)" in block
        assert "OVERRIDE" in block
        assert "- Always use the Opti deck template" in block

    def test_oversized_standard_not_dropped(self):
        # A single directive larger than any reserved budget must still be emitted
        # in full — never a header-only block (CR: don't drop an oversized standard).
        long_rule = "y" * 5000
        block = format_standards_block([_mem(long_rule)], max_chars=100)
        assert long_rule in block

    def test_all_standards_emitted_no_truncation(self):
        rules = [_mem(f"Rule number {i}") for i in range(50)]
        block = format_standards_block(rules, max_chars=100)
        for i in range(50):
            assert f"Rule number {i}" in block


class TestContextInjectionWithStandards:
    def test_standards_prepended_above_context(self):
        categories = {"preference": [_mem("Prefers tabs")]}
        out = format_context_for_injection(
            categories, standards=[_mem("All decks use the Opti template")]
        )
        # Standards come first, then the ordinary context, separated by a rule.
        assert out.index("AUTHORITATIVE Standards") < out.index("Neuralscape Memory Context")
        assert "All decks use the Opti template" in out

    def test_standards_survive_when_context_budget_exhausted(self):
        # A huge category set that blows the char budget must NOT evict standards.
        big = {"preference": [_mem("x" * 500) for _ in range(50)]}
        out = format_context_for_injection(
            big, max_chars=1000, standards=[_mem("Binding org rule")]
        )
        assert "Binding org rule" in out

    def test_standards_only_still_injected(self):
        out = format_context_for_injection({}, standards=[_mem("Binding org rule")])
        assert "Binding org rule" in out

    def test_no_standards_is_unchanged(self):
        categories = {"preference": [_mem("Prefers tabs")]}
        out = format_context_for_injection(categories)
        assert out.startswith("# Neuralscape Memory Context")
