"""Pluggable fact extractors.

The facts path of the ingest pipeline distils a document into ``[category] fact``
lines via an LLM. Which *prompt* runs — and how the response is parsed — is the
swappable part: a coding-assistant adapter extracts preferences/decisions; a
trading adapter extracts executable rules (rule_ast + executable_expression +
source_quote/page_ref).

A :class:`FactExtractor` owns two steps:

- ``build_messages(text) -> list[dict]`` — the LLM messages (the domain prompt
  wrapped around the document text);
- ``parse(response_text) -> list[tuple[str, str]]`` — the ``(category, content)``
  tuples the pipeline stores.

:meth:`MemoryService.extract_facts_only` calls the extractor, runs the shared
Gemini client + junk filter, and stores the results — so the LLM plumbing,
retries, and provenance envelope stay identical across adapters.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from prompts import build_extraction_messages, parse_extraction_response


@runtime_checkable
class FactExtractor(Protocol):
    """Build the extraction prompt and parse its response into facts."""

    def build_messages(self, text: str) -> list[dict]:
        ...

    def parse(self, response_text: str) -> list[tuple[str, str]]:
        ...


class DefaultExtractor:
    """Today's coding-assistant extractor — the default for every adapter.

    Wraps :func:`prompts.build_extraction_messages` /
    :func:`prompts.parse_extraction_response` unchanged so the default facts
    path is byte-for-byte identical to the pre-adapter pipeline.
    """

    name = "default"

    def build_messages(self, text: str) -> list[dict]:
        return build_extraction_messages([{"role": "user", "content": text}])

    def parse(self, response_text: str) -> list[tuple[str, str]]:
        return parse_extraction_response(response_text)


FACT_EXTRACTORS: dict[str, FactExtractor] = {
    DefaultExtractor.name: DefaultExtractor(),
}


def register_extractor(name: str, extractor: FactExtractor) -> None:
    """Register (or replace) a fact extractor under ``name``."""
    FACT_EXTRACTORS[name] = extractor


def get_extractor(name: str | None) -> FactExtractor:
    """Resolve a fact extractor by id, falling back to the default.

    An unknown/None id degrades to the coding-assistant extractor.
    """
    if not name:
        return FACT_EXTRACTORS[DefaultExtractor.name]
    return FACT_EXTRACTORS.get(name, FACT_EXTRACTORS[DefaultExtractor.name])
