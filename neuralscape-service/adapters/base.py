"""Pluggable knowledge adapters.

A :class:`KnowledgeAdapter` bundles everything the ingest pipeline can swap
*per document* while keeping the **fixed metadata envelope** identical:

- the **taxonomy** — the category set + (optionally) which categories are
  global/project-scoped;
- the **chunking strategy** — resolved by id from
  :mod:`ingest.chunking_strategies`;
- the **fact-extraction prompt** — resolved by id from :mod:`ingest.extractors`;
- the **graph ontology** — Graphiti custom ``entity_types`` / ``edge_types`` /
  ``edge_type_map`` / ``custom_extraction_instructions`` threaded down to
  ``add_episode``;
- the **synthesis policy** — which cumulative-synthesizer extension owns this
  content and what key it groups by.

The envelope (``source_ref``, ``memory_kind``, content-hash dedup, scope,
visibility, provenance) is *never* adapter-specific — see the guardrail test in
``tests/test_adapters.py``.

Adapters are **declarative profiles instantiated once at import** (not built
from user input), so this is a plain frozen dataclass rather than a Pydantic
model: it holds ``type[BaseModel]`` values and tuple-keyed dicts (Graphiti's
``edge_type_map``) that don't round-trip cleanly through Pydantic validation and
never need to. Only the adapter *name* (a string) is threaded across the ARQ
queue; the worker re-resolves the full profile from :data:`ADAPTER_REGISTRY` via
:func:`get_adapter`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from schemas import MEMORY_CATEGORIES

DEFAULT_ADAPTER_NAME = "default"


@dataclass(frozen=True)
class KnowledgeAdapter:
    """A declarative ingestion profile.

    Every field has a default that reproduces today's behavior, so the
    ``"default"`` adapter is a no-op and any partially-specified custom adapter
    inherits the rest.
    """

    name: str
    version: str = "1.0"

    # ── (1) Taxonomy ──
    # Category -> human description. Empty ⇒ the core 13-item MEMORY_CATEGORIES.
    # Custom adapters register their categories into the shared taxonomy at
    # import (see ``register_categories`` in schemas.py) so ``store_raw``'s
    # membership check and the fact parser both accept them; this field is the
    # adapter-local view used for prompts/docs.
    categories: dict[str, str] = field(default_factory=dict)

    # ── (2) Chunking ── resolved from ingest.chunking_strategies
    chunking_strategy: str = "paragraph_aware"
    default_max_chars: int = 1500
    default_overlap: int = 150

    # ── (3) Extraction ── resolved from ingest.extractors
    extractor: str = "default"

    # ── (4) Graph ontology (Graphiti custom types) ──
    # None ⇒ Graphiti's built-in generic entity/edge extraction (current behavior).
    entity_types: dict[str, type[BaseModel]] | None = None
    edge_types: dict[str, type[BaseModel]] | None = None
    edge_type_map: dict[tuple[str, str], list[str]] | None = None
    excluded_entity_types: list[str] | None = None
    custom_extraction_instructions: str | None = None

    # ── (5) Synthesis policy ──
    synthesizer: str | None = None
    synthesis_group_key: str | None = None

    def resolved_categories(self) -> dict[str, str]:
        """The category set this adapter classifies into.

        Falls back to the core taxonomy when the adapter didn't declare its own.
        """
        return self.categories or dict(MEMORY_CATEGORIES)

    def has_graph_ontology(self) -> bool:
        """True when this adapter supplies any Graphiti custom types."""
        return any(
            (
                self.entity_types,
                self.edge_types,
                self.edge_type_map,
                self.excluded_entity_types,
                self.custom_extraction_instructions,
            )
        )

    def graph_ontology_kwargs(self) -> dict | None:
        """The adapter's Graphiti custom types as ``add_episode`` kwargs.

        Returns ``None`` for the default (no-ontology) adapter so the graph
        write path is untouched; otherwise a dict of only the set fields, ready
        to forward through ``enrich_graph → MemoryGraph.add → add_episode``.
        """
        if not self.has_graph_ontology():
            return None
        kwargs: dict = {}
        if self.entity_types:
            kwargs["entity_types"] = self.entity_types
        if self.edge_types:
            kwargs["edge_types"] = self.edge_types
        if self.edge_type_map:
            kwargs["edge_type_map"] = self.edge_type_map
        if self.excluded_entity_types:
            kwargs["excluded_entity_types"] = self.excluded_entity_types
        if self.custom_extraction_instructions:
            kwargs["custom_extraction_instructions"] = self.custom_extraction_instructions
        return kwargs


# The default adapter re-exposes today's behavior exactly: the core 13
# categories, the paragraph-aware chunker, the coding-assistant extraction
# prompt, and no Graphiti custom types. Selecting it must be byte-for-byte
# equivalent to the pre-adapter ingest path.
DEFAULT_ADAPTER = KnowledgeAdapter(
    name=DEFAULT_ADAPTER_NAME,
    categories=dict(MEMORY_CATEGORIES),
    chunking_strategy="paragraph_aware",
    extractor="default",
)


ADAPTER_REGISTRY: dict[str, KnowledgeAdapter] = {
    DEFAULT_ADAPTER_NAME: DEFAULT_ADAPTER,
}


def register_adapter(adapter: KnowledgeAdapter) -> None:
    """Register (or replace) an adapter under its ``name``.

    Idempotent — re-registering the same name overwrites, so a module that
    defines an adapter can be imported more than once without error.
    """
    ADAPTER_REGISTRY[adapter.name] = adapter


def get_adapter(name: str | None = DEFAULT_ADAPTER_NAME) -> KnowledgeAdapter:
    """Resolve an adapter by name, falling back to the default.

    A ``None`` or unknown name resolves to :data:`DEFAULT_ADAPTER` so a stale
    or typo'd ``adapter=`` value degrades to current behavior instead of failing
    an ingest.
    """
    if not name:
        return DEFAULT_ADAPTER
    return ADAPTER_REGISTRY.get(name, DEFAULT_ADAPTER)


def list_adapters() -> list[str]:
    """Return the registered adapter names, sorted."""
    return sorted(ADAPTER_REGISTRY.keys())
