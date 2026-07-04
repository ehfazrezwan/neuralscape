"""The ``code_graph`` knowledge adapter — profile + registration.

Registration happens via :func:`register` (called from ``adapters/__init__``
through :func:`adapters.code_graph.register`, which gates on the optional
graphifyy library being installed). It:

1. registers the 5 code-native categories into the shared taxonomy so
   ``store_raw``, the ingest validators, and the fact parser accept them
   (additive — the core 13 are untouched);
2. registers the GRAPH_REPORT.md section chunker + the report fact extractor;
3. registers the ``code_graph`` adapter with its minimal summary-layer ontology.

Code-graph categories are deliberately kept OUT of ``CATEGORY_VAULT_PATHS``
(same call as the trading adapter): the coding-assistant wiki synthesizer
iterates that dict, and code-graph knowledge is anchored to a specific
graph.json via source_refs — the librarian links out to the live graph rather
than re-rendering it.
"""

from __future__ import annotations

from adapters.base import KnowledgeAdapter, register_adapter
from ingest.chunking_strategies import register_chunking_strategy
from ingest.extractors import register_extractor
from schemas import register_categories

from adapters.code_graph import ADAPTER_NAME
from adapters.code_graph.chunking import (
    REPORT_MAX_CHARS,
    REPORT_OVERLAP,
    GraphReportSectionStrategy,
)
from adapters.code_graph.extractor import CodeGraphReportExtractor
from adapters.code_graph.ontology import (
    CUSTOM_EXTRACTION_INSTRUCTIONS,
    EDGE_TYPE_MAP,
    EDGE_TYPES,
    ENTITY_TYPES,
)

# ── (1) Taxonomy ──
CODE_GRAPH_CATEGORIES: dict[str, str] = {
    "module": "A module/architectural community of a codebase and its purpose",
    "boundary": "A coupling or boundary observation between modules (esp. surprising cross-module connections)",
    "invariant": "A rule the code maintains across commits (e.g. 'writes are async, reads are sync')",
    "rationale": "A documented WHY from the code: design rationale, NOTE/HACK comments, trade-offs",
    "hotspot": "A god node / core abstraction with wide blast radius, or a churn/complexity risk",
}


def register() -> KnowledgeAdapter:
    """Register categories, chunker, extractor, and the adapter. Idempotent."""
    # Code-graph categories are knowledge about a specific codebase — keep them
    # flexible (scope follows the caller's project_id, like domain_knowledge).
    # Not added to CATEGORY_VAULT_PATHS (see module docstring).
    register_categories(CODE_GRAPH_CATEGORIES)

    register_chunking_strategy(GraphReportSectionStrategy.name, GraphReportSectionStrategy())
    register_extractor(CodeGraphReportExtractor.name, CodeGraphReportExtractor())

    adapter = KnowledgeAdapter(
        name=ADAPTER_NAME,
        version="1.0",
        categories=dict(CODE_GRAPH_CATEGORIES),
        chunking_strategy=GraphReportSectionStrategy.name,
        default_max_chars=REPORT_MAX_CHARS,
        default_overlap=REPORT_OVERLAP,
        extractor=CodeGraphReportExtractor.name,
        entity_types=ENTITY_TYPES,
        edge_types=EDGE_TYPES,
        edge_type_map=EDGE_TYPE_MAP,
        custom_extraction_instructions=CUSTOM_EXTRACTION_INSTRUCTIONS,
    )
    register_adapter(adapter)
    return adapter
