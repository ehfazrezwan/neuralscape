"""The ``trading_strategy`` knowledge adapter — profile + registration.

Importing this module (side effects):
1. registers the 12 trading categories into the shared taxonomy so ``store_raw``,
   the ingest validators, and the fact parser accept them (additive — the core
   13 are untouched);
2. registers the section-aware chunker + the trading fact extractor;
3. registers the ``trading_strategy`` adapter into ``ADAPTER_REGISTRY``.

Trading categories are deliberately kept OUT of ``CATEGORY_VAULT_PATHS`` so the
coding-assistant ``wiki_synthesizer`` (which iterates that dict) never tries to
render trading memories — the trading playbooks are the ``strategy_synthesizer``
extension's job (Phase 3), grouped by strategy name, not category.
"""

from __future__ import annotations

from adapters.base import KnowledgeAdapter, register_adapter
from ingest.chunking_strategies import register_chunking_strategy
from ingest.extractors import register_extractor
from schemas import register_categories

from adapters.trading.chunking import SectionAwareStrategy
from adapters.trading.extractor import TradingStrategyExtractor
from adapters.trading.ontology import (
    CUSTOM_EXTRACTION_INSTRUCTIONS,
    EDGE_TYPE_MAP,
    EDGE_TYPES,
    ENTITY_TYPES,
)

ADAPTER_NAME = "trading_strategy"

# ── (1) Taxonomy ──
TRADING_CATEGORIES: dict[str, str] = {
    "strategy": "A named strategy container (thesis, bias, member setups)",
    "setup": "A tradable pattern/catalyst (kangaroo tail, big shadow, last kiss, …)",
    "entry_rule": "The trigger: order type, reference price, offset, time validity",
    "exit_rule": "Dynamic trade management (zone/split/ladder/three-bar/trailing/drawdown-cut)",
    "stop_rule": "Stop-loss placement (anchor + offset)",
    "take_profit_rule": "Fixed target placement (RR multiple / next opposing zone)",
    "market_condition": "Regime/context gate (trend, range, exhaustion, session)",
    "sr_concept": "Support/resistance zones (the core primitive) + round numbers/trendlines",
    "risk_rule": "Position sizing, risk-per-trade, survival limits",
    "psychology_rule": "Discipline/temperament guardrails (gunner vs runner)",
    "checklist": "The ordered routine that gates a trade",
    "glossary": "Definitions + aliases (kangaroo tail ≈ pin bar) for cross-book normalization",
}


def _register() -> KnowledgeAdapter:
    # Trading categories are reference knowledge — keep them flexible (scope
    # follows the caller's project_id, like domain_knowledge) rather than forcing
    # global/project. Not added to CATEGORY_VAULT_PATHS (see module docstring).
    register_categories(TRADING_CATEGORIES)

    register_chunking_strategy(SectionAwareStrategy.name, SectionAwareStrategy())
    register_extractor(TradingStrategyExtractor.name, TradingStrategyExtractor())

    adapter = KnowledgeAdapter(
        name=ADAPTER_NAME,
        version="1.0",
        categories=dict(TRADING_CATEGORIES),
        chunking_strategy=SectionAwareStrategy.name,
        default_max_chars=4000,
        default_overlap=400,
        extractor=TradingStrategyExtractor.name,
        entity_types=ENTITY_TYPES,
        edge_types=EDGE_TYPES,
        edge_type_map=EDGE_TYPE_MAP,
        custom_extraction_instructions=CUSTOM_EXTRACTION_INSTRUCTIONS,
        synthesizer="strategy_synthesizer",
        synthesis_group_key="strategy_name",
    )
    register_adapter(adapter)
    return adapter


TRADING_ADAPTER = _register()
