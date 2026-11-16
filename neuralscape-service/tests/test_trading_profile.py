"""Direct tests for adapters/trading/profile.py: the adapter's declarative profile and its registration."""

from __future__ import annotations

import re

import pytest

import adapters
from adapters.base import ADAPTER_REGISTRY, KnowledgeAdapter, get_adapter, require_adapter
from adapters.trading import profile
from ingest.chunking_strategies import CHUNKING_STRATEGIES, get_chunking_strategy
from ingest.extractors import FACT_EXTRACTORS, get_extractor
from schemas import (
    CATEGORY_VAULT_PATHS,
    GLOBAL_CATEGORIES,
    MEMORY_CATEGORIES,
    PROJECT_CATEGORIES,
    default_scope_for_category,
)

EXPECTED_CATEGORIES = {
    "strategy", "setup", "entry_rule", "exit_rule", "stop_rule", "take_profit_rule",
    "market_condition", "sr_concept", "risk_rule", "psychology_rule", "checklist", "glossary",
}


def test_adapter_name_and_registry_entry():
    assert profile.ADAPTER_NAME == "trading_strategy"
    assert ADAPTER_REGISTRY["trading_strategy"] is profile.TRADING_ADAPTER
    assert get_adapter("trading_strategy") is profile.TRADING_ADAPTER
    assert require_adapter("trading_strategy") is profile.TRADING_ADAPTER
    assert isinstance(profile.TRADING_ADAPTER, KnowledgeAdapter)


def test_twelve_trading_categories_with_descriptions():
    assert set(profile.TRADING_CATEGORIES) == EXPECTED_CATEGORIES
    assert len(profile.TRADING_CATEGORIES) == 12
    for name, desc in profile.TRADING_CATEGORIES.items():
        assert re.fullmatch(r"[a-z_]+", name)
        assert desc.strip(), f"{name} has no description"
    assert profile.TRADING_ADAPTER.categories == profile.TRADING_CATEGORIES
    assert profile.TRADING_ADAPTER.categories is not profile.TRADING_CATEGORIES  # a copy, not an alias


def test_categories_are_added_to_the_shared_taxonomy_without_touching_the_core():
    for name in EXPECTED_CATEGORIES:
        assert name in MEMORY_CATEGORIES
    for core in ("preference", "decision", "tech_stack", "domain_knowledge"):
        assert core in MEMORY_CATEGORIES


def test_trading_categories_are_flexible_scope_not_global_or_project():
    """The profile declares no scope sets, so scope follows the caller's project_id (like domain_knowledge)."""
    assert profile.TRADING_ADAPTER.global_categories == frozenset()
    assert profile.TRADING_ADAPTER.project_categories == frozenset()
    assert not (EXPECTED_CATEGORIES & GLOBAL_CATEGORIES)
    assert not (EXPECTED_CATEGORIES & PROJECT_CATEGORIES)
    for name in EXPECTED_CATEGORIES:
        default_scope_for_category(name)  # accepted, no error


def test_trading_categories_stay_out_of_the_wiki_vault_paths():
    """The coding-assistant wiki synthesizer iterates CATEGORY_VAULT_PATHS; trading is the strategy synthesizer's."""
    assert not (EXPECTED_CATEGORIES & set(CATEGORY_VAULT_PATHS))


def test_chunker_and_extractor_are_registered_and_wired_into_the_profile():
    a = profile.TRADING_ADAPTER
    assert a.chunking_strategy == "section_aware" and a.extractor in FACT_EXTRACTORS
    assert CHUNKING_STRATEGIES["section_aware"] is get_chunking_strategy("section_aware")
    assert get_extractor(a.extractor) is FACT_EXTRACTORS[a.extractor]
    assert (a.default_max_chars, a.default_overlap) == (4000, 400)


def test_graph_ontology_and_synthesis_policy():
    a = profile.TRADING_ADAPTER
    assert a.entity_types and a.edge_types and a.edge_type_map
    assert "VisualExemplar" in a.entity_types
    assert a.custom_extraction_instructions and "trading" in a.custom_extraction_instructions.lower()
    assert (a.synthesizer, a.synthesis_group_key) == ("strategy_synthesizer", "strategy_name")
    assert a.version == "1.0"


def test_register_is_idempotent_and_returns_the_registered_adapter(monkeypatch):
    monkeypatch.setitem(ADAPTER_REGISTRY, "trading_strategy", profile.TRADING_ADAPTER)  # restored at teardown
    before = set(ADAPTER_REGISTRY)
    again = profile._register()
    assert ADAPTER_REGISTRY["trading_strategy"] is again
    assert again.categories == profile.TRADING_CATEGORIES
    assert set(ADAPTER_REGISTRY) == before          # nothing else added or removed


def test_the_adapters_package_registers_trading_on_import():
    assert "trading_strategy" in ADAPTER_REGISTRY
    assert getattr(adapters, "__name__", "adapters") == "adapters"


def test_unknown_adapter_name_is_not_trading():
    assert get_adapter("no-such-adapter") is not profile.TRADING_ADAPTER
    with pytest.raises(Exception):
        require_adapter("no-such-adapter")
