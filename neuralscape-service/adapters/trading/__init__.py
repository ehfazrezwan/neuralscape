"""Trading-strategy knowledge adapter.

Registers the ``trading_strategy`` adapter: the trading taxonomy, a
section-aware chunker for books, a rule-extracting fact extractor, and the
Graphiti trading ontology (Strategy/Setup/EntryCondition/…/VisualExemplar).
Importing this module registers the adapter + its categories as a side effect
(see :mod:`adapters.trading.profile`).
"""

from adapters.trading.profile import ADAPTER_NAME, TRADING_ADAPTER, TRADING_CATEGORIES

__all__ = ["ADAPTER_NAME", "TRADING_ADAPTER", "TRADING_CATEGORIES"]
