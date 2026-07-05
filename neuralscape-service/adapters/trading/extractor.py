"""The trading-strategy fact extractor.

Extracts **executable rules, not prose**. For each rule the LLM emits a trading
``[category]`` plus a machine-checkable ``rule_ast`` (boolean/expression tree), an
``executable_expression``, and the verbatim ``source_quote`` + ``page_ref``. The
parser flattens each rule into one memory whose body keeps all of that inline —
so the rule is (a) recallable by vector search, (b) parseable by Bellwether's
compiler, and (c) rich enough for Graphiti to extract the trading ontology
(Setup / EntryCondition / StopLoss / …) from it.

Implements the :class:`ingest.extractors.FactExtractor` protocol
(``build_messages`` + ``parse``); the shared Gemini client, retries, and junk
filter live in ``MemoryService.extract_facts_only``.
"""

from __future__ import annotations

import json
import logging
import re

from schemas import MEMORY_CATEGORIES

logger = logging.getLogger(__name__)


TRADING_EXTRACTION_PROMPT = """You are a trading-strategy knowledge engine. The text below is from a trading book or a trader's notes. Extract the strategy's **executable rules**, not prose summaries.

Extract distinct rules. For EACH rule, produce an object with:
- "category": ONE of these trading categories:
  - "strategy"          — a named strategy container (thesis, bias)
  - "setup"             — a tradable pattern/catalyst (kangaroo tail, big shadow, last kiss, …)
  - "entry_rule"        — the trigger: order type, reference price, offset, time validity
  - "exit_rule"         — dynamic trade management (zone/split/ladder/three-bar/trailing/drawdown-cut)
  - "stop_rule"         — stop-loss placement (anchor + offset)
  - "take_profit_rule"  — fixed target (RR multiple or next opposing zone)
  - "market_condition"  — regime/context gate (trend, range, exhaustion, session)
  - "sr_concept"        — support/resistance zones (the core primitive), round numbers, trendlines
  - "risk_rule"         — position sizing, risk-per-trade, survival limits
  - "psychology_rule"   — discipline/temperament guardrails
  - "checklist"         — the ordered routine that gates a trade
  - "glossary"          — a definition + aliases (e.g. "kangaroo tail" ≈ "pin bar")
- "strategy_name": the named strategy/setup this rule belongs to (for grouping), or null
- "statement": one clear standalone sentence describing the rule
- "rule_ast": a boolean/expression tree over OHLC/zone/equity, or null. Use nested objects like
  {"op":"AND","children":[...]} for operators and {"fact":"candle.range","operator":">","value":"max(range,10)"} for leaves.
  Operators you may use: AND, OR, NOT, GT, LT, GTE, LTE, EQ, crossesAbove, crossesBelow, closeInThird, openInThird, engulfs, insideRange, roomToLeft, onZone.
- "executable_expression": a concrete assignment when applicable, e.g. "buy_stop = pattern.high + offset_pips(5) + spread", or null
- "source_quote": the verbatim book sentence this rule came from (keep it short), or null
- "page_ref": page/chapter citation if identifiable (e.g. "Ch8 p.142"), or null

CRITICAL FIDELITY RULE: a catalyst is only a trade when it prints ON a support/resistance zone (and, for continuation setups, in the right regime). When a setup requires a zone/regime, say so explicitly in the statement and encode it in the rule_ast (onZone leaf).

Rules:
1. Prefer the book's exact words and numbers. Never invent pip values or offsets not in the text.
2. Each rule must stand alone (no "as above").
3. Skip pure narrative, marketing, and anecdotes.

Respond with a JSON object:
{
  "rules": [
    {"category":"...","strategy_name":"...","statement":"...","rule_ast":{...}|null,"executable_expression":"..."|null,"source_quote":"..."|null,"page_ref":"..."|null}
  ]
}

If no rules can be extracted, return {"rules": []}.

TEXT:
"""


def _clean_json(text: str) -> str:
    """Strip a Markdown code fence if the model wrapped its JSON."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t


def _render_rule(rule: dict) -> str:
    """Flatten one extracted rule into a single memory body, keeping all fields inline."""
    parts: list[str] = []
    strat = (rule.get("strategy_name") or "").strip()
    statement = (rule.get("statement") or "").strip()
    parts.append(f"[{strat}] {statement}" if strat else statement)

    ast = rule.get("rule_ast")
    if ast not in (None, "", {}, []):
        try:
            parts.append(f"Rule (AST): {json.dumps(ast, separators=(',', ':'))}")
        except (TypeError, ValueError):
            parts.append(f"Rule (AST): {ast}")

    expr = (rule.get("executable_expression") or "").strip()
    if expr:
        parts.append(f"Executable: {expr}")

    quote = (rule.get("source_quote") or "").strip()
    page = (rule.get("page_ref") or "").strip()
    if quote and page:
        parts.append(f'Source ({page}): "{quote}"')
    elif quote:
        parts.append(f'Source: "{quote}"')
    elif page:
        parts.append(f"Source: {page}")
    return "\n".join(p for p in parts if p)


class TradingStrategyExtractor:
    """FactExtractor that emits executable trading rules (rule_ast + expression + citation)."""

    name = "trading_strategy"

    def build_messages(self, text: str) -> list[dict]:
        return [{"role": "user", "content": TRADING_EXTRACTION_PROMPT + text}]

    def parse(self, response_text: str) -> list[tuple[str, str]]:
        try:
            data = json.loads(_clean_json(response_text))
            rules = data.get("rules", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Trading extraction: response was not valid JSON — 0 rules")
            return []

        out: list[tuple[str, str]] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            category = str(rule.get("category", "")).strip().lower()
            # A trading category must have been registered into MEMORY_CATEGORIES
            # by the adapter's profile import; fall back to domain_knowledge so an
            # off-vocab category never fails store_raw's membership check.
            if category not in MEMORY_CATEGORIES:
                logger.debug("Trading extraction: unknown category %r → domain_knowledge", category)
                category = "domain_knowledge"
            body = _render_rule(rule)
            if body.strip():
                out.append((category, body))
        return out
