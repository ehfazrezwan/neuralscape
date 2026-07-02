"""Graphiti custom entity + edge types for the trading-strategy adapter.

These Pydantic models are handed to ``graphiti.add_episode(entity_types=...,
edge_types=..., edge_type_map=...)`` so Graphiti classifies a trading book's
prose into a *structured, executable* strategy graph instead of generic
entities.

Two hard rules Graphiti imposes (do not break):

1. **The class docstring is load-bearing.** Graphiti feeds it to the LLM as the
   type's definition — write it as a crisp "this node represents …".
2. **Avoid reserved field names** (``uuid``, ``name``, ``group_id``, ``labels``,
   ``created_at``, ``summary``, ``attributes``, ``name_embedding``); make every
   attribute ``Optional`` with a rich ``Field(description=...)``.

**#1111 hedge (residual risk from the plan):** Graphiti's custom *edge*-attribute
population has a known gap. So all load-bearing, compiler-facing data (rule ASTs,
executable expressions, offsets, anchors) lives on **entity** nodes
(``EntryCondition``/``StopLoss``/``TakeProfit``/``RuleNode``), never on edges.
Edges here are thin relationship markers — semantics live in the node types they
connect and the ``edge_type_map``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Entity types ────────────────────────────────────────────────────


class Strategy(BaseModel):
    """A named, tradable strategy container — the top-level unit a trader picks and runs.

    Groups member setups under one thesis/bias (e.g. "Naked Forex — Reversal").
    """

    thesis: str | None = Field(default=None, description="The core idea/edge the strategy exploits")
    bias: str | None = Field(default=None, description="Directional bias: 'long', 'short', 'both', or 'neutral'")
    market_style: str | None = Field(default=None, description="e.g. 'reversal', 'continuation', 'breakout', 'mean-reversion'")
    status: str | None = Field(default=None, description="Lifecycle: draft|backtested|validated|live")
    source_book: str | None = Field(default=None, description="Book/author this strategy was learned from")


class Setup(BaseModel):
    """A tradable pattern or catalyst that triggers a trade (e.g. kangaroo tail, big shadow, last kiss).

    A setup is only actionable when its gate conditions are met (see the REQUIRES
    edge to a SupportResistanceZone and/or MarketRegime — the 3-part gate).
    """

    direction: str | None = Field(default=None, description="'bullish' / 'bearish' / 'either'")
    setup_kind: str | None = Field(default=None, description="'reversal' or 'continuation'")
    identification: str | None = Field(default=None, description="How to recognize the pattern in prose")
    aliases: str | None = Field(default=None, description="Other names for the same pattern (e.g. 'pin bar' for kangaroo tail)")
    source_quote: str | None = Field(default=None, description="Verbatim book text defining the setup")
    page_ref: str | None = Field(default=None, description="Page/chapter citation, e.g. 'Ch8 p.142'")


class SupportResistanceZone(BaseModel):
    """A support/resistance zone — the core price primitive that gates every setup.

    Zones (not exact lines) are where price reacts; a catalyst is only a trade
    when it prints *on* a zone. Includes round numbers and trendlines.
    """

    zone_kind: str | None = Field(default=None, description="'support', 'resistance', 'round_number', 'trendline'")
    how_identified: str | None = Field(default=None, description="How the zone is drawn/located")
    membership_test: str | None = Field(default=None, description="Predicate deciding whether a price is 'on' the zone")


class MarketRegime(BaseModel):
    """A market context/regime gate: trend, range, exhaustion, session, volatility state.

    Continuation setups require the right regime (e.g. an established trend).
    """

    regime_kind: str | None = Field(default=None, description="e.g. 'trend', 'range', 'exhaustion', 'session'")
    detection: str | None = Field(default=None, description="How to detect this regime")


class EntryCondition(BaseModel):
    """The trigger that opens a trade: order type, reference price, and offset.

    Load-bearing for the compiler — carries the executable entry expression.
    """

    order_type: str | None = Field(default=None, description="e.g. 'stop', 'limit', 'market'")
    reference_price: str | None = Field(default=None, description="Anchor, e.g. 'pattern.high', 'zone.edge'")
    offset: str | None = Field(default=None, description="Offset from the reference, e.g. '+5 pips + spread'")
    time_validity: str | None = Field(default=None, description="Order validity, e.g. 'next 1 candle else cancel'")
    rule_ast: str | None = Field(default=None, description="Boolean/expression tree (JSON string) over OHLC/zone/equity")
    executable_expression: str | None = Field(default=None, description="e.g. 'buy_stop = pattern.high + offset_pips'")
    source_quote: str | None = Field(default=None, description="Verbatim book text")
    page_ref: str | None = Field(default=None, description="Page/chapter citation")


class StopLoss(BaseModel):
    """Stop-loss placement for a setup — the anchor + offset defining risk."""

    anchor: str | None = Field(default=None, description="What the stop is placed against, e.g. 'pattern.low', 'tail tip'")
    offset: str | None = Field(default=None, description="Offset from the anchor, e.g. '-few pips'")
    stop_kind: str | None = Field(default=None, description="e.g. 'emergency', 'structural'")
    executable_expression: str | None = Field(default=None, description="e.g. 'stop = pattern.low - offset_pips'")
    source_quote: str | None = Field(default=None, description="Verbatim book text")
    page_ref: str | None = Field(default=None, description="Page/chapter citation")


class TakeProfit(BaseModel):
    """Fixed profit-target placement — an RR multiple or the next opposing zone."""

    anchor: str | None = Field(default=None, description="e.g. 'next opposing zone', 'RR 1:2'")
    rr_multiple: str | None = Field(default=None, description="Reward:risk multiple if fixed, e.g. '2' or '3'")
    executable_expression: str | None = Field(default=None, description="Target price expression")
    source_quote: str | None = Field(default=None, description="Verbatim book text")
    page_ref: str | None = Field(default=None, description="Page/chapter citation")


class ExitCondition(BaseModel):
    """Dynamic trade management / exit: zone, split, ladder, three-bar, trailing, drawdown-cut."""

    exit_kind: str | None = Field(default=None, description="'zone', 'split', 'ladder', 'three_bar', 'trailing', 'drawdown_cut'")
    trigger: str | None = Field(default=None, description="What triggers the exit action")
    executable_expression: str | None = Field(default=None, description="Exit logic expression")
    source_quote: str | None = Field(default=None, description="Verbatim book text")
    page_ref: str | None = Field(default=None, description="Page/chapter citation")


class RiskRule(BaseModel):
    """Position sizing / risk management rule: risk-per-trade, RR floor, survival limits."""

    risk_per_trade: str | None = Field(default=None, description="e.g. '1-2% of equity'")
    sizing_expression: str | None = Field(default=None, description="e.g. 'size = equity*risk_pct / (stop_pips*pip_value)'")
    rr_floor: str | None = Field(default=None, description="Minimum reward:risk, e.g. '1:2'")
    source_quote: str | None = Field(default=None, description="Verbatim book text")
    page_ref: str | None = Field(default=None, description="Page/chapter citation")


class PsychologyRule(BaseModel):
    """A discipline/temperament guardrail (e.g. gunner vs runner temperament)."""

    guidance: str | None = Field(default=None, description="The discipline rule")
    source_quote: str | None = Field(default=None, description="Verbatim book text")
    page_ref: str | None = Field(default=None, description="Page/chapter citation")


class Checklist(BaseModel):
    """An ordered routine that gates a trade — the steps a trader confirms before entering."""

    steps: str | None = Field(default=None, description="Ordered checklist steps")
    source_quote: str | None = Field(default=None, description="Verbatim book text")
    page_ref: str | None = Field(default=None, description="Page/chapter citation")


class RuleNode(BaseModel):
    """A node in a rule's boolean/expression AST — the compilable representation of a condition.

    Either an operator node (op over children) or a leaf (fact/operator/value).
    """

    op: str | None = Field(default=None, description="Operator: AND|OR|GT|LT|crossesAbove|closeInThird|engulfs|insideRange|…")
    fact: str | None = Field(default=None, description="Leaf fact reference, e.g. 'candle.range', 'price'")
    operator: str | None = Field(default=None, description="Leaf comparator, e.g. '>', '<', '=='")
    value: str | None = Field(default=None, description="Leaf comparison value")
    expression: str | None = Field(default=None, description="Full expression this node represents")


class Instrument(BaseModel):
    """A tradable instrument or asset class the strategy applies to (e.g. EUR/USD, spot FX, indices)."""

    symbol: str | None = Field(default=None, description="Ticker/pair if specific, e.g. 'EUR/USD'")
    asset_class: str | None = Field(default=None, description="e.g. 'spot_fx', 'equities', 'crypto', 'indices'")


class Timeframe(BaseModel):
    """A chart timeframe the setup operates on (e.g. H1, H4, D1)."""

    value: str | None = Field(default=None, description="Timeframe, e.g. 'H1', 'H4', 'D1'")
    preference: str | None = Field(default=None, description="e.g. 'minimum H1, best on H4/D1'")


class Indicator(BaseModel):
    """A technical indicator a setup uses (rare in pure price-action books; needed for others)."""

    indicator_kind: str | None = Field(default=None, description="e.g. 'EMA', 'RSI', 'MACD'")
    parameters: str | None = Field(default=None, description="Indicator parameters, e.g. 'period=20'")


class Signal(BaseModel):
    """A normalized trade signal (shape mirrors QuantConnect LEAN's Insight).

    Emitted by a setup/strategy at inference; combined by ensembles.
    """

    symbol: str | None = Field(default=None, description="Instrument the signal is for")
    direction: str | None = Field(default=None, description="'up' / 'down' / 'flat'")
    period: str | None = Field(default=None, description="Expected duration of the prediction")
    signal_type: str | None = Field(default=None, description="e.g. 'price', 'volatility'")
    magnitude: str | None = Field(default=None, description="Predicted move size")
    confidence: str | None = Field(default=None, description="0..1 confidence")
    weight: str | None = Field(default=None, description="Portfolio weight suggestion")
    source_model: str | None = Field(default=None, description="Which model/strategy produced it")


class Backtest(BaseModel):
    """Validation results for a strategy (walk-forward / CPCV / DSR / PBO), written back after evaluation."""

    method: str | None = Field(default=None, description="e.g. 'CPCV', 'walk-forward'")
    deflated_sharpe: str | None = Field(default=None, description="Deflated Sharpe Ratio")
    pbo: str | None = Field(default=None, description="Probability of Backtest Overfitting")
    trial_count: str | None = Field(default=None, description="Number of trials (needed for DSR)")
    result_summary: str | None = Field(default=None, description="Human-readable outcome")


class Ensemble(BaseModel):
    """A combination of strategies run together, with a weighting/gating policy (composition target)."""

    weighting_policy: str | None = Field(default=None, description="How member signals are combined")
    regime_gate: str | None = Field(default=None, description="Regime that activates this ensemble")


class VisualExemplar(BaseModel):
    """A canonical chart image of a setup extracted from a trading book (see VISUAL_EXEMPLARS_SPEC).

    Stores the multimodal model's *structured read* of an annotated setup picture
    so a live chart-vision agent can few-shot against it. The image bytes live in
    the object store; ``image_uri`` points to them.
    """

    setup_name: str | None = Field(default=None, description="Which setup this image exemplifies, e.g. 'Kangaroo Tail'")
    direction: str | None = Field(default=None, description="'bullish' / 'bearish'")
    caption: str | None = Field(default=None, description="Nearby caption text from the book")
    visual_description: str | None = Field(default=None, description="Model's structured read: tail/body position, zone location, 'room to the left', relative size — the checkable visual features")
    key_levels: str | None = Field(default=None, description="Annotated prices if legible")
    chart_context: str | None = Field(default=None, description="Timeframe/instrument if shown")
    image_uri: str | None = Field(default=None, description="URI of the stored image bytes")
    page_ref: str | None = Field(default=None, description="Page/chapter citation")


# ── Edge types (thin markers — semantics live in the nodes; see #1111 hedge) ──


class HAS_SETUP(BaseModel):
    """Strategy contains this Setup as a member."""


class TRADES(BaseModel):
    """Strategy trades this Instrument."""


class ON_TIMEFRAME(BaseModel):
    """Setup operates on this Timeframe."""


class USES_INDICATOR(BaseModel):
    """Setup uses this Indicator."""


class HAS_ENTRY(BaseModel):
    """Setup has this EntryCondition."""


class HAS_EXIT(BaseModel):
    """Setup has this ExitCondition."""


class HAS_STOP(BaseModel):
    """Setup has this StopLoss."""


class HAS_TARGET(BaseModel):
    """Setup has this TakeProfit."""


class CONSTRAINED_BY(BaseModel):
    """Strategy/Setup is constrained by this RiskRule."""


class ACTIVE_IN_REGIME(BaseModel):
    """Setup/Strategy is active only in this MarketRegime."""


class REQUIRES(BaseModel):
    """The gate: this Setup only fires when the required SupportResistanceZone / MarketRegime holds.

    Naked Forex's central law — a catalyst off a zone is not a trade.
    """


class HAS_CHILD(BaseModel):
    """Parent RuleNode has this child RuleNode (AST edge / bracket triad)."""


class DERIVED_FROM(BaseModel):
    """This node was derived from a source page/chunk/episode (provenance)."""


class COMPOSED_OF(BaseModel):
    """Ensemble is composed of this Strategy."""


class VALIDATED_BY(BaseModel):
    """Strategy is validated by this Backtest."""


class EXEMPLIFIES(BaseModel):
    """VisualExemplar exemplifies this Setup (a canonical picture of the pattern)."""


# ── Registries handed to add_episode ────────────────────────────────

ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Strategy": Strategy,
    "Setup": Setup,
    "SupportResistanceZone": SupportResistanceZone,
    "MarketRegime": MarketRegime,
    "EntryCondition": EntryCondition,
    "StopLoss": StopLoss,
    "TakeProfit": TakeProfit,
    "ExitCondition": ExitCondition,
    "RiskRule": RiskRule,
    "PsychologyRule": PsychologyRule,
    "Checklist": Checklist,
    "RuleNode": RuleNode,
    "Instrument": Instrument,
    "Timeframe": Timeframe,
    "Indicator": Indicator,
    "Signal": Signal,
    "Backtest": Backtest,
    "Ensemble": Ensemble,
    "VisualExemplar": VisualExemplar,
}

EDGE_TYPES: dict[str, type[BaseModel]] = {
    "HAS_SETUP": HAS_SETUP,
    "TRADES": TRADES,
    "ON_TIMEFRAME": ON_TIMEFRAME,
    "USES_INDICATOR": USES_INDICATOR,
    "HAS_ENTRY": HAS_ENTRY,
    "HAS_EXIT": HAS_EXIT,
    "HAS_STOP": HAS_STOP,
    "HAS_TARGET": HAS_TARGET,
    "CONSTRAINED_BY": CONSTRAINED_BY,
    "ACTIVE_IN_REGIME": ACTIVE_IN_REGIME,
    "REQUIRES": REQUIRES,
    "HAS_CHILD": HAS_CHILD,
    "DERIVED_FROM": DERIVED_FROM,
    "COMPOSED_OF": COMPOSED_OF,
    "VALIDATED_BY": VALIDATED_BY,
    "EXEMPLIFIES": EXEMPLIFIES,
}

# (source_entity_type, target_entity_type) -> allowed edge type names.
# Constrains Graphiti so, e.g., a Setup→Zone edge can only be REQUIRES (the gate).
EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("Strategy", "Setup"): ["HAS_SETUP"],
    ("Strategy", "Instrument"): ["TRADES"],
    ("Strategy", "RiskRule"): ["CONSTRAINED_BY"],
    ("Strategy", "MarketRegime"): ["ACTIVE_IN_REGIME"],
    ("Strategy", "Backtest"): ["VALIDATED_BY"],
    ("Setup", "SupportResistanceZone"): ["REQUIRES"],
    ("Setup", "MarketRegime"): ["REQUIRES", "ACTIVE_IN_REGIME"],
    ("Setup", "EntryCondition"): ["HAS_ENTRY"],
    ("Setup", "ExitCondition"): ["HAS_EXIT"],
    ("Setup", "StopLoss"): ["HAS_STOP"],
    ("Setup", "TakeProfit"): ["HAS_TARGET"],
    ("Setup", "Timeframe"): ["ON_TIMEFRAME"],
    ("Setup", "Indicator"): ["USES_INDICATOR"],
    ("Setup", "RiskRule"): ["CONSTRAINED_BY"],
    ("RuleNode", "RuleNode"): ["HAS_CHILD"],
    ("Ensemble", "Strategy"): ["COMPOSED_OF"],
    ("VisualExemplar", "Setup"): ["EXEMPLIFIES"],
}

# Guidance injected into Graphiti's extraction prompts so its own entity
# extraction is trading-aware (complements the fact extractor).
CUSTOM_EXTRACTION_INSTRUCTIONS = (
    "This text is from a trading-strategy book. Extract the strategy, its member "
    "setups (patterns/catalysts like kangaroo tail, big shadow, last kiss), and for "
    "each setup its entry/stop/target/exit conditions, the support/resistance zones "
    "and market regimes it REQUIRES (the gate — a catalyst is only a trade when it "
    "prints on a zone), risk rules, and timeframes. Preserve exact price offsets, "
    "anchors, and any executable expressions on the entity attributes. When a page "
    "or chapter is identifiable, record it as page_ref, and keep the verbatim book "
    "wording as source_quote."
)
