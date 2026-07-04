"""Configuration for the dreaming extension.

All keys live under the ``DREAMING_*`` prefix. The extension replaces the
wiki_synthesizer as Neuralscape's synthesis system (see
docs/DREAMING_MODE_SPEC.md); the post-write graph-patch window setting
(``attach_window_seconds``) moved here with it.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class DreamingSettings(BaseSettings):
    """Environment-driven settings for the dreaming extension."""

    enabled: bool = Field(
        default=False,
        description=(
            "Master switch. When False the cron no-ops and the admin "
            "endpoint refuses to run. Dark until the operator opts in."
        ),
    )

    cron_hours: int = Field(
        default=24,
        ge=1,
        le=24,
        description=(
            "Sweep cadence in hours (24 = nightly). The cron fires at :35 "
            "to stagger against dedup (:00), expiry (:15) and the "
            "strategy-playbook synth (:55) on the same graph worker."
        ),
    )

    cron_anchor_hour: int = Field(
        default=3,
        ge=0,
        le=23,
        description=(
            "Hour-of-day (worker clock, normally UTC) the sweep cadence is "
            "anchored to. Containers run on UTC, so operators away from UTC "
            "should set this so the nightly sweep lands in their quiet "
            "hours — e.g. 21 puts the default nightly run at 03:35 UTC+6."
        ),
    )

    # ── Gate economy (cheap → expensive; see gate.py) ──
    min_hours: float = Field(
        default=24.0,
        ge=0.0,
        description="Cheap gate: minimum hours since a pool's last dream.",
    )
    min_new_memories: int = Field(
        default=20,
        ge=1,
        description=(
            "Expensive gate: minimum new/changed memories in the pool since "
            "its last dream. Counted during the LIGHT scroll."
        ),
    )

    # ── Adoption posture (hybrid; see consolidate/apply) ──
    auto_apply_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Destructive actions (INVALIDATE / PRUNE) at or above this "
            "confidence are applied; below it they land in the DreamRun "
            "report unapplied (shadow trial). Reversible actions (MERGE / "
            "REWRITE / TEMPORAL-REFRAME / LINK-ENRICH) always apply."
        ),
    )

    # ── Retention strength (Ebbinghaus; see scoring.py) ──
    prune_strength_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Retention-strength floor. Memories whose decayed, recall-"
            "reinforced strength falls below this become PRUNE candidates "
            "offered to the consolidation pass (never auto-pruned by score "
            "alone — the LLM must concur and the confidence gate applies)."
        ),
    )
    strength_half_life_days: float = Field(
        default=45.0,
        gt=0.0,
        description=(
            "Half-life of the Ebbinghaus retention decay, in days since "
            "last recall (or creation when never recalled)."
        ),
    )

    # ── Salience dynamics (A4; see dynamics.py) ──
    dynamics_enabled: bool = Field(
        default=True,
        description=(
            "Use recall-driven salience dynamics (strength / stability / "
            "decay) as the retention-strength input to promotion/prune "
            "scoring. When False — and for any memory without a recorded "
            "dynamics state — scoring falls back to the legacy single-"
            "half-life Ebbinghaus path. Never gates retrieval either way."
        ),
    )
    dynamics_strength_delta: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Strength increment per recall; a co-recalled memory (returned "
            "together with a sibling by the same query) earns the same "
            "again. Recalls by an already-seen query are damped so "
            "single-query hammering cannot compound."
        ),
    )
    dynamics_strength_cap: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Hard cap on accumulated strength — reinforcement saturates "
            "(guardrail: no rich-get-richer runaway)."
        ),
    )
    salience_recall_k: float = Field(
        default=0.0,
        ge=0.0,
        le=0.1,
        description=(
            "Bounded recall tie-breaker: score * (1 + k*log1p(strength "
            "signal)). 0.0 (default) disables it entirely — the search "
            "path then stays byte-identical and Redis-free. Hard-bounded "
            "at 0.1 (≤ ~16% boost at the strength cap) and 0.05 is the "
            "recommended ceiling: relevance must always dominate; "
            "salience shapes consolidation and vault presentation, never "
            "gates retrieval."
        ),
    )

    # ── Settling guard (A3-lite; see gate.check_settling_gate) ──
    settling_minutes: float = Field(
        default=30.0,
        ge=0.0,
        description=(
            "Defer a pool whose latest write (created_at/updated_at) is "
            "younger than this — never consolidate mid-conversation. The "
            "pool reports status 'settling' and is retried next pass; "
            "force=true bypasses; 0 disables the guard."
        ),
    )

    # ── Promotion gates (deep phase) ──
    min_recall_count: int = Field(
        default=2,
        ge=0,
        description="Deep promotion gate: minimum recalls for reinforcement-based promotion.",
    )
    min_unique_queries: int = Field(
        default=2,
        ge=0,
        description="Deep promotion gate: minimum distinct query hashes.",
    )

    # ── REM ──
    reflection_enabled: bool = Field(
        default=True,
        description="Toggle the REM reflection phase (insight memories + diary).",
    )
    max_reflections_per_pool: int = Field(
        default=5,
        ge=1,
        le=25,
        description="Ceiling on new insight memories a single sweep may write per pool.",
    )

    # ── Batch / scroll limits ──
    max_memories_per_pool: int = Field(
        default=200,
        ge=10,
        le=2000,
        description=(
            "Hard ceiling on memories staged per pool per sweep. Pools "
            "larger than this consolidate incrementally across sweeps "
            "(newest first)."
        ),
    )

    # ── Trace window (recall reinforcement) ──
    trace_ttl_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Rolling window for recall traces in Redis.",
    )

    # ── LLM ──
    model: str = Field(
        default="",
        description=(
            "Override the model used for consolidation/reflection. Empty "
            "inherits the Neuralscape default. The dreamer is not latency-"
            "constrained — prefer a stronger model than the extraction path."
        ),
    )
    llm_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=1800,
        description="Hard timeout on any single LLM call inside a sweep.",
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Retries per LLM call (1s/2s/4s backoff).",
    )

    # ── Post-write graph patch (moved from the wiki_synthesizer) ──
    attach_window_seconds: int = Field(
        default=120,
        ge=10,
        le=1800,
        description=(
            "Time window for the ``attach_memory_id`` post-write Cypher "
            "patch: Graphiti nodes with created_at >= write_started_at - "
            "this are stamped. Bump on instances with >2-minute graph "
            "writes; shrink to reduce false positives under concurrency."
        ),
    )

    # ── Vault output ──
    obsidian_vault_path: str = Field(
        default="/data/vault",
        description="Root of the Obsidian vault (same value as the conversation_compiler).",
    )
    vault_pages_enabled: bool = Field(
        default=True,
        description=(
            "Write humane topic pages (Projects/<pid>/<Topic>.md + hubs + "
            "Home.md) after each sweep — the wiki_synthesizer's successor "
            "output. The dream diary under Dreams/ is written regardless."
        ),
    )

    bridges_enabled: bool = Field(
        default=True,
        description=(
            "Cross-hub Bridges pass (B3a): after the per-pool librarian "
            "passes, topic pages whose subject spans multiple hubs gain a "
            "reciprocal '## Bridges' section. Requires vault_pages_enabled."
        ),
    )
    bridge_graph_limit: int = Field(
        default=200,
        ge=10,
        le=2000,
        description=(
            "Row ceiling on the single shared-entity Cypher the bridges "
            "pass runs per sweep (graph enrichment on top of the "
            "deterministic slug/source-id signals)."
        ),
    )

    identity_card_enabled: bool = Field(
        default=True,
        description=(
            "Identity-card pass (B4): the sweep maintains a pinned, "
            "grammar-constrained card per user pool and per project pool "
            "(Redis + Me/Card.md / Projects/<pid>/Card.md), exposed via "
            "the get_card MCP tool and GET /v1/extensions/dreaming/card."
        ),
    )

    dry_run_default: bool = Field(
        default=False,
        description="When True, sweeps report planned actions without writing.",
    )

    model_config = {
        "env_prefix": "DREAMING_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "protected_namespaces": (),  # allow the `model` field name
    }

    @property
    def vault_path(self) -> Path:
        return Path(self.obsidian_vault_path).expanduser().resolve()

    @property
    def dreams_dir(self) -> Path:
        """Root of the dream-diary tree (``{vault}/Dreams``)."""
        return self.vault_path / "Dreams"


dreaming_settings = DreamingSettings()
