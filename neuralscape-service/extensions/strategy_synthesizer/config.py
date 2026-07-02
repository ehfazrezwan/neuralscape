"""Configuration for the strategy_synthesizer extension.

Mirrors the wiki_synthesizer's settings but under the
``STRATEGY_SYNTHESIZER_*`` env prefix, and writes to a ``Playbooks/`` tree
instead of ``Wiki/``. Dark by default (``enabled=False``).
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class StrategySynthesizerSettings(BaseSettings):
    """Environment-driven settings for the trading-strategy playbook synthesizer."""

    enabled: bool = Field(
        default=False,
        description=(
            "Master switch. When False the cron no-ops. Defaults to False so "
            "the synthesizer stays dark until the operator opts in."
        ),
    )

    cron_hours: int = Field(
        default=6, ge=1, le=24,
        description="How often the synthesizer cron runs, in hours.",
    )

    max_memories_per_playbook: int = Field(
        default=200, ge=1, le=2000,
        description=(
            "Hard ceiling on how many source memories a single strategy playbook "
            "may aggregate (a book can produce a lot of rules)."
        ),
    )

    gemini_timeout_seconds: int = Field(
        default=300, ge=30, le=1800,
        description="Hard timeout on any single Gemini merge call.",
    )

    gemini_max_retries: int = Field(
        default=2, ge=0, le=5,
        description="Retry count for Gemini merge calls (1s/2s/4s backoff).",
    )

    obsidian_vault_path: str = Field(
        default="/data/vault",
        description="Root of the Obsidian vault (same value as the other extensions).",
    )

    model_config = {
        "env_prefix": "STRATEGY_SYNTHESIZER_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def vault_path(self) -> Path:
        return Path(self.obsidian_vault_path).expanduser().resolve()

    @property
    def playbook_dir(self) -> Path:
        """Root of the synthesized playbook tree (``{vault}/Playbooks``)."""
        return self.vault_path / "Playbooks"


strategy_synthesizer_settings = StrategySynthesizerSettings()
