"""Configuration for the wiki_synthesizer extension."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class SynthesizerSettings(BaseSettings):
    """Environment-driven settings for the wiki synthesizer.

    All keys live under the ``WIKI_SYNTHESIZER_*`` prefix to avoid
    colliding with NeuralScape core settings.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Master switch. When False, the cron job no-ops and the admin "
            "endpoint refuses to run. Defaults to False so the synthesizer "
            "stays dark until the operator opts in."
        ),
    )

    cron_hours: int = Field(
        default=6,
        ge=1,
        le=24,
        description=(
            "How often the synthesizer cron runs, in hours. Matches the "
            "dedup cron cadence by default."
        ),
    )

    max_memories_per_page: int = Field(
        default=50,
        ge=1,
        le=500,
        description=(
            "Hard ceiling on how many source memories any single wiki page "
            "may aggregate. Pages that exceed this trigger a 'split needed' "
            "log entry — v1 won't auto-split; that's a v2 feature."
        ),
    )

    gemini_model: str = Field(
        default="",
        description=(
            "Override the Gemini model used for incremental wiki merges. "
            "Empty string means inherit NeuralScape's default."
        ),
    )

    obsidian_vault_path: str = Field(
        default="/data/vault",
        description="Root of the Obsidian vault. Same value as the conversation_compiler.",
    )

    model_config = {
        "env_prefix": "WIKI_SYNTHESIZER_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def vault_path(self) -> Path:
        return Path(self.obsidian_vault_path).expanduser().resolve()

    @property
    def wiki_dir(self) -> Path:
        """Root of the synthesized wiki tree (``{vault}/Wiki``)."""
        return self.vault_path / "Wiki"


synthesizer_settings = SynthesizerSettings()
