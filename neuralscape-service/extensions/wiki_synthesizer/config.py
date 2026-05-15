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

    auto_build_communities: bool = Field(
        default=True,
        description=(
            "Before walking communities, call ``Graphiti.build_communities`` "
            "for any shared group_id that has zero Community nodes. "
            "Graphiti's incremental ``update_communities=True`` flag only "
            "refreshes EXISTING communities; without this pre-build, "
            "freshly populated groups would never produce synthesis output."
        ),
    )

    attach_window_seconds: int = Field(
        default=120,
        ge=10,
        le=1800,
        description=(
            "Time window for the ``attach_memory_id`` post-write Cypher "
            "patch. The patcher matches Graphiti nodes whose "
            "``created_at >= write_started_at - this``. Bump it on "
            "instances that see >2 minute graph writes (slow Gemini days, "
            "very large episodes); shrink to reduce false positives in "
            "high-concurrency settings."
        ),
    )

    gemini_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=1800,
        description=(
            "Hard timeout (seconds) on any single Gemini call inside the "
            "wiki synthesizer. Calls exceeding this are aborted and the "
            "community is recorded as an error in the synthesis result."
        ),
    )

    gemini_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description=(
            "Retry count for Gemini calls during synthesis. Each retry "
            "uses 1s, 2s, 4s exponential backoff. 0 disables retries; "
            "2 (default) gives one extra attempt after the first failure."
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
