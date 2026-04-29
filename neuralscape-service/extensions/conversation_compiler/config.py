"""Configuration for the conversation-compiler extension."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class CompilerSettings(BaseSettings):
    """Configuration for the conversation-compiler extension.

    All values are read from environment variables with sensible defaults.
    """

    obsidian_vault_path: str = Field(
        default="/data/vault",
        description="Root path to the Obsidian vault (set via OBSIDIAN_VAULT_PATH env var)",
    )

    compiler_llm_model: str = Field(
        default="",
        description="LLM model for extraction/compilation. Empty string = use NeuralScape default.",
    )

    compile_after_hour: int = Field(
        default=18,
        ge=0,
        le=23,
        description="Hour (24h) after which auto-compilation runs (default: 18 = 6 PM)",
    )

    auto_compile: bool = Field(
        default=True,
        description="Whether to auto-compile daily logs after compile_after_hour",
    )

    neuralscape_url: str = Field(
        default="http://localhost:8199",
        description="NeuralScape API base URL (used for health checks only)",
    )

    model_config = {
        "env_prefix": "",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def vault_path(self) -> Path:
        """Return the vault path as a resolved Path object."""
        return Path(self.obsidian_vault_path).expanduser().resolve()

    def get_llm_model(self, fallback: str) -> str:
        """Return the configured LLM model, or the fallback if not set."""
        return self.compiler_llm_model or fallback


compiler_settings = CompilerSettings()
