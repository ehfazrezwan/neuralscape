"""Configuration for mem0 Track A control runs.

All models locked to match NS Track A:
- backbone: gemini-3.1-flash-lite
- embedder: gemini-embedding-001
- judge: gemini-3.1-flash-lite (temp 0)
- vector store: Qdrant local/on-disk (isolated from production)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Locked models for controlled comparison
BACKBONE_MODEL = "gemini-flash-1.5"  # mem0 config format for Gemini 1.5 Flash
EMBEDDER_MODEL = "gemini-embedding-001"
JUDGE_MODEL = "gemini-3.1-flash-lite"
JUDGE_TEMP = 0.0

# Answer generation model (same as backbone)
ANSWER_MODEL = "gemini-3.1-flash-lite"
ANSWER_TEMP = 0.0


@dataclass
class Mem0Config:
    """mem0 Memory instance config."""

    api_key: str
    vector_store_path: Path

    def to_mem0_dict(self) -> dict:
        """Build the config dict for mem0.Memory initialization.

        Returns a config matching mem0's expected schema:
        {
          "llm": {"provider": "...", "config": {...}},
          "embedder": {"provider": "...", "config": {...}},
          "vector_store": {"provider": "...", "config": {...}}
        }
        """
        return {
            "llm": {
                "provider": "google-genai",
                "config": {
                    "model": BACKBONE_MODEL,
                    "temperature": 0.0,
                    "api_key": self.api_key,
                }
            },
            "embedder": {
                "provider": "google-genai",
                "config": {
                    "model": EMBEDDER_MODEL,
                    "api_key": self.api_key,
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0_tracka",
                    "embedding_model_dims": 768,
                    "path": str(self.vector_store_path),
                    "on_disk": True,
                }
            },
            "version": "v1.1",
        }


def get_config(vector_store_path: Path | None = None) -> Mem0Config:
    """Build config from environment, with isolated vector store."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable required")

    if vector_store_path is None:
        # Default to isolated path under trackb/
        base = Path(__file__).parent.parent
        vector_store_path = base / ".mem0_tracka_qdrant"

    return Mem0Config(api_key=api_key, vector_store_path=vector_store_path)
