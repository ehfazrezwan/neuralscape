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

# Locked models for controlled comparison (MUST match NS Track A exactly).
# The vendored mem0 GeminiLLM passes ``config.model`` straight through to the
# google-genai client (see mem0/mem0/llms/gemini.py), so the SAME backbone id
# NS uses works verbatim — this is the whole point of the control. config.py is
# the single source of truth for the model ids; report.py reads these constants.
BACKBONE_MODEL = "gemini-3.1-flash-lite"
EMBEDDER_MODEL = "gemini-embedding-001"
JUDGE_MODEL = "gemini-3.1-flash-lite"
JUDGE_TEMP = 0.0

# Answer generation model (same as backbone — single source of truth).
ANSWER_MODEL = BACKBONE_MODEL
ANSWER_TEMP = 0.0

# mem0 provider ids (see mem0/mem0/utils/factory.py provider registry).
# Both the LLM and the embedder are registered under the "gemini" provider key.
LLM_PROVIDER = "gemini"
EMBEDDER_PROVIDER = "gemini"
EMBEDDING_DIMS = 768


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
                "provider": LLM_PROVIDER,
                "config": {
                    "model": BACKBONE_MODEL,
                    "temperature": 0.0,
                    "api_key": self.api_key,
                }
            },
            "embedder": {
                "provider": EMBEDDER_PROVIDER,
                "config": {
                    "model": EMBEDDER_MODEL,
                    "embedding_dims": EMBEDDING_DIMS,
                    "api_key": self.api_key,
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0_tracka",
                    "embedding_model_dims": EMBEDDING_DIMS,
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
