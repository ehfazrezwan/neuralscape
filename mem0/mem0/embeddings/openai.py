import logging
import os
import re
import warnings
from typing import Literal, Optional

from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

logger = logging.getLogger(__name__)

# NS fork: some OpenAI-compatible backends (e.g. a Vertex gemini-embedding-001
# gateway endpoint) accept only ONE input per embeddings request and reject
# batched input with a message like "accepts only one input per request".
_SINGLE_INPUT_REJECTION_RE = re.compile(r"\b(?:only\s+)?(?:one|single|1)\s+input\b", re.IGNORECASE)


def _is_single_input_rejection(exc: Exception) -> bool:
    """True when the backend rejected a batched request because it only accepts one input."""
    return bool(_SINGLE_INPUT_REJECTION_RE.search(str(exc)))


class OpenAIEmbedding(EmbeddingBase):
    DEFAULT_MAX_BATCH = 100
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "text-embedding-3-small"
        # Only pass `dimensions` to the API when the user set embedding_dims; non-matryoshka
        # OpenAI-compatible backends (vLLM, Voyage, etc.) reject the parameter
        self._pass_dimensions_to_api = self.config.embedding_dims is not None
        self.config.embedding_dims = self.config.embedding_dims or 1536

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = (
            self.config.openai_base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        if os.environ.get("OPENAI_API_BASE"):
            warnings.warn(
                "The environment variable 'OPENAI_API_BASE' is deprecated and will be removed in the 0.1.80. "
                "Please use 'OPENAI_BASE_URL' instead.",
                DeprecationWarning,
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        kwargs = {
            "input": [text],
            "model": self.config.model,
            "encoding_format": "float",
        }
        if self._pass_dimensions_to_api:
            kwargs["dimensions"] = self.config.embedding_dims
        return self.client.embeddings.create(**kwargs).data[0].embedding

    def _create_embeddings(self, chunk: list) -> list:
        """One embeddings API call for ``chunk``; returns vectors in input order."""
        kwargs = {
            "input": chunk,
            "model": self.config.model,
            "encoding_format": "float",
        }
        if self._pass_dimensions_to_api:
            kwargs["dimensions"] = self.config.embedding_dims
        response = self.client.embeddings.create(**kwargs)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    def embed_batch(self, texts, memory_action="add"):
        """Embed multiple texts, chunking into batched OpenAI API calls.

        Chunk size defaults to 100 (API limit) and can be lowered via
        ``config.embedding_batch_size`` — NS fork: OpenAI-compatible backends
        that accept only one input per request (e.g. a Vertex
        gemini-embedding-001 gateway endpoint) need ``embedding_batch_size=1``.
        As a safety net, a batched call rejected with a single-input error is
        retried per item; any other failure propagates to the caller.
        """
        configured = getattr(self.config, "embedding_batch_size", None)
        max_batch = max(1, min(configured or self.DEFAULT_MAX_BATCH, self.DEFAULT_MAX_BATCH))
        texts = [text.replace("\n", " ") for text in texts]
        all_embeddings = []
        for i in range(0, len(texts), max_batch):
            chunk = texts[i : i + max_batch]
            try:
                all_embeddings.extend(self._create_embeddings(chunk))
            except Exception as e:
                if len(chunk) > 1 and _is_single_input_rejection(e):
                    logger.warning(
                        "Batched embed rejected by single-input-only backend (%s); "
                        "retrying %d inputs individually. Set embedding_batch_size=1 "
                        "to skip the failed batch attempt.",
                        e,
                        len(chunk),
                    )
                    for text in chunk:
                        all_embeddings.extend(self._create_embeddings([text]))
                else:
                    raise
        return all_embeddings
