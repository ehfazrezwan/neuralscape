import os
from typing import Literal, Optional

from google import genai
from google.genai import types

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class GoogleGenAIEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "models/gemini-embedding-001"
        self.config.embedding_dims = self.config.embedding_dims or self.config.output_dimensionality or 768

        api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")

        # NEURALSCAPE PATCH: bound every request (HttpOptions.timeout is in milliseconds) so a
        # stalled endpoint cannot block the caller indefinitely. None keeps the SDK default.
        timeout_s = getattr(self.config, "request_timeout_s", None)
        http_options = types.HttpOptions(timeout=max(1, int(timeout_s * 1000))) if timeout_s else None
        self.client = genai.Client(api_key=api_key, http_options=http_options)

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using Google Generative AI.
        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")

        # Create config for embedding parameters
        config = types.EmbedContentConfig(output_dimensionality=self.config.embedding_dims)

        # Call the embed_content method with the correct parameters
        response = self.client.models.embed_content(model=self.config.model, contents=text, config=config)

        return response.embeddings[0].values

    def embed_batch(self, texts: list[str], memory_action: Optional[Literal["add", "search", "update"]] = None) -> list[list]:
        """Batch-embed multiple texts in a single Gemini API call.

        Args:
            texts: List of texts to embed.
            memory_action: Unused, kept for interface compatibility.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []

        cleaned = [t.replace("\n", " ") for t in texts]
        config = types.EmbedContentConfig(output_dimensionality=self.config.embedding_dims)

        response = self.client.models.embed_content(
            model=self.config.model,
            contents=cleaned,
            config=config,
        )

        return [e.values for e in response.embeddings]
