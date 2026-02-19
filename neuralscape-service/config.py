from pathlib import Path
from urllib.parse import urlparse

from arq.connections import RedisSettings
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Gemini
    google_api_key: str = ""
    gemini_llm_model: str = "gemini-3-flash-preview"
    gemini_llm_fallback_model: str = "gemini-2.5-flash"
    gemini_embedder_model: str = "gemini-embedding-001"

    # Neo4j
    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "memory"

    # Graphiti options
    store_raw_episode_content: bool = True
    update_communities: bool = False

    # Qdrant vector store
    qdrant_url: str | None = None  # e.g. "http://localhost:6333" — if set, uses Qdrant server mode
    qdrant_on_disk: bool = True
    qdrant_path: str = "~/.neuralscape/qdrant"  # only used when qdrant_url is not set
    qdrant_collection: str = "neuralscape_memories"

    # Redis / ARQ
    redis_url: str = "redis://localhost:6379"
    arq_queue_name: str = "neuralscape:queue"
    arq_max_retries: int = 3
    arq_job_timeout: int = 300  # 5 min max per task

    # LLM retry (exponential backoff for transient 503/429 errors)
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0  # seconds
    llm_retry_max_delay: float = 30.0  # seconds

    # Dedup cron
    dedup_similarity_threshold: float = 0.95
    dedup_batch_size: int = 100
    dedup_cron_hours: set = {0, 6, 12, 18}

    # Service
    host: str = "0.0.0.0"
    port: int = 8199
    default_user_id: str = "default_user"
    default_project_id: str | None = None
    mcp_transport: str = "stdio"  # "stdio" or "http"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def validate_required(self) -> None:
        """Validate that all required configuration fields are set.

        Raises:
            ValueError: If any required field is empty or missing.
        """
        errors = []
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required but not set")
        if not self.neo4j_password:
            errors.append("NEO4J_PASSWORD is required but not set")
        if not self.neo4j_uri:
            errors.append("NEO4J_URI is required but not set")
        if not self.redis_url:
            errors.append("REDIS_URL is required but not set")
        if errors:
            raise ValueError(
                "Missing required configuration:\n  - " + "\n  - ".join(errors)
            )

    def get_mem0_config(self) -> dict:
        """Build mem0 config dict for Memory(config=...)."""
        # Qdrant: server mode (url) or local on-disk mode (path)
        qdrant_config: dict = {
            "collection_name": self.qdrant_collection,
            "embedding_model_dims": 768,
        }
        if self.qdrant_url:
            qdrant_config["url"] = self.qdrant_url
        else:
            qdrant_config["path"] = str(Path(self.qdrant_path).expanduser())
            qdrant_config["on_disk"] = self.qdrant_on_disk

        return {
            "llm": {
                "provider": "gemini",
                "config": {
                    "model": self.gemini_llm_model,
                    "api_key": self.google_api_key,
                },
            },
            "embedder": {
                "provider": "gemini",
                "config": {
                    "model": self.gemini_embedder_model,
                    "api_key": self.google_api_key,
                    "embedding_dims": 768,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": qdrant_config,
            },
            "graph_store": {
                "provider": "graphiti",
                "config": {
                    "url": self.neo4j_uri,
                    "username": self.neo4j_user,
                    "password": self.neo4j_password,
                    "database": self.neo4j_database,
                    "graphiti_llm_provider": "gemini",
                    "graphiti_llm_model": self.gemini_llm_model,
                    "graphiti_llm_fallback_model": self.gemini_llm_fallback_model,
                    "graphiti_llm_api_key": self.google_api_key,
                    "graphiti_embedder_provider": "gemini",
                    "graphiti_embedder_model": self.gemini_embedder_model,
                    "graphiti_embedder_api_key": self.google_api_key,
                    "graphiti_reranker_provider": "gemini",
                    "store_raw_episode_content": self.store_raw_episode_content,
                    "update_communities": self.update_communities,
                },
            },
            "version": "v1.1",
        }


settings = Settings()


def parse_redis_settings() -> RedisSettings:
    """Parse redis_url into ARQ RedisSettings using urllib.parse for robustness.

    Handles formats: redis://host:port/db, redis://user:password@host:port/db
    """
    parsed = urlparse(settings.redis_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    database = int(parsed.path.lstrip("/")) if parsed.path and parsed.path.strip("/") else 0
    password = parsed.password

    return RedisSettings(
        host=host,
        port=port,
        database=database,
        password=password,
        conn_timeout=10,
        conn_retries=5,
        conn_retry_delay=2,
        retry_on_timeout=True,
    )
