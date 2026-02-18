import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Gemini
    google_api_key: str = ""
    gemini_llm_model: str = "gemini-2.5-flash"
    gemini_embedder_model: str = "text-embedding-004"

    # Neo4j
    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "memory"

    # Graphiti options
    store_raw_episode_content: bool = True
    update_communities: bool = False

    # Service
    host: str = "0.0.0.0"
    port: int = 8199
    default_user_id: str = "default_user"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def get_mem0_config(self) -> dict:
        """Build mem0 config dict for Memory(config=...)."""
        return {
            "graph_store": {
                "provider": "graphiti",
                "config": {
                    "url": self.neo4j_uri,
                    "username": self.neo4j_user,
                    "password": self.neo4j_password,
                    "database": self.neo4j_database,
                    "graphiti_llm_provider": "gemini",
                    "graphiti_llm_model": self.gemini_llm_model,
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
