import logging
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.cross_encoder.client import CrossEncoderClient  # type: ignore
from graphiti_core.driver.driver import GraphDriver  # type: ignore
from graphiti_core.driver.neo4j_driver import Neo4jDriver  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.embedder import EmbedderClient  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

from graph_service.config import Settings, ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)


class ZepGraphiti(Graphiti):
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        llm_client: LLMClient | None = None,
        embedder: EmbedderClient | None = None,
        cross_encoder: CrossEncoderClient | None = None,
        graph_driver: GraphDriver | None = None,
    ):
        super().__init__(
            uri=uri,
            user=user,
            password=password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            graph_driver=graph_driver,
        )

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
        return new_node

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)

        for node in nodes:
            await node.delete(self.driver)

        for episode in episodes:
            await episode.delete(self.driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


def _create_llm_client(settings: Settings) -> LLMClient | None:
    provider = settings.llm_provider.lower()

    if provider == 'gemini':
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.gemini_client import GeminiClient

        if not settings.google_api_key:
            raise ValueError('GOOGLE_API_KEY is required when llm_provider is gemini')

        llm_config = LLMConfig(
            api_key=settings.google_api_key,
            model=settings.model_name,
            small_model=settings.small_model_name,
        )
        return GeminiClient(config=llm_config)

    elif provider == 'openai':
        from graphiti_core.llm_client import OpenAIClient
        from graphiti_core.llm_client.config import LLMConfig

        if not settings.openai_api_key:
            raise ValueError('OPENAI_API_KEY is required when llm_provider is openai')

        llm_config = LLMConfig(
            api_key=settings.openai_api_key,
            model=settings.model_name,
            base_url=settings.openai_base_url,
            small_model=settings.small_model_name,
        )
        return OpenAIClient(config=llm_config)

    else:
        raise ValueError(f'Unsupported llm_provider: {provider}')


def _create_embedder(settings: Settings) -> EmbedderClient | None:
    provider = settings.embedding_provider.lower()

    if provider == 'gemini':
        from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

        if not settings.google_api_key:
            raise ValueError('GOOGLE_API_KEY is required when embedding_provider is gemini')

        config = GeminiEmbedderConfig(
            api_key=settings.google_api_key,
            embedding_model=settings.embedding_model_name or 'gemini-embedding-001',
        )
        return GeminiEmbedder(config=config)

    elif provider == 'openai':
        from graphiti_core.embedder import OpenAIEmbedder
        from graphiti_core.embedder.openai import OpenAIEmbedderConfig

        config = OpenAIEmbedderConfig(
            api_key=settings.openai_api_key,
            embedding_model=settings.embedding_model_name or 'text-embedding-3-small',
            base_url=settings.openai_base_url,
        )
        return OpenAIEmbedder(config=config)

    else:
        raise ValueError(f'Unsupported embedding_provider: {provider}')


def _create_cross_encoder(settings: Settings) -> CrossEncoderClient:
    provider = settings.llm_provider.lower()

    if provider == 'gemini':
        from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
        from graphiti_core.llm_client.config import LLMConfig

        config = LLMConfig(api_key=settings.google_api_key)
        return GeminiRerankerClient(config=config)

    else:
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

        return OpenAIRerankerClient()


async def get_graphiti(settings: ZepEnvDep):
    llm_client = _create_llm_client(settings)
    embedder = _create_embedder(settings)
    cross_encoder = _create_cross_encoder(settings)

    driver = Neo4jDriver(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )

    client = ZepGraphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )
    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: Settings):
    llm_client = _create_llm_client(settings)
    embedder = _create_embedder(settings)
    cross_encoder = _create_cross_encoder(settings)

    driver = Neo4jDriver(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )

    client = ZepGraphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )
    await client.build_indices_and_constraints()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
