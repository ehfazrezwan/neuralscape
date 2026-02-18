"""Quickstart script for Graphiti with a custom Neo4j database and Google Gemini."""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[2] / '.env')

# Neo4j connection
neo4j_uri = os.environ['NEO4J_URI']
neo4j_user = os.environ['NEO4J_USER']
neo4j_password = os.environ['NEO4J_PASSWORD']
neo4j_database = os.environ.get('NEO4J_DATABASE', 'memory')

# Google AI Studio
google_api_key = os.environ['GOOGLE_API_KEY']
model_name = os.environ.get('MODEL_NAME', 'gemini-2.5-flash')
small_model_name = os.environ.get('SMALL_MODEL_NAME', 'gemini-2.5-flash-lite')
embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME', 'text-embedding-001')


async def main():
    # 1. Create driver targeting the 'memory' database
    driver = Neo4jDriver(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
    )

    # 2. Gemini LLM client
    llm_client = GeminiClient(
        config=LLMConfig(
            api_key=google_api_key,
            model=model_name,
            small_model=small_model_name,
        ),
    )

    # 3. Gemini reranker (cross-encoder)
    cross_encoder = GeminiRerankerClient(
        config=LLMConfig(
            api_key=google_api_key,
            model=small_model_name,
        ),
    )

    # 4. Gemini embedder
    embedder = GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=google_api_key,
            embedding_model=embedding_model_name,
        ),
    )

    # 5. Assemble Graphiti
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    try:
        # 5. Add sample episodes
        episodes = [
            {
                'content': (
                    'Kamala Harris is the Attorney General of California. She was previously '
                    'the district attorney for San Francisco.'
                ),
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            {
                'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'state': 'California',
                    'previous_role': 'Lieutenant Governor',
                    'previous_location': 'San Francisco',
                },
                'type': EpisodeType.json,
                'description': 'podcast metadata',
            },
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'term_start': 'January 7, 2019',
                    'term_end': 'Present',
                },
                'type': EpisodeType.json,
                'description': 'podcast metadata',
            },
        ]

        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'Freakonomics Radio {i}',
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content']),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added episode: Freakonomics Radio {i} ({episode["type"].value})')

        # 6. Edge search (hybrid semantic + BM25)
        print("\nSearching for: 'Who was the California Attorney General?'")
        results = await graphiti.search('Who was the California Attorney General?')

        print('\nSearch Results:')
        for result in results:
            print(f'UUID: {result.uuid}')
            print(f'Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')
            print('---')

        # 7. Node search using a search recipe
        print('\nPerforming node search with NODE_HYBRID_SEARCH_RRF:')
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 5

        node_search_results = await graphiti._search(
            query='California Governor',
            config=node_search_config,
        )

        print('\nNode Search Results:')
        for node in node_search_results.nodes:
            print(f'Node UUID: {node.uuid}')
            print(f'Node Name: {node.name}')
            summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
            print(f'Summary: {summary}')
            print(f'Labels: {", ".join(node.labels)}')
            print('---')

    finally:
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())
