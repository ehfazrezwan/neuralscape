"""Data-layer ingestion: chunk external documents into passages + distilled facts.

The ingest pipeline turns a fetched document (from a connector or a manual
``/v1/ingest`` call) into memories that carry a ``source_ref`` provenance
descriptor — so every memory can be traced back to its origin and a consuming
agent knows which tool/connector to call to retrieve more.
"""

from ingest.chunking import Chunk, chunk_text
from ingest.pipeline import IngestDoc, ingest_document

__all__ = ["Chunk", "chunk_text", "IngestDoc", "ingest_document"]
