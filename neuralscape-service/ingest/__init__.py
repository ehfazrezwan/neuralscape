"""Data-layer ingestion: chunk external documents into passages + distilled facts.

The ingest pipeline turns a fetched document (from a connector or a manual
``/v1/ingest`` call) into memories that carry a ``source_ref`` provenance
descriptor — so every memory can be traced back to its origin and a consuming
agent knows which tool/connector to call to retrieve more.
"""

from ingest.archive import ArchiveError, ArchiveTooLarge, is_zip, iter_archive
from ingest.chunking import Chunk, chunk_text
from ingest.extract import UnsupportedFile, extract_text
from ingest.pipeline import IngestDoc, ingest_document
from ingest.storage import (
    StoredArtifact,
    artifact_source_ref,
    find_artifact,
    read_artifact,
    store_artifact,
)

__all__ = [
    "Chunk",
    "chunk_text",
    "IngestDoc",
    "ingest_document",
    "extract_text",
    "UnsupportedFile",
    "iter_archive",
    "is_zip",
    "ArchiveError",
    "ArchiveTooLarge",
    "StoredArtifact",
    "store_artifact",
    "read_artifact",
    "find_artifact",
    "artifact_source_ref",
]
