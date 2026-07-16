from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from scholarmind.config import get_settings
from scholarmind.ingestion.chunker import chunk_document
from scholarmind.ingestion.embedder import Embedder
from scholarmind.ingestion.loader import load_path
from scholarmind.ingestion.parser import parse_document
from scholarmind.ingestion.store import ChunkStore

if TYPE_CHECKING:
    from scholarmind.config import Settings


@dataclass
class IngestResult:
    papers_ingested: int
    chunks_created: int
    collection_name: str


def run_ingestion(path: Path, settings: "Settings | None" = None) -> IngestResult:
    settings = settings or get_settings()

    raw_documents = load_path(Path(path))

    embedder = Embedder(settings.embedding_model)
    store = ChunkStore(
        qdrant_path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        vector_size=embedder.dimension,
    )

    papers_ingested = 0
    chunks_created = 0

    try:
        for raw in raw_documents:
            parsed = parse_document(raw)
            chunks = chunk_document(parsed, settings.chunk_size, settings.chunk_overlap)

            if chunks:
                vectors = embedder.embed_chunks(chunks)
                store.upsert_chunks(chunks, vectors)

            papers_ingested += 1
            chunks_created += len(chunks)
    finally:
        store.close()

    return IngestResult(
        papers_ingested=papers_ingested,
        chunks_created=chunks_created,
        collection_name=settings.qdrant_collection,
    )
