import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from scholarmind.config import get_settings
from scholarmind.ingestion.chunker import chunk_document
from scholarmind.ingestion.loader import load_path
from scholarmind.ingestion.parser import ParsedDocument, ParsedSection, parse_document
from scholarmind.ingestion.store import ChunkStore
from scholarmind.model_cache import get_embedder
from scholarmind.retrieval.papers import list_papers, normalize_title

if TYPE_CHECKING:
    from scholarmind.config import Settings


@dataclass
class IngestResult:
    papers_ingested: int
    chunks_created: int
    collection_name: str
    # Titles that already existed in the library under a different paper_id at ingestion
    # time — a likely accidental duplicate (e.g. a re-downloaded/regenerated PDF whose bytes
    # differ from the copy already ingested). Non-blocking: the paper is still ingested: run
    # `scholarmind dedupe` to review and remove the extra copy.
    duplicate_title_warnings: list[str] = field(default_factory=list)


def _images_dir_for(settings: "Settings") -> Path:
    # Mirrors scholarmind.discovery.ingest's one-line derivation of a sibling storage folder
    # from qdrant_path — kept local rather than shared to avoid an extra cross-module import
    # for a single line.
    return Path(settings.qdrant_path).resolve().parent / "figures"


def run_ingestion(path: Path, settings: "Settings | None" = None) -> IngestResult:
    settings = settings or get_settings()

    raw_documents = load_path(Path(path), images_dir=_images_dir_for(settings))

    # Snapshot existing titles once, up front, so a duplicate PDF (byte-different from what's
    # already ingested, e.g. a re-downloaded or regenerated copy) can be flagged even though its
    # content-hash paper_id makes it look like a brand-new paper to the store itself.
    existing_paper_ids_by_title: dict[str, set[str]] = {}
    for paper in list_papers(settings):
        key = normalize_title(paper.label)
        if key is not None:
            existing_paper_ids_by_title.setdefault(key, set()).add(paper.paper_id)

    embedder = get_embedder(settings.embedding_model)
    store = ChunkStore(
        qdrant_path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        vector_size=embedder.dimension,
    )

    papers_ingested = 0
    chunks_created = 0
    duplicate_title_warnings: list[str] = []

    try:
        for raw in raw_documents:
            parsed = parse_document(raw)

            title_key = normalize_title(parsed.title)
            if title_key is not None:
                known_ids = existing_paper_ids_by_title.get(title_key, set())
                if known_ids and parsed.paper_id not in known_ids:
                    duplicate_title_warnings.append(parsed.title)

            chunks = chunk_document(
                parsed, settings.chunk_size, settings.chunk_overlap, ingested_at=time.time()
            )

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
        duplicate_title_warnings=duplicate_title_warnings,
    )


def ingest_metadata_record(
    paper_id: str,
    title: str | None,
    authors: list[str],
    year: int | None,
    venue: str | None,
    text: str,
    source_filename: str | None = None,
    doi: str | None = None,
    settings: "Settings | None" = None,
) -> IngestResult:
    # Ingests a single title+abstract record with no source PDF (e.g. a literature-discovery
    # candidate with no open-access PDF available). Deliberately duplicates run_ingestion's
    # embed/store sequence rather than sharing it, since run_ingestion batches an embedder and
    # store across every document in a directory and this is always exactly one document.
    settings = settings or get_settings()

    parsed = ParsedDocument(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        sections=[ParsedSection(heading="Abstract", text=text, page_start=1, page_end=1)],
        source_filename=source_filename,
        doi=doi,
        is_metadata_only=True,
    )

    embedder = get_embedder(settings.embedding_model)
    store = ChunkStore(
        qdrant_path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        vector_size=embedder.dimension,
    )

    try:
        chunks = chunk_document(
            parsed, settings.chunk_size, settings.chunk_overlap, ingested_at=time.time()
        )
        if chunks:
            vectors = embedder.embed_chunks(chunks)
            store.upsert_chunks(chunks, vectors)
    finally:
        store.close()

    return IngestResult(
        papers_ingested=1,
        chunks_created=len(chunks),
        collection_name=settings.qdrant_collection,
    )
