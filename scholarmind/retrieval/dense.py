from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from scholarmind.ingestion.embedder import Embedder


@dataclass
class DenseResult:
    paper_id: str
    title: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    section: str | None
    page_start: int
    page_end: int
    chunk_index: int
    text: str
    score: float
    source_filename: str | None = None
    ingested_at: float | None = None
    doi: str | None = None
    is_metadata_only: bool = False
    chunk_type: str = "text"
    image_path: str | None = None
    resolved_doi: str | None = None
    resolved_authors: list[str] | None = None
    resolved_year: int | None = None
    resolved_venue: str | None = None
    resolved_source: str | None = None


def paper_scope_filter(
    paper_id: str | None = None, paper_ids: list[str] | None = None
) -> Filter | None:
    # Shared by dense_search and sparse_search. paper_id (single paper, e.g. Ask/Summarize)
    # and paper_ids (multiple papers, e.g. Writing's scope) are mutually exclusive — paper_id
    # wins if both are somehow passed.
    if paper_id is not None:
        return Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))])
    if paper_ids:
        return Filter(must=[FieldCondition(key="paper_id", match=MatchAny(any=paper_ids))])
    return None


def dense_search(
    query: str,
    embedder: Embedder,
    qdrant_path: str,
    collection_name: str,
    limit: int,
    paper_id: str | None = None,
    paper_ids: list[str] | None = None,
) -> list[DenseResult]:
    query_vector = embedder.embed_text(query)

    client = QdrantClient(path=qdrant_path)
    try:
        if not client.collection_exists(collection_name):
            return []

        query_filter = paper_scope_filter(paper_id, paper_ids)

        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )

        return [
            DenseResult(**point.payload, score=point.score)
            for point in response.points
        ]
    finally:
        client.close()
