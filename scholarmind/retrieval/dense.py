from dataclasses import dataclass

from qdrant_client import QdrantClient

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


def dense_search(
    query: str,
    embedder: Embedder,
    qdrant_path: str,
    collection_name: str,
    limit: int,
) -> list[DenseResult]:
    query_vector = embedder._model.encode([query], convert_to_numpy=True)[0].tolist()

    client = QdrantClient(path=qdrant_path)
    try:
        if not client.collection_exists(collection_name):
            return []

        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return [
            DenseResult(**point.payload, score=point.score)
            for point in response.points
        ]
    finally:
        client.close()
