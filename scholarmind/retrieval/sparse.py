from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from scholarmind.retrieval.dense import DenseResult


def sparse_search(
    query: str,
    qdrant_path: str,
    collection_name: str,
    limit: int,
) -> list[DenseResult]:
    client = QdrantClient(path=qdrant_path)
    try:
        if not client.collection_exists(collection_name):
            return []

        points = []
        offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=collection_name,
                with_payload=True,
                limit=256,
                offset=offset,
            )
            points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        if not points:
            return []

        corpus_tokens = [point.payload["text"].lower().split() for point in points]
        bm25 = BM25Okapi(corpus_tokens)

        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )[:limit]

        return [
            DenseResult(**points[index].payload, score=float(scores[index]))
            for index in ranked_indices
        ]
    finally:
        client.close()
