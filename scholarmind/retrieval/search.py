from scholarmind.config import Settings, get_settings
from scholarmind.guardrails import passes_confidence
from scholarmind.ingestion.embedder import Embedder
from scholarmind.retrieval.dense import DenseResult, dense_search
from scholarmind.retrieval.hybrid import hybrid_rank
from scholarmind.retrieval.reranker import Reranker
from scholarmind.retrieval.sparse import sparse_search


def search(query: str, settings: "Settings | None" = None) -> list[DenseResult]:
    if settings is None:
        settings = get_settings()

    embedder = Embedder(settings.embedding_model)

    dense_candidates = dense_search(
        query,
        embedder,
        settings.qdrant_path,
        settings.qdrant_collection,
        settings.retrieval_candidate_k,
    )

    sparse_candidates = sparse_search(
        query,
        settings.qdrant_path,
        settings.qdrant_collection,
        settings.retrieval_candidate_k,
    )

    if not dense_candidates and not sparse_candidates:
        return []

    hybrid_candidates = hybrid_rank(dense_candidates, sparse_candidates)

    reranker = Reranker(settings.reranker_model)

    scored = reranker.rerank_with_scores(
        query, hybrid_candidates, settings.retrieval_top_k
    )

    return [
        candidate
        for candidate, score in scored
        if passes_confidence(score, settings)
    ]
