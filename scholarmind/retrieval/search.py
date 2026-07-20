from scholarmind.config import Settings, get_settings
from scholarmind.guardrails import passes_confidence
from scholarmind.model_cache import get_embedder, get_reranker
from scholarmind.retrieval.dense import DenseResult, dense_search
from scholarmind.retrieval.hybrid import hybrid_rank
from scholarmind.retrieval.sparse import sparse_search


def search(
    query: str,
    settings: "Settings | None" = None,
    paper_id: str | None = None,
    paper_ids: list[str] | None = None,
) -> list[DenseResult]:
    if settings is None:
        settings = get_settings()

    embedder = get_embedder(settings.embedding_model)

    dense_candidates = dense_search(
        query,
        embedder,
        settings.qdrant_path,
        settings.qdrant_collection,
        settings.retrieval_candidate_k,
        paper_id=paper_id,
        paper_ids=paper_ids,
    )

    sparse_candidates = sparse_search(
        query,
        settings.qdrant_path,
        settings.qdrant_collection,
        settings.retrieval_candidate_k,
        paper_id=paper_id,
        paper_ids=paper_ids,
    )

    if not dense_candidates and not sparse_candidates:
        return []

    hybrid_candidates = hybrid_rank(dense_candidates, sparse_candidates)

    reranker = get_reranker(settings.reranker_model)

    scored = reranker.rerank_with_scores(
        query, hybrid_candidates, settings.retrieval_top_k
    )

    return [
        candidate
        for candidate, score in scored
        if passes_confidence(score, settings)
    ]
