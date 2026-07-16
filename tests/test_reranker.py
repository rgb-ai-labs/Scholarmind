from scholarmind.config import Settings
from scholarmind.retrieval.dense import DenseResult
from scholarmind.retrieval.reranker import Reranker


def _result(paper_id: str, text: str, score: float) -> DenseResult:
    return DenseResult(
        paper_id=paper_id,
        title=None,
        authors=[],
        year=None,
        venue=None,
        section=None,
        page_start=1,
        page_end=1,
        chunk_index=0,
        text=text,
        score=score,
    )


def test_rerank_orders_most_relevant_candidate_first():
    settings = Settings()
    reranker = Reranker(settings.reranker_model)

    candidates = [
        _result("p1", "Page layout and margins in document typesetting conventions.", 0.5),
        _result(
            "p2",
            "Hallucination in language models occurs when the model generates "
            "factually incorrect or unsupported content not grounded in its training data.",
            0.1,
        ),
        _result("p3", "A brief history of typewriter keyboard layouts.", 0.4),
    ]

    result = reranker.rerank(
        "what causes hallucination in language models", candidates, top_k=3
    )

    assert result[0].paper_id == "p2"


def test_rerank_respects_top_k():
    settings = Settings()
    reranker = Reranker(settings.reranker_model)

    candidates = [
        _result("p1", "unrelated text about gardening", 0.5),
        _result("p2", "hallucination in language models", 0.1),
        _result("p3", "cooking recipes for pasta", 0.4),
    ]

    result = reranker.rerank(
        "what causes hallucination in language models", candidates, top_k=2
    )

    assert len(result) == 2


def test_rerank_empty_candidates_returns_empty_list():
    settings = Settings()
    reranker = Reranker(settings.reranker_model)

    result = reranker.rerank("anything", [], top_k=5)

    assert result == []


def test_rerank_does_not_mutate_scores():
    settings = Settings()
    reranker = Reranker(settings.reranker_model)

    candidates = [
        _result("p1", "hallucination in language models", 0.1),
        _result("p2", "unrelated gardening text", 0.9),
    ]
    original_scores = {c.paper_id: c.score for c in candidates}

    result = reranker.rerank(
        "what causes hallucination in language models", candidates, top_k=2
    )

    for c in result:
        assert c.score == original_scores[c.paper_id]
