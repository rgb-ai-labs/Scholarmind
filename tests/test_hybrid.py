import pytest

from scholarmind.retrieval.dense import DenseResult
from scholarmind.retrieval.hybrid import hybrid_rank, keyword_score


def _result(paper_id: str, chunk_index: int, text: str, score: float) -> DenseResult:
    return DenseResult(
        paper_id=paper_id,
        title=None,
        authors=[],
        year=None,
        venue=None,
        section=None,
        page_start=1,
        page_end=1,
        chunk_index=chunk_index,
        text=text,
        score=score,
    )


def test_keyword_score_all_terms_match():
    score = keyword_score(
        "retrieval augmented generation",
        "This paper introduces retrieval augmented generation for open-domain QA.",
    )
    assert score == 1.0


def test_keyword_score_one_of_three_terms_match():
    score = keyword_score(
        "retrieval augmented generation",
        "This paper is about generation of images.",
    )
    assert score == pytest.approx(1 / 3)


def test_keyword_score_no_terms_match():
    score = keyword_score(
        "retrieval augmented generation",
        "This text is unrelated to the query terms.",
    )
    assert score == 0.0


def test_keyword_score_empty_query():
    score = keyword_score("", "some text")
    assert score == 0.0


def test_hybrid_rank_reorders_based_on_fusion():
    candidates = [
        _result("p1", 0, "completely unrelated text about gardening", 0.9),
        _result("p2", 0, "retrieval augmented generation retrieval augmented generation", 0.5),
        _result("p3", 0, "retrieval only", 0.4),
        _result("p4", 0, "another unrelated snippet about cooking", 0.3),
    ]

    result = hybrid_rank("retrieval augmented generation", candidates)

    assert [c.paper_id for c in result] != [c.paper_id for c in candidates]

    assert len(result) == len(candidates)
    input_keys = {(c.paper_id, c.chunk_index) for c in candidates}
    output_keys = {(c.paper_id, c.chunk_index) for c in result}
    assert input_keys == output_keys


def test_hybrid_rank_does_not_mutate_scores():
    candidates = [
        _result("p1", 0, "retrieval augmented generation", 0.9),
        _result("p2", 0, "unrelated", 0.1),
    ]
    original_scores = {c.paper_id: c.score for c in candidates}

    result = hybrid_rank("retrieval augmented generation", candidates)

    for c in result:
        assert c.score == original_scores[c.paper_id]


def test_hybrid_rank_empty_list():
    assert hybrid_rank("retrieval augmented generation", []) == []
