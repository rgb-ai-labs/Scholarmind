from scholarmind.retrieval.dense import DenseResult
from scholarmind.retrieval.hybrid import hybrid_rank


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


def test_hybrid_rank_includes_sparse_only_chunk():
    dense_results = [
        _result("p1", 0, "dense only chunk", 0.9),
        _result("p2", 0, "shared chunk", 0.5),
    ]
    sparse_results = [
        _result("p3", 0, "sparse only chunk", 12.0),
        _result("p2", 0, "shared chunk", 8.0),
    ]

    result = hybrid_rank(dense_results, sparse_results)

    output_keys = {(c.paper_id, c.chunk_index) for c in result}
    assert ("p3", 0) in output_keys


def test_hybrid_rank_dedup_keeps_dense_object():
    dense_shared = _result("p2", 0, "shared chunk", 0.5)
    sparse_shared = _result("p2", 0, "shared chunk", 8.0)
    dense_results = [_result("p1", 0, "dense only", 0.9), dense_shared]
    sparse_results = [_result("p3", 0, "sparse only", 12.0), sparse_shared]

    result = hybrid_rank(dense_results, sparse_results)

    matches = [c for c in result if (c.paper_id, c.chunk_index) == ("p2", 0)]
    assert len(matches) == 1
    assert matches[0] is dense_shared
    assert matches[0] is not sparse_shared


def test_hybrid_rank_does_not_mutate_scores():
    dense_results = [
        _result("p1", 0, "dense only", 0.9),
        _result("p2", 0, "shared chunk", 0.5),
    ]
    sparse_results = [
        _result("p3", 0, "sparse only", 12.0),
        _result("p2", 0, "shared chunk", 8.0),
    ]

    result = hybrid_rank(dense_results, sparse_results)

    scores_by_key = {(c.paper_id, c.chunk_index): c.score for c in result}
    assert scores_by_key[("p1", 0)] == 0.9
    assert scores_by_key[("p2", 0)] == 0.5
    assert scores_by_key[("p3", 0)] == 12.0


def test_hybrid_rank_empty_lists():
    assert hybrid_rank([], []) == []


def test_hybrid_rank_dense_only_no_sparse():
    dense_only = [
        _result("p1", 0, "a", 0.9),
        _result("p2", 0, "b", 0.5),
        _result("p3", 0, "c", 0.1),
    ]

    result = hybrid_rank(dense_only, [])

    assert len(result) == len(dense_only)
    assert {(c.paper_id, c.chunk_index) for c in result} == {
        (c.paper_id, c.chunk_index) for c in dense_only
    }
