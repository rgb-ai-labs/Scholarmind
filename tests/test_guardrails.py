from scholarmind.config import Settings
from scholarmind.guardrails import passes_confidence, refusal_threshold, split_citation_markers


def test_refusal_threshold() -> None:
    settings = Settings(retrieval_min_rerank_score=-7.0)
    assert refusal_threshold(settings) == -7.0


def test_passes_confidence_above_threshold() -> None:
    settings = Settings(retrieval_min_rerank_score=-7.0)
    assert passes_confidence(0.0, settings) is True


def test_passes_confidence_below_threshold() -> None:
    settings = Settings(retrieval_min_rerank_score=-7.0)
    assert passes_confidence(-10.0, settings) is False


def test_passes_confidence_at_boundary() -> None:
    settings = Settings(retrieval_min_rerank_score=-7.0)
    assert passes_confidence(-7.0, settings) is True


def test_split_citation_markers_all_valid() -> None:
    assert split_citation_markers([1, 2, 3], 3) == ([1, 2, 3], [])


def test_split_citation_markers_mixed() -> None:
    assert split_citation_markers([1, 5, 2], 2) == ([1, 2], [5])


def test_split_citation_markers_zero_invalid() -> None:
    assert split_citation_markers([0, 1], 3) == ([1], [0])


def test_split_citation_markers_empty() -> None:
    assert split_citation_markers([], 3) == ([], [])
