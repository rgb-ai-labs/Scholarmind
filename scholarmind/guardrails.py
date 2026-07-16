from scholarmind.config import Settings


def refusal_threshold(settings: "Settings") -> float:
    return settings.retrieval_min_rerank_score


def passes_confidence(rerank_score: float, settings: "Settings") -> bool:
    return rerank_score >= settings.retrieval_min_rerank_score


def split_citation_markers(markers: list[int], num_sources: int) -> tuple[list[int], list[int]]:
    valid: list[int] = []
    invalid: list[int] = []
    for marker in markers:
        if 1 <= marker <= num_sources:
            valid.append(marker)
        else:
            invalid.append(marker)
    return valid, invalid
