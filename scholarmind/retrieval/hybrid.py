from dataclasses import dataclass

from scholarmind.retrieval.dense import DenseResult


def keyword_score(query: str, text: str) -> float:
    terms = set(query.lower().split())
    if not terms:
        return 0.0
    lowered_text = text.lower()
    matched_terms = sum(1 for term in terms if term in lowered_text)
    return matched_terms / len(terms)


@dataclass
class _RankedCandidate:
    dense_rank: int
    candidate: "DenseResult"


def hybrid_rank(query: str, candidates: list["DenseResult"]) -> list["DenseResult"]:
    if not candidates:
        return []

    ranked = [
        _RankedCandidate(dense_rank=index, candidate=candidate)
        for index, candidate in enumerate(candidates)
    ]

    keyword_sorted = sorted(
        ranked,
        key=lambda item: keyword_score(query, item.candidate.text),
        reverse=True,
    )

    keyword_rank_by_id = {
        id(item.candidate): keyword_rank
        for keyword_rank, item in enumerate(keyword_sorted)
    }

    def rrf_score(item: "_RankedCandidate") -> float:
        keyword_rank = keyword_rank_by_id[id(item.candidate)]
        return 1 / (60 + item.dense_rank) + 1 / (60 + keyword_rank)

    fused = sorted(ranked, key=rrf_score, reverse=True)

    return [item.candidate for item in fused]
