from scholarmind.retrieval.dense import DenseResult


def hybrid_rank(
    dense_results: list["DenseResult"],
    sparse_results: list["DenseResult"],
) -> list["DenseResult"]:
    dense_rank_by_key = {
        (result.paper_id, result.chunk_index): rank
        for rank, result in enumerate(dense_results)
    }
    sparse_rank_by_key = {
        (result.paper_id, result.chunk_index): rank
        for rank, result in enumerate(sparse_results)
    }

    candidates_by_key: dict[tuple[str, int], DenseResult] = {}
    for result in sparse_results:
        candidates_by_key[(result.paper_id, result.chunk_index)] = result
    for result in dense_results:
        candidates_by_key[(result.paper_id, result.chunk_index)] = result

    def rrf_score(key: tuple[str, int]) -> float:
        dense_rank = dense_rank_by_key.get(key)
        sparse_rank = sparse_rank_by_key.get(key)
        score = 0.0
        if dense_rank is not None:
            score += 1 / (60 + dense_rank)
        if sparse_rank is not None:
            score += 1 / (60 + sparse_rank)
        return score

    sorted_keys = sorted(candidates_by_key.keys(), key=rrf_score, reverse=True)

    return [candidates_by_key[key] for key in sorted_keys]
