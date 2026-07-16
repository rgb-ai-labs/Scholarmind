from sentence_transformers import CrossEncoder

from scholarmind.retrieval.dense import DenseResult


class Reranker:
    def __init__(self, model_name: str) -> None:
        self._model = CrossEncoder(model_name)

    def rerank_with_scores(
        self, query: str, candidates: list["DenseResult"], top_k: int
    ) -> list[tuple["DenseResult", float]]:
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores), key=lambda item: item[1], reverse=True
        )

        return [(candidate, float(score)) for candidate, score in ranked[:top_k]]

    def rerank(
        self, query: str, candidates: list["DenseResult"], top_k: int
    ) -> list["DenseResult"]:
        return [
            candidate
            for candidate, _ in self.rerank_with_scores(query, candidates, top_k)
        ]
