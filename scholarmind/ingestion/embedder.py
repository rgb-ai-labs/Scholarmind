from sentence_transformers import SentenceTransformer

from scholarmind.ingestion.chunker import Chunk


class Embedder:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    def embed_chunks(self, chunks: list["Chunk"], batch_size: int = 32) -> list[list[float]]:
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_text(self, text: str) -> list[float]:
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()
