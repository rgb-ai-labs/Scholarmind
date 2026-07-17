import uuid
from dataclasses import asdict

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from scholarmind.ingestion.chunker import Chunk


class ChunkStore:
    def __init__(self, qdrant_path: str, collection_name: str, vector_size: int) -> None:
        self._client = QdrantClient(path=qdrant_path)
        self._collection_name = collection_name

        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    @property
    def client(self) -> QdrantClient:
        return self._client

    def upsert_chunks(self, chunks: list["Chunk"], vectors: list[list[float]]) -> int:
        points = [
            PointStruct(
                id=self._point_id(chunk),
                vector=vector,
                payload=asdict(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        self._client.upsert(collection_name=self._collection_name, points=points)
        return len(points)

    @staticmethod
    def _point_id(chunk: "Chunk") -> str:
        return str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.paper_id}:{chunk.chunk_index}")
        )

    def close(self) -> None:
        self._client.close()
