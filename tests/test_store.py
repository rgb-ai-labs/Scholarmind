from scholarmind.ingestion.chunker import Chunk
from scholarmind.ingestion.store import ChunkStore


def _chunk(paper_id: str, chunk_index: int, text: str) -> Chunk:
    return Chunk(
        paper_id=paper_id,
        title="Sample Title",
        authors=["Ada Lovelace", "Grace Hopper"],
        year=2024,
        venue="ICML",
        section="Abstract",
        page_start=1,
        page_end=1,
        chunk_index=chunk_index,
        text=text,
    )


def test_upsert_chunks_stores_expected_number_of_points_with_payload():
    store = ChunkStore(qdrant_path=":memory:", collection_name="test_chunks", vector_size=4)
    chunks = [
        _chunk("paper-1", 0, "First chunk text."),
        _chunk("paper-1", 1, "Second chunk text."),
    ]
    vectors = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ]

    count = store.upsert_chunks(chunks, vectors)

    assert count == 2

    result = store.client.count(collection_name="test_chunks", exact=True)
    assert result.count == 2

    points = store.client.scroll(
        collection_name="test_chunks",
        limit=10,
        with_payload=True,
        with_vectors=False,
    )[0]
    payloads = {point.payload["chunk_index"]: point.payload for point in points}

    assert payloads[0]["paper_id"] == "paper-1"
    assert payloads[0]["title"] == "Sample Title"
    assert payloads[0]["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert payloads[0]["year"] == 2024
    assert payloads[0]["venue"] == "ICML"
    assert payloads[0]["section"] == "Abstract"
    assert payloads[0]["page_start"] == 1
    assert payloads[0]["page_end"] == 1
    assert payloads[0]["text"] == "First chunk text."

    store.close()


def test_reupserting_same_chunks_is_idempotent():
    store = ChunkStore(qdrant_path=":memory:", collection_name="test_chunks", vector_size=4)
    chunks = [
        _chunk("paper-1", 0, "First chunk text."),
        _chunk("paper-1", 1, "Second chunk text."),
    ]
    vectors_v1 = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ]
    vectors_v2 = [
        [0.9, 0.8, 0.7, 0.6],
        [0.5, 0.4, 0.3, 0.2],
    ]

    store.upsert_chunks(chunks, vectors_v1)
    count = store.upsert_chunks(chunks, vectors_v2)

    assert count == 2

    result = store.client.count(collection_name="test_chunks", exact=True)
    assert result.count == 2

    store.close()
