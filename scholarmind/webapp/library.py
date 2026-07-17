from pathlib import Path

from qdrant_client import QdrantClient

from scholarmind.config import Settings


def get_library_stats(settings: "Settings") -> tuple[int, int]:
    # Embedded Qdrant is single-process: this opens and closes its own short-lived
    # client, the same pattern used throughout scholarmind/retrieval/, so it never
    # holds a lock across a Streamlit rerun.
    client = QdrantClient(path=settings.qdrant_path)
    try:
        if not client.collection_exists(settings.qdrant_collection):
            return 0, 0

        chunk_count = client.count(settings.qdrant_collection, exact=True).count

        paper_ids: set[str] = set()
        offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=settings.qdrant_collection,
                with_payload=["paper_id"],
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for point in batch:
                paper_ids.add(point.payload["paper_id"])
            if next_offset is None:
                break
            offset = next_offset

        return len(paper_ids), chunk_count
    finally:
        client.close()


def papers_dir_for(settings: "Settings") -> Path:
    return Path(settings.qdrant_path).resolve().parent / "uploads"
