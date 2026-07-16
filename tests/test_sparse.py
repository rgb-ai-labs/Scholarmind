from pathlib import Path

from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.retrieval.dense import DenseResult
from scholarmind.retrieval.sparse import sparse_search

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


def test_sparse_search_returns_relevant_results(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_sparse_chunks",
    )

    run_ingestion(FIXTURE_PATH, settings)

    results = sparse_search(
        "sentence-transformers local embeddings",
        qdrant_path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        limit=5,
    )

    assert len(results) >= 1
    top = results[0]
    assert isinstance(top, DenseResult)
    assert "sentence-transformers" in top.text.lower()
    assert "local" in top.text.lower()
    assert "Retrieval-Augmented Generation" in (top.title or "")
    assert top.authors == ["Ada Lovelace", "Grace Hopper"]


def test_sparse_search_returns_empty_list_when_collection_missing(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="does_not_exist_collection",
    )

    results = sparse_search(
        "anything",
        qdrant_path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        limit=5,
    )

    assert results == []
