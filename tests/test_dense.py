from pathlib import Path

from scholarmind.config import Settings
from scholarmind.ingestion.embedder import Embedder
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.retrieval.dense import DenseResult, dense_search

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


def test_dense_search_returns_relevant_results(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_dense_chunks",
    )

    run_ingestion(FIXTURE_PATH, settings)

    embedder = Embedder(settings.embedding_model)

    results = dense_search(
        "retrieval augmented generation",
        embedder,
        qdrant_path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        limit=5,
    )

    assert len(results) >= 1
    top = results[0]
    assert isinstance(top, DenseResult)
    assert top.text.strip() != ""
    assert "Retrieval-Augmented Generation" in (top.title or "")
    assert top.authors == ["Ada Lovelace", "Grace Hopper"]
    assert top.paper_id != ""
    assert isinstance(top.score, float)
    assert isinstance(top.page_start, int)
    assert isinstance(top.page_end, int)
    assert isinstance(top.chunk_index, int)


def test_dense_search_returns_empty_list_when_collection_missing(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="does_not_exist_collection",
    )
    embedder = Embedder(settings.embedding_model)

    results = dense_search(
        "anything",
        embedder,
        qdrant_path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        limit=5,
    )

    assert results == []
