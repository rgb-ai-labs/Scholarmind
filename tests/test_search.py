from pathlib import Path

from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.retrieval.dense import DenseResult
from scholarmind.retrieval.search import search

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


def test_search_end_to_end_returns_relevant_results(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_search_chunks",
    )

    run_ingestion(FIXTURE_PATH, settings)

    results = search("retrieval augmented generation", settings)

    assert len(results) >= 1
    assert len(results) <= settings.retrieval_top_k

    top = results[0]
    assert isinstance(top, DenseResult)
    assert "Retrieval-Augmented Generation" in (top.title or "")
    assert top.authors == ["Ada Lovelace", "Grace Hopper"]

    for result in results:
        assert result.text.strip() != ""


def test_search_returns_empty_list_when_nothing_ingested(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_search_empty_chunks",
    )

    results = search("anything", settings)

    assert results == []
