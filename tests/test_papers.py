from pathlib import Path

from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.retrieval.dense import DenseResult
from scholarmind.retrieval.papers import PaperSummary, get_paper_chunks, list_papers

RAG_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"
TIDAL_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper_2.pdf"

RAG_TITLE = "A Study of Retrieval-Augmented Generation for Scholarly Question Answering"
TIDAL_TITLE = "Tidal Energy Harvesting Efficiency in Coastal Turbine Arrays"


def test_list_papers_returns_one_entry_per_paper_with_title_label(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_papers_list_chunks",
    )
    run_ingestion(RAG_FIXTURE_PATH, settings)
    run_ingestion(TIDAL_FIXTURE_PATH, settings)

    papers = list_papers(settings)

    assert len(papers) == 2
    labels = {p.label for p in papers}
    assert labels == {RAG_TITLE, TIDAL_TITLE}
    for paper in papers:
        assert isinstance(paper, PaperSummary)
        assert paper.chunk_count > 0
        assert paper.paper_id != ""


def test_list_papers_orders_most_recently_ingested_first(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_papers_order_chunks",
    )
    run_ingestion(RAG_FIXTURE_PATH, settings)
    run_ingestion(TIDAL_FIXTURE_PATH, settings)

    papers = list_papers(settings)

    assert papers[0].label == TIDAL_TITLE


def test_list_papers_returns_empty_list_when_nothing_ingested(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_papers_empty_chunks",
    )

    assert list_papers(settings) == []


def test_get_paper_chunks_returns_only_that_paper_in_reading_order(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_paper_chunks_chunks",
    )
    run_ingestion(RAG_FIXTURE_PATH, settings)
    run_ingestion(TIDAL_FIXTURE_PATH, settings)

    papers = {p.label: p.paper_id for p in list_papers(settings)}
    rag_chunks = get_paper_chunks(papers[RAG_TITLE], settings)

    assert len(rag_chunks) > 0
    for chunk in rag_chunks:
        assert isinstance(chunk, DenseResult)
        assert chunk.title == RAG_TITLE
    assert [c.chunk_index for c in rag_chunks] == sorted(c.chunk_index for c in rag_chunks)


def test_get_paper_chunks_returns_empty_list_for_unknown_paper_id(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_paper_chunks_missing_chunks",
    )
    run_ingestion(RAG_FIXTURE_PATH, settings)

    assert get_paper_chunks("does-not-exist", settings) == []
