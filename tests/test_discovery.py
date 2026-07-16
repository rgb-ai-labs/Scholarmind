from pathlib import Path

from scholarmind.agents.base import AgentResult
from scholarmind.agents.discovery import discover
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion

RAG_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"
TIDAL_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper_2.pdf"

RAG_TITLE = "A Study of Retrieval-Augmented Generation for Scholarly Question Answering"
TIDAL_TITLE = "Tidal Energy Harvesting Efficiency in Coastal Turbine Arrays"


def test_discover_surfaces_relevant_papers_and_discriminates(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_discovery_chunks",
    )
    run_ingestion(RAG_FIXTURE_PATH, settings)
    run_ingestion(TIDAL_FIXTURE_PATH, settings)

    rag_result = discover("retrieval augmented generation", settings=settings)

    assert isinstance(rag_result, AgentResult)
    assert rag_result.sources_found > 0
    assert RAG_TITLE in rag_result.text

    tidal_result = discover("tidal energy coastal turbines", settings=settings)

    assert tidal_result.sources_found > 0
    assert TIDAL_TITLE in tidal_result.text


def test_discover_dedupes_distinct_papers_in_text(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_discovery_dedup_chunks",
    )
    run_ingestion(RAG_FIXTURE_PATH, settings)

    result = discover("retrieval augmented generation", settings=settings)

    assert result.text.count(RAG_TITLE) == 1


def test_discover_returns_empty_result_when_nothing_ingested(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant_empty"),
        qdrant_collection="test_discovery_empty_chunks",
    )

    result = discover("anything", settings=settings)

    assert result.text == ""
    assert result.sources == []
    assert result.sources_found == 0
