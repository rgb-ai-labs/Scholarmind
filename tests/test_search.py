from pathlib import Path

from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.retrieval.dense import DenseResult
from scholarmind.retrieval.papers import list_papers
from scholarmind.retrieval.search import search

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"
TIDAL_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper_2.pdf"


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


def test_search_default_threshold_still_returns_relevant_results(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_search_default_threshold_chunks",
    )

    run_ingestion(FIXTURE_PATH, settings)

    results = search("retrieval augmented generation", settings)

    assert results != []


def test_search_high_threshold_filters_out_all_results(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_search_high_threshold_chunks",
        retrieval_min_rerank_score=100.0,
    )

    run_ingestion(FIXTURE_PATH, settings)

    results = search("retrieval augmented generation", settings)

    assert results == []


def test_search_irrelevant_query_returns_empty_with_default_threshold(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_search_irrelevant_query_chunks",
    )

    run_ingestion(FIXTURE_PATH, settings)

    results = search(
        "what is the best way to grill a steak for a summer barbecue", settings
    )

    assert results == []


def test_search_with_paper_id_scopes_results_to_that_paper_even_off_topic(
    tmp_path: Path,
):
    # Regression test: a query naming a filename/topic outside the scoped paper (the
    # kind of thing a user would type into a "which paper" free-text box) must still
    # only surface chunks from the paper_id-filtered paper, not whatever the library-wide
    # semantic search would otherwise rank highest.
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_search_scoped_chunks",
    )

    run_ingestion(FIXTURE_PATH, settings)
    run_ingestion(TIDAL_FIXTURE_PATH, settings)

    rag_paper_id = next(
        p.paper_id
        for p in list_papers(settings)
        if "Retrieval-Augmented Generation" in p.label
    )

    results = search("tidal energy coastal turbines", settings, paper_id=rag_paper_id)

    assert all(r.paper_id == rag_paper_id for r in results)
