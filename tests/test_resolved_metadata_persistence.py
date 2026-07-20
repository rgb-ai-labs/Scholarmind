from pathlib import Path

from scholarmind.citations import export as export_module
from scholarmind.citations.export import paper_to_metadata
from scholarmind.citations.metadata import NormalizedMetadata
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.retrieval.papers import list_papers, persist_resolved_metadata

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


def _settings(tmp_path: Path, name: str) -> Settings:
    return Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection=name)


def test_persist_resolved_metadata_is_visible_via_list_papers(tmp_path: Path):
    settings = _settings(tmp_path, "test_persist_visible_chunks")
    run_ingestion(FIXTURE_PATH, settings)
    [paper] = list_papers(settings)
    assert paper.resolved_source is None  # nothing resolved yet

    persist_resolved_metadata(
        paper.paper_id,
        doi="10.1234/real",
        authors=["Real Author One", "Real Author Two"],
        year=2019,
        venue="Real Venue",
        source="crossref",
        settings=settings,
    )

    [refreshed] = list_papers(settings)
    assert refreshed.resolved_source == "crossref"
    assert refreshed.resolved_doi == "10.1234/real"
    assert refreshed.resolved_authors == ["Real Author One", "Real Author Two"]
    assert refreshed.resolved_year == 2019
    assert refreshed.resolved_venue == "Real Venue"
    # The raw library-derived fields (from the PDF itself) are untouched by persistence.
    assert refreshed.chunk_count == paper.chunk_count


def test_persist_resolved_metadata_updates_every_chunk_of_the_paper(tmp_path: Path):
    from scholarmind.retrieval.papers import get_paper_chunks

    settings = _settings(tmp_path, "test_persist_allchunks_chunks")
    run_ingestion(FIXTURE_PATH, settings)
    [paper] = list_papers(settings)

    persist_resolved_metadata(
        paper.paper_id,
        doi="10.1234/real",
        authors=["Real Author"],
        year=2019,
        venue="Real Venue",
        source="crossref",
        settings=settings,
    )

    chunks = get_paper_chunks(paper.paper_id, settings)
    assert len(chunks) == paper.chunk_count
    assert all(c.resolved_source == "crossref" for c in chunks)
    assert all(c.resolved_doi == "10.1234/real" for c in chunks)


def test_persist_resolved_metadata_with_unknown_paper_id_is_a_noop(tmp_path: Path):
    settings = _settings(tmp_path, "test_persist_unknown_chunks")
    run_ingestion(FIXTURE_PATH, settings)

    # Must not raise, and must not affect the real paper that IS in the store.
    persist_resolved_metadata(
        "not-a-real-paper-id",
        doi="10.1234/x",
        authors=[],
        year=None,
        venue=None,
        source="crossref",
        settings=settings,
    )

    [paper] = list_papers(settings)
    assert paper.resolved_source is None


def test_persist_resolved_metadata_against_empty_store_is_a_noop(tmp_path: Path):
    settings = _settings(tmp_path, "test_persist_empty_chunks")

    # No collection exists yet at all — must not raise.
    persist_resolved_metadata(
        "any-id", doi=None, authors=[], year=None, venue=None, source="crossref", settings=settings
    )


def test_paper_to_metadata_uses_persisted_resolution_without_a_second_network_call(
    tmp_path: Path, monkeypatch
):
    settings = _settings(tmp_path, "test_persist_reuse_chunks")
    run_ingestion(FIXTURE_PATH, settings)
    [paper] = list_papers(settings)

    call_count = {"n": 0}

    def _resolved(title, authors, year):
        call_count["n"] += 1
        return NormalizedMetadata(
            doi="10.9999/resolved",
            title=title,
            authors=["Real Author"],
            year=2019,
            venue="Resolved Venue",
            source="crossref",
        )

    monkeypatch.setattr(export_module, "normalize_metadata", _resolved)

    first = paper_to_metadata(paper, settings)
    assert first.source == "crossref"
    assert call_count["n"] == 1

    # Re-fetch the paper the same way a fresh page render would — it should now carry the
    # persisted resolution, so a second call must not hit normalize_metadata again.
    [refreshed_paper] = list_papers(settings)

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("normalize_metadata should not be called once resolution is persisted")

    monkeypatch.setattr(export_module, "normalize_metadata", _should_not_be_called)

    second = paper_to_metadata(refreshed_paper, settings)

    assert second.source == "crossref"
    assert second.doi == "10.9999/resolved"
    assert second.authors == ["Real Author"]
    assert second.year == 2019
    assert second.venue == "Resolved Venue"
    assert call_count["n"] == 1  # unchanged — no second network call


def test_paper_to_metadata_does_not_persist_an_unresolved_result(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "test_persist_no_unresolved_chunks")
    run_ingestion(FIXTURE_PATH, settings)
    [paper] = list_papers(settings)

    def _unresolved(title, authors, year):
        return NormalizedMetadata(
            doi=None, title=title, authors=authors, year=year, venue=None, source="unresolved"
        )

    monkeypatch.setattr(export_module, "normalize_metadata", _unresolved)

    paper_to_metadata(paper, settings)

    [refreshed] = list_papers(settings)
    assert refreshed.resolved_source is None  # a failed lookup is never persisted
