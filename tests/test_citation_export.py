import httpx
import pytest

from scholarmind.citations import export as export_module
from scholarmind.citations import metadata as metadata_module
from scholarmind.citations.export import export_bibtex, paper_to_metadata
from scholarmind.citations.metadata import NormalizedMetadata, normalize_metadata
from scholarmind.retrieval.papers import PaperSummary


def _paper(**overrides) -> PaperSummary:
    defaults = dict(
        paper_id="p1",
        label="Attention Is All You Need",
        chunk_count=5,
        ingested_at=1.0,
        doi="10.1234/attention",
        is_metadata_only=False,
        authors=["Ashish Vaswani"],
        year=2017,
        venue="NeurIPS",
    )
    defaults.update(overrides)
    return PaperSummary(**defaults)


@pytest.fixture(autouse=True)
def _stub_normalize_metadata(monkeypatch):
    # paper_to_metadata calls normalize_metadata (Crossref/OpenAlex/Semantic Scholar) to enrich
    # the raw library record — stub it to a deterministic passthrough so these tests exercise
    # export/formatting logic without a live network call. Tests covering the external-resolution
    # behavior itself override this stub locally.
    def _passthrough(title, authors, year):
        return NormalizedMetadata(
            doi=None, title=title, authors=authors, year=year, venue=None, source="unresolved"
        )

    monkeypatch.setattr(export_module, "normalize_metadata", _passthrough)


def test_paper_to_metadata_maps_fields():
    metadata = paper_to_metadata(_paper())

    assert metadata.doi == "10.1234/attention"
    assert metadata.title == "Attention Is All You Need"
    assert metadata.authors == ["Ashish Vaswani"]
    assert metadata.year == 2017
    assert metadata.venue == "NeurIPS"
    assert metadata.source == "unresolved"


def test_paper_to_metadata_prefers_resolved_metadata_over_library_fields(monkeypatch):
    def _resolved(title, authors, year):
        return NormalizedMetadata(
            doi="10.9999/resolved",
            title=title,
            authors=["Real Author"],
            year=2018,
            venue="Resolved Venue",
            source="crossref",
        )

    monkeypatch.setattr(export_module, "normalize_metadata", _resolved)

    metadata = paper_to_metadata(_paper(doi=None, authors=[], year=None, venue=None))

    assert metadata.source == "crossref"
    assert metadata.doi == "10.9999/resolved"
    assert metadata.authors == ["Real Author"]
    assert metadata.year == 2018
    assert metadata.venue == "Resolved Venue"


def test_paper_to_metadata_falls_back_to_library_doi_and_venue_when_unresolved():
    # normalize_metadata is stubbed (by the autouse fixture) to return no doi/venue at all —
    # paper_to_metadata should still surface the paper's own stored doi/venue rather than
    # discarding them.
    metadata = paper_to_metadata(_paper(doi="10.1234/attention", venue="NeurIPS"))

    assert metadata.source == "unresolved"
    assert metadata.doi == "10.1234/attention"
    assert metadata.venue == "NeurIPS"


def test_paper_to_metadata_self_heals_after_an_earlier_failed_lookup(monkeypatch):
    # Regression test for the exact production bug: a paper ingested with no usable PDF-embedded
    # metadata (empty authors, no year — e.g. an ACM/LaTeX-built PDF with no /Author field) whose
    # very first live lookup attempt fails everywhere (Crossref not-yet-indexed, OpenAlex and
    # Semantic Scholar both empty). Without restarting the process or re-ingesting the paper, a
    # later render of the References panel must still be able to resolve it once a source has the
    # data — it must not be permanently stuck on the first failure.
    monkeypatch.setattr(export_module, "normalize_metadata", normalize_metadata)  # use the real one
    metadata_module._CACHE.clear()

    def _crossref_down(*args, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(metadata_module.httpx, "get", _crossref_down)
    monkeypatch.setattr(metadata_module, "search_openalex", lambda *a, **kw: [])
    monkeypatch.setattr(metadata_module, "search_semantic_scholar", lambda *a, **kw: [])

    paper = _paper(
        label="From Correctness to Collaboration: A Human-Centered Taxonomy of AI Agent Behavior",
        authors=[],
        year=None,
        doi=None,
        venue=None,
    )

    first_render = paper_to_metadata(paper)
    assert first_render.source == "unresolved"
    assert first_render.authors == []

    # A later render (e.g. Crossref finished indexing the DOI, or the network recovered) — no
    # re-ingestion, no process restart, just calling paper_to_metadata again with fresh data
    # available from a source.
    class _CrossrefHit:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "items": [
                        {
                            "DOI": "10.1145/example",
                            "title": [paper.label],
                            "author": [{"given": "Tao", "family": "Dong"}],
                            "published": {"date-parts": [[2026]]},
                            "container-title": ["Proceedings of CHI"],
                            "score": 60.0,
                        }
                    ]
                }
            }

    monkeypatch.setattr(metadata_module.httpx, "get", lambda *a, **kw: _CrossrefHit())

    second_render = paper_to_metadata(paper)

    assert second_render.source == "crossref"
    assert second_render.authors == ["Tao Dong"]
    assert second_render.year == 2026
    assert second_render.doi == "10.1145/example"


def test_export_bibtex_empty_library_returns_empty_string():
    assert export_bibtex([]) == ""


def test_export_bibtex_single_paper_produces_one_entry():
    result = export_bibtex([_paper()])

    assert result.startswith("@article{vaswani2017,")
    assert result.count("@article{") == 1


def test_export_bibtex_disambiguates_colliding_keys():
    papers = [
        _paper(paper_id="p1", label="Paper One", authors=["Ada Lovelace"], year=2020),
        _paper(paper_id="p2", label="Paper Two", authors=["Ada Lovelace"], year=2020, doi=None),
    ]

    result = export_bibtex(papers)

    assert "@article{lovelace2020," in result
    assert "@article{lovelace2020a," in result
    assert result.count("@article{") == 2
