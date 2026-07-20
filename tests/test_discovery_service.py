from scholarmind.discovery import service as service_module
from scholarmind.discovery.models import Candidate, DiscoverySourceError
from scholarmind.discovery.service import get_citation_graph, search_external


def _candidate(source: str, external_id: str, title: str = "A Paper") -> Candidate:
    return Candidate(
        title=title,
        authors=[],
        year=2020,
        venue=None,
        abstract=None,
        doi=None,
        url=None,
        pdf_url=None,
        source=source,
        external_id=external_id,
    )


def test_search_external_aggregates_all_sources(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "_SEARCH_SOURCES",
        (
            ("arxiv", lambda q, limit, settings: [_candidate("arxiv", "1", title="Paper A")]),
            (
                "semantic_scholar",
                lambda q, limit, settings: [_candidate("semantic_scholar", "2", title="Paper B")],
            ),
            ("openalex", lambda q, limit, settings: [_candidate("openalex", "3", title="Paper C")]),
        ),
    )
    monkeypatch.setattr(service_module, "mark_already_ingested", lambda candidates, settings: candidates)

    result = search_external("transformers")

    assert len(result.candidates) == 3
    assert result.errors == []


def test_search_external_continues_when_one_source_fails(monkeypatch):
    def _boom(q, limit, settings):
        raise DiscoverySourceError("arxiv", "connection refused")

    monkeypatch.setattr(
        service_module,
        "_SEARCH_SOURCES",
        (
            ("arxiv", _boom),
            ("semantic_scholar", lambda q, limit, settings: [_candidate("semantic_scholar", "2")]),
            ("openalex", lambda q, limit, settings: []),
        ),
    )
    monkeypatch.setattr(service_module, "mark_already_ingested", lambda candidates, settings: candidates)

    result = search_external("transformers")

    assert len(result.candidates) == 1
    assert len(result.errors) == 1
    assert "arxiv" in result.errors[0]


def test_get_citation_graph_uses_explicit_s2_id_without_resolving(monkeypatch):
    def _fail_resolve(**kwargs):
        raise AssertionError("resolve_paper_id should not be called when s2_paper_id is given")

    monkeypatch.setattr(service_module, "resolve_paper_id", _fail_resolve)
    monkeypatch.setattr(
        service_module, "get_references", lambda paper_id, limit, settings: [_candidate("s2", "r1")]
    )
    monkeypatch.setattr(
        service_module, "get_citations", lambda paper_id, limit, settings: [_candidate("s2", "c1")]
    )
    monkeypatch.setattr(service_module, "mark_already_ingested", lambda candidates, settings: candidates)

    result = get_citation_graph(s2_paper_id="abc123")

    assert result.s2_paper_id == "abc123"
    assert len(result.references) == 1
    assert len(result.citing) == 1
    assert result.errors == []


def test_get_citation_graph_reports_unresolvable_paper(monkeypatch):
    monkeypatch.setattr(service_module, "resolve_paper_id", lambda **kwargs: None)

    result = get_citation_graph(title="A totally unknown paper")

    assert result.s2_paper_id is None
    assert result.references == []
    assert result.citing == []
    assert len(result.errors) == 1


def test_get_citation_graph_collects_partial_errors(monkeypatch):
    monkeypatch.setattr(service_module, "resolve_paper_id", lambda **kwargs: "abc123")
    monkeypatch.setattr(
        service_module,
        "get_references",
        lambda paper_id, limit, settings: (_ for _ in ()).throw(
            DiscoverySourceError("semantic_scholar", "rate limited")
        ),
    )
    monkeypatch.setattr(
        service_module, "get_citations", lambda paper_id, limit, settings: [_candidate("s2", "c1")]
    )
    monkeypatch.setattr(service_module, "mark_already_ingested", lambda candidates, settings: candidates)

    result = get_citation_graph(s2_paper_id="abc123")

    assert result.references == []
    assert len(result.citing) == 1
    assert len(result.errors) == 1
