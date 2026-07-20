import httpx
import pytest

from scholarmind.discovery import semantic_scholar as s2_module
from scholarmind.discovery.models import DiscoverySourceError
from scholarmind.discovery.semantic_scholar import (
    get_citations,
    get_references,
    resolve_paper_id,
    search_semantic_scholar,
)


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self) -> dict:
        return self._payload


_ITEM = {
    "paperId": "abc123",
    "title": "Attention Is All You Need",
    "authors": [{"name": "Ashish Vaswani"}],
    "year": 2017,
    "venue": "NeurIPS",
    "abstract": "We propose the Transformer.",
    "externalIds": {"DOI": "10.1234/attention"},
    "openAccessPdf": {"url": "https://example.org/paper.pdf"},
    "url": "https://semanticscholar.org/paper/abc123",
}


def test_search_semantic_scholar_parses_results(monkeypatch):
    monkeypatch.setattr(
        s2_module.httpx, "get", lambda *a, **kw: _FakeResponse({"data": [_ITEM]})
    )

    [result] = search_semantic_scholar("transformers")

    assert result.title == "Attention Is All You Need"
    assert result.doi == "10.1234/attention"
    assert result.pdf_url == "https://example.org/paper.pdf"
    assert result.source == "semantic_scholar"
    assert result.external_id == "abc123"


def test_search_semantic_scholar_rate_limit_raises_readable_error(monkeypatch):
    monkeypatch.setattr(
        s2_module.httpx, "get", lambda *a, **kw: _FakeResponse({}, status_code=429)
    )

    with pytest.raises(DiscoverySourceError) as excinfo:
        search_semantic_scholar("anything")

    assert excinfo.value.source == "semantic_scholar"
    assert "rate limited" in excinfo.value.message


def test_get_references_extracts_cited_paper(monkeypatch):
    monkeypatch.setattr(
        s2_module.httpx,
        "get",
        lambda *a, **kw: _FakeResponse({"data": [{"citedPaper": _ITEM}, {"citedPaper": None}]}),
    )

    results = get_references("abc123")

    assert len(results) == 1
    assert results[0].external_id == "abc123"


def test_get_citations_extracts_citing_paper(monkeypatch):
    monkeypatch.setattr(
        s2_module.httpx, "get", lambda *a, **kw: _FakeResponse({"data": [{"citingPaper": _ITEM}]})
    )

    results = get_citations("abc123")

    assert len(results) == 1


def test_resolve_paper_id_by_doi(monkeypatch):
    monkeypatch.setattr(
        s2_module.httpx, "get", lambda *a, **kw: _FakeResponse({"paperId": "abc123"})
    )

    assert resolve_paper_id(doi="10.1234/attention") == "abc123"


def test_resolve_paper_id_falls_back_to_title_search_on_404(monkeypatch):
    calls = []

    def _fake_get(url, **kwargs):
        calls.append(url)
        if "DOI:" in url:
            return _FakeResponse({}, status_code=404)
        return _FakeResponse({"data": [{"paperId": "found-by-title"}]})

    monkeypatch.setattr(s2_module.httpx, "get", _fake_get)

    result = resolve_paper_id(doi="10.9999/does-not-exist", title="Attention Is All You Need")

    assert result == "found-by-title"
    assert len(calls) == 2


def test_resolve_paper_id_returns_none_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(
        s2_module.httpx, "get", lambda *a, **kw: _FakeResponse({"data": []})
    )

    assert resolve_paper_id(title="a title matching nothing") is None
