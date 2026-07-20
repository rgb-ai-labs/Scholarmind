import httpx
import pytest

from scholarmind.discovery import openalex as openalex_module
from scholarmind.discovery.models import DiscoverySourceError
from scholarmind.discovery.openalex import search_openalex


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self) -> dict:
        return self._payload


_WORK = {
    "id": "https://openalex.org/W123",
    "title": "Attention Is All You Need",
    "publication_year": 2017,
    "doi": "https://doi.org/10.1234/attention",
    "authorships": [{"author": {"display_name": "Ashish Vaswani"}}],
    "primary_location": {
        "pdf_url": "https://example.org/paper.pdf",
        "source": {"display_name": "NeurIPS"},
    },
    "open_access": {"is_oa": True, "oa_url": "https://example.org/oa.pdf"},
    "abstract_inverted_index": {"We": [0], "propose": [1], "a": [2], "model.": [3]},
}


def test_search_openalex_parses_results(monkeypatch):
    monkeypatch.setattr(
        openalex_module.httpx, "get", lambda *a, **kw: _FakeResponse({"results": [_WORK]})
    )

    [result] = search_openalex("transformers")

    assert result.title == "Attention Is All You Need"
    assert result.doi == "10.1234/attention"
    assert result.year == 2017
    assert result.venue == "NeurIPS"
    assert result.pdf_url == "https://example.org/paper.pdf"
    assert result.abstract == "We propose a model."
    assert result.external_id == "W123"
    assert result.source == "openalex"


def test_search_openalex_falls_back_to_oa_url_when_no_primary_pdf(monkeypatch):
    work = dict(_WORK, primary_location={"source": {"display_name": "NeurIPS"}})
    monkeypatch.setattr(
        openalex_module.httpx, "get", lambda *a, **kw: _FakeResponse({"results": [work]})
    )

    [result] = search_openalex("transformers")

    assert result.pdf_url == "https://example.org/oa.pdf"


def test_search_openalex_handles_missing_abstract(monkeypatch):
    work = dict(_WORK, abstract_inverted_index=None)
    monkeypatch.setattr(
        openalex_module.httpx, "get", lambda *a, **kw: _FakeResponse({"results": [work]})
    )

    [result] = search_openalex("transformers")

    assert result.abstract is None


def test_search_openalex_rate_limit_raises_readable_error(monkeypatch):
    monkeypatch.setattr(
        openalex_module.httpx, "get", lambda *a, **kw: _FakeResponse({}, status_code=429)
    )

    with pytest.raises(DiscoverySourceError) as excinfo:
        search_openalex("anything")

    assert excinfo.value.source == "openalex"
