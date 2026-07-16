import httpx
import pytest

from scholarmind.citations import metadata as metadata_module
from scholarmind.citations.metadata import normalize_metadata


def test_normalize_metadata_real_paper_matches_crossref():
    result = normalize_metadata(
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        ["Patrick Lewis"],
        2020,
    )

    assert result.source == "crossref"
    assert result.doi is not None
    assert "10." in result.doi
    assert "/" in result.doi
    assert result.title is not None


def test_normalize_metadata_unfindable_title_falls_back_unresolved():
    gibberish_title = "Zqxvblorf Wpnjk Trflmzq Yhbdqz Kwplzxv Nvbqrzt Fjxplmw Study 9384756"
    authors = ["A. Nobody"]
    year = 1901

    result = normalize_metadata(gibberish_title, authors, year)

    assert result.source == "unresolved"
    assert result.doi is None
    assert result.venue is None
    assert result.title == gibberish_title
    assert result.authors == authors
    assert result.year == year


def test_normalize_metadata_none_title_is_immediately_unresolved():
    result = normalize_metadata(None, ["Someone"], 2020)

    assert result.source == "unresolved"
    assert result.doi is None
    assert result.venue is None
    assert result.title is None
    assert result.authors == ["Someone"]
    assert result.year == 2020


def test_normalize_metadata_blank_title_is_immediately_unresolved():
    result = normalize_metadata("   ", ["Someone"], 2020)

    assert result.source == "unresolved"
    assert result.doi is None
    assert result.title == "   "


def test_normalize_metadata_caches_and_avoids_second_network_call(monkeypatch):
    metadata_module._CACHE.clear()

    title = "Attention Is All You Need"
    authors = ["Ashish Vaswani"]
    year = 2017

    first_result = normalize_metadata(title, authors, year)

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("httpx.get should not be called on cached lookup")

    monkeypatch.setattr(metadata_module.httpx, "get", _raise_if_called)

    second_result = normalize_metadata(title, authors, year)

    assert second_result == first_result


def test_normalize_metadata_caches_unresolved_results(monkeypatch):
    metadata_module._CACHE.clear()

    gibberish_title = "Blorptastic Fnargle Wubzik Quixolotl Zephyrform 1122334455"
    authors = ["Nobody Notable"]
    year = 1899

    first_result = normalize_metadata(gibberish_title, authors, year)
    assert first_result.source == "unresolved"

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("httpx.get should not be called on cached lookup")

    monkeypatch.setattr(metadata_module.httpx, "get", _raise_if_called)

    second_result = normalize_metadata(gibberish_title, authors, year)

    assert second_result == first_result
    assert second_result.source == "unresolved"


def test_normalize_metadata_network_failure_falls_back_unresolved(monkeypatch):
    metadata_module._CACHE.clear()

    def _raise(*args, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(metadata_module.httpx, "get", _raise)

    title = "Some Title That Would Normally Be Looked Up"
    authors = ["Jane Doe"]
    year = 2022

    result = normalize_metadata(title, authors, year)

    assert result.source == "unresolved"
    assert result.doi is None
    assert result.venue is None
    assert result.title == title
    assert result.authors == authors
    assert result.year == year
