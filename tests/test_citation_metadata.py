import httpx
import pytest

from scholarmind.citations import metadata as metadata_module
from scholarmind.citations.metadata import normalize_metadata
from scholarmind.discovery.models import Candidate


def _stub_no_fallback_matches(monkeypatch) -> None:
    # These tests exercise the Crossref path only; without this, a Crossref miss would fall
    # through to the (real, network-calling) OpenAlex/Semantic Scholar clients.
    monkeypatch.setattr(metadata_module, "search_openalex", lambda *args, **kwargs: [])
    monkeypatch.setattr(metadata_module, "search_semantic_scholar", lambda *args, **kwargs: [])


def _candidate(**overrides) -> Candidate:
    defaults = dict(
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=["Patrick Lewis"],
        year=2020,
        venue="Advances in Neural Information Processing Systems",
        abstract=None,
        doi="10.5555/fallback",
        url=None,
        pdf_url=None,
        source="openalex",
        external_id="w123",
    )
    defaults.update(overrides)
    return Candidate(**defaults)


def _crossref_reachable() -> bool:
    try:
        httpx.get("https://api.crossref.org/works", params={"rows": 0}, timeout=5.0)
        return True
    except Exception:
        return False


_CROSSREF_UP = _crossref_reachable()


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _crossref_item(score: float) -> dict:
    return {
        "DOI": "10.1234/example.doi",
        "title": ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"],
        "author": [{"given": "Patrick", "family": "Lewis"}],
        "published": {"date-parts": [[2020]]},
        "container-title": ["Advances in Neural Information Processing Systems"],
        "score": score,
    }


@pytest.mark.skipif(not _CROSSREF_UP, reason="Crossref API not reachable")
def test_normalize_metadata_real_paper_matches_crossref():
    metadata_module._CACHE.clear()
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


def test_normalize_metadata_high_score_match_is_used(monkeypatch):
    metadata_module._CACHE.clear()

    def _fake_get(*args, **kwargs):
        return _FakeResponse({"message": {"items": [_crossref_item(40.0)]}})

    monkeypatch.setattr(metadata_module.httpx, "get", _fake_get)

    result = normalize_metadata(
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        ["P. Lewis"],
        2020,
    )

    assert result.source == "crossref"
    assert result.doi == "10.1234/example.doi"
    assert result.year == 2020
    assert result.authors == ["Patrick Lewis"]


def test_normalize_metadata_low_score_match_is_rejected(monkeypatch):
    metadata_module._CACHE.clear()

    def _fake_get(*args, **kwargs):
        return _FakeResponse({"message": {"items": [_crossref_item(6.0)]}})

    monkeypatch.setattr(metadata_module.httpx, "get", _fake_get)
    _stub_no_fallback_matches(monkeypatch)

    gibberish_title = "Zqxvblorf Wpnjk Trflmzq Yhbdqz Kwplzxv Study 9384756"
    authors = ["A. Nobody"]
    year = 1901

    result = normalize_metadata(gibberish_title, authors, year)

    assert result.source == "unresolved"
    assert result.doi is None
    assert result.venue is None
    assert result.title == gibberish_title
    assert result.authors == authors
    assert result.year == year


def test_normalize_metadata_rejects_crossref_hit_with_overlapping_but_different_title(
    monkeypatch,
):
    # Regression test: "Attention is All you Need" previously resolved to an unrelated 2025
    # paper "Is Attention All You Need?" by P. Mineault — a different paper with a
    # confusingly similar (word-reordered) title, not a legitimate match. A high Crossref
    # `score` alone doesn't guarantee the top hit is the requested paper.
    metadata_module._CACHE.clear()

    def _fake_get(*args, **kwargs):
        return _FakeResponse(
            {
                "message": {
                    "items": [
                        {
                            "DOI": "10.9999/mineault.wrong.paper",
                            "title": ["Is Attention All You Need?"],
                            "author": [{"given": "P.", "family": "Mineault"}],
                            "published": {"date-parts": [[2025]]},
                            "container-title": ["arXiv"],
                            "score": 90.0,
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(metadata_module.httpx, "get", _fake_get)
    _stub_no_fallback_matches(monkeypatch)

    title = "Attention is All you Need"
    authors = ["Ashish Vaswani"]
    year = 2017

    result = normalize_metadata(title, authors, year)

    assert result.source == "unresolved"
    assert result.doi is None
    assert result.title == title
    assert result.authors == authors
    assert result.year == year


def test_normalize_metadata_rejects_crossref_hit_for_fictional_test_fixture_paper(monkeypatch):
    # Regression test: a fictional test-fixture paper "A Study of Retrieval-Augmented
    # Generation for Scholarly Question Answering" (Ada Lovelace / Grace Hopper) previously
    # resolved to an unrelated real paper "Question Answering in the Construction Domain
    # Using Retrieval-Augmented Generation" by D. Busch — same topic keywords, different paper.
    metadata_module._CACHE.clear()

    def _fake_get(*args, **kwargs):
        return _FakeResponse(
            {
                "message": {
                    "items": [
                        {
                            "DOI": "10.9999/busch.wrong.paper",
                            "title": [
                                "Question Answering in the Construction Domain Using "
                                "Retrieval-Augmented Generation"
                            ],
                            "author": [{"given": "D.", "family": "Busch"}],
                            "published": {"date-parts": [[2024]]},
                            "container-title": ["Construction Informatics"],
                            "score": 85.0,
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(metadata_module.httpx, "get", _fake_get)
    _stub_no_fallback_matches(monkeypatch)

    title = "A Study of Retrieval-Augmented Generation for Scholarly Question Answering"
    authors = ["Ada Lovelace", "Grace Hopper"]
    year = 2023

    result = normalize_metadata(title, authors, year)

    assert result.source == "unresolved"
    assert result.doi is None
    assert result.title == title
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

    call_count = {"n": 0}

    def _fake_get(*args, **kwargs):
        call_count["n"] += 1
        return _FakeResponse({"message": {"items": [_crossref_item(40.0)]}})

    monkeypatch.setattr(metadata_module.httpx, "get", _fake_get)

    title = "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    authors = ["Ashish Vaswani"]
    year = 2017

    first_result = normalize_metadata(title, authors, year)
    second_result = normalize_metadata(title, authors, year)

    assert second_result == first_result
    assert call_count["n"] == 1


def test_normalize_metadata_does_not_cache_unresolved_results_so_it_can_self_heal(monkeypatch):
    # Regression test for a real production bug: a long-lived process (e.g. a Streamlit session
    # left running for a while) that failed to resolve a paper once — say, because Crossref
    # hadn't indexed its DOI yet, or a transient network error — must NOT keep returning
    # "unresolved" forever. Every call that didn't resolve retries from scratch on the next call,
    # so a paper that becomes resolvable later (or simply had a flaky first attempt) self-heals
    # without restarting the process.
    metadata_module._CACHE.clear()

    def _raise(*args, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(metadata_module.httpx, "get", _raise)
    _stub_no_fallback_matches(monkeypatch)

    title = "A Paper Not Yet Indexed Anywhere On The First Attempt"
    authors = ["Nobody Notable"]
    year = 1899

    first_result = normalize_metadata(title, authors, year)
    assert first_result.source == "unresolved"

    # Simulate the paper becoming resolvable on a later attempt (e.g. Crossref finished
    # indexing it, or the network hiccup passed) — this must actually be tried again, not
    # served from a stale cached failure.
    monkeypatch.setattr(
        metadata_module,
        "search_openalex",
        lambda query, limit, *a, **kw: [_candidate(title=title, authors=["Real Author"])],
    )

    second_result = normalize_metadata(title, authors, year)

    assert second_result.source == "openalex"
    assert second_result.authors == ["Real Author"]


def test_normalize_metadata_network_failure_falls_back_unresolved(monkeypatch):
    metadata_module._CACHE.clear()

    def _raise(*args, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(metadata_module.httpx, "get", _raise)
    _stub_no_fallback_matches(monkeypatch)

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


def test_normalize_metadata_falls_back_to_openalex_when_crossref_fails(monkeypatch):
    metadata_module._CACHE.clear()

    def _raise(*args, **kwargs):
        raise httpx.ConnectError("boom")

    s2_calls = {"n": 0}

    def _s2_should_not_be_called(*args, **kwargs):
        s2_calls["n"] += 1
        return []

    monkeypatch.setattr(metadata_module.httpx, "get", _raise)
    monkeypatch.setattr(
        metadata_module,
        "search_openalex",
        lambda title, limit, *a, **kw: [_candidate(title=title)],
    )
    monkeypatch.setattr(metadata_module, "search_semantic_scholar", _s2_should_not_be_called)

    title = "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    result = normalize_metadata(title, [], None)

    assert result.source == "openalex"
    assert result.doi == "10.5555/fallback"
    assert result.authors == ["Patrick Lewis"]
    assert result.year == 2020
    assert result.venue == "Advances in Neural Information Processing Systems"
    assert s2_calls["n"] == 0


def test_normalize_metadata_falls_back_to_semantic_scholar_when_openalex_also_fails(monkeypatch):
    metadata_module._CACHE.clear()

    def _raise(*args, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(metadata_module.httpx, "get", _raise)
    monkeypatch.setattr(metadata_module, "search_openalex", lambda *a, **kw: [])
    monkeypatch.setattr(
        metadata_module,
        "search_semantic_scholar",
        lambda title, limit, *a, **kw: [_candidate(title=title, source="semantic_scholar")],
    )

    title = "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    result = normalize_metadata(title, [], None)

    assert result.source == "semantic_scholar"
    assert result.doi == "10.5555/fallback"
    assert result.authors == ["Patrick Lewis"]


def test_normalize_metadata_rejects_fallback_result_with_mismatched_title(monkeypatch):
    metadata_module._CACHE.clear()

    def _raise(*args, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(metadata_module.httpx, "get", _raise)
    monkeypatch.setattr(
        metadata_module,
        "search_openalex",
        lambda *a, **kw: [_candidate(title="A Completely Unrelated Paper About Gardening")],
    )
    monkeypatch.setattr(metadata_module, "search_semantic_scholar", lambda *a, **kw: [])

    title = "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    authors = ["Patrick Lewis"]
    year = 2020

    result = normalize_metadata(title, authors, year)

    assert result.source == "unresolved"
    assert result.title == title
    assert result.authors == authors
    assert result.year == year
