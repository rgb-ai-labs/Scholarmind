import re
from dataclasses import dataclass

import httpx

from scholarmind.discovery.models import Candidate
from scholarmind.discovery.openalex import search_openalex
from scholarmind.discovery.semantic_scholar import search_semantic_scholar

_CACHE: dict[tuple, "NormalizedMetadata"] = {}
_CROSSREF_URL = "https://api.crossref.org/works"
_USER_AGENT = "ScholarMind/0.1 (https://github.com/scholarmind; mailto:contact@example.com)"
_MIN_MATCH_SCORE = 15.0


@dataclass
class NormalizedMetadata:
    doi: str | None
    title: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    source: str


def _unresolved(title: str | None, authors: list[str], year: int | None) -> NormalizedMetadata:
    return NormalizedMetadata(
        doi=None,
        title=title,
        authors=authors,
        year=year,
        venue=None,
        source="unresolved",
    )


def _collapse_title(title: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower()) if title else ""


def _titles_plausibly_match(requested: str | None, candidate: str | None) -> bool:
    # Fallback sources return a top search hit with no confidence score (unlike Crossref's
    # `score`), so this substitutes a title sanity check to reject an obviously wrong match
    # (e.g. a generic-titled paper returning an unrelated top hit).
    requested_key = _collapse_title(requested)
    candidate_key = _collapse_title(candidate)
    if not requested_key or not candidate_key:
        return False
    return (
        requested_key == candidate_key
        or requested_key in candidate_key
        or candidate_key in requested_key
    )


def _from_candidate(candidate: "Candidate", source: str) -> NormalizedMetadata:
    return NormalizedMetadata(
        doi=candidate.doi,
        title=candidate.title,
        authors=candidate.authors,
        year=candidate.year,
        venue=candidate.venue,
        source=source,
    )


def _try_fallback_source(
    title: str, search_fn, source_name: str
) -> NormalizedMetadata | None:
    try:
        results = search_fn(title, 1)
    except Exception:
        return None
    if not results:
        return None
    top = results[0]
    if _titles_plausibly_match(title, top.title):
        return _from_candidate(top, source_name)
    return None


def _resolve_via_fallback_sources(
    title: str, authors: list[str], year: int | None
) -> NormalizedMetadata | None:
    # Calls search_openalex/search_semantic_scholar by their bare (module-global) names rather
    # than through a precomputed list of function references, so tests can monkeypatch
    # scholarmind.citations.metadata.search_openalex / .search_semantic_scholar directly.
    return _try_fallback_source(title, search_openalex, "openalex") or _try_fallback_source(
        title, search_semantic_scholar, "semantic_scholar"
    )


def _extract_year(item: dict, fallback: int | None) -> int | None:
    for key in ("published-print", "published-online", "published"):
        date_field = item.get(key)
        if not date_field:
            continue
        try:
            return int(date_field["date-parts"][0][0])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return fallback


def _extract_authors(item: dict, fallback: list[str]) -> list[str]:
    names: list[str] = []
    for author in item.get("author", []):
        family = author.get("family")
        if not family:
            continue
        given = author.get("given", "")
        names.append(f"{given} {family}".strip())
    return names if names else fallback


def normalize_metadata(
    title: str | None, authors: list[str], year: int | None
) -> NormalizedMetadata:
    # Only successful resolutions are cached (see below) — a failed attempt is never pinned
    # in _CACHE, so it can't outlive the reason it failed (e.g. a transient network hiccup, or
    # a paper whose DOI wasn't registered/indexed yet at the time of an earlier attempt). This
    # module-level cache lives for the whole process (e.g. an entire long-running Streamlit
    # session) — permanently caching "unresolved" would mean a paper that failed once could
    # never be re-looked-up again for the rest of that process's life, even after a code fix
    # or once the source catches up, without a full process restart.
    key = (title, tuple(authors), year)
    if key in _CACHE:
        return _CACHE[key]

    if title is None or not title.strip():
        # Nothing about this input will ever change on a retry — safe (and free) to cache.
        result = _unresolved(title, authors, year)
        _CACHE[key] = result
        return result

    try:
        response = httpx.get(
            _CROSSREF_URL,
            params={"query.bibliographic": title, "rows": 1},
            timeout=10.0,
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
        items = response.json()["message"]["items"]
        if not items:
            raise ValueError("no crossref matches")

        item = items[0]
        if float(item.get("score") or 0.0) < _MIN_MATCH_SCORE:
            raise ValueError("crossref match confidence too low")

        crossref_title = title
        if item.get("title"):
            crossref_title = item["title"][0]

        # Crossref's `score` reflects search relevance, not correctness — a high-scoring top
        # hit can still be a different paper with an overlapping-but-different title (e.g.
        # "Attention is All you Need" vs. "Is Attention All You Need?"). Reject those the same
        # way the fallback sources already do, rather than returning a wrong DOI/author/venue
        # that looks authoritative.
        if not _titles_plausibly_match(title, crossref_title):
            raise ValueError("crossref top hit title does not match requested title")

        crossref_venue = None
        if item.get("container-title"):
            crossref_venue = item["container-title"][0]

        result = NormalizedMetadata(
            doi=item.get("DOI"),
            title=crossref_title,
            authors=_extract_authors(item, authors),
            year=_extract_year(item, year),
            venue=crossref_venue,
            source="crossref",
        )
    except Exception:
        fallback_result = _resolve_via_fallback_sources(title, authors, year)
        if fallback_result is None:
            # Every source failed this attempt — return unresolved WITHOUT caching it, so the
            # next call (next page render) tries again from scratch instead of repeating this
            # exact failure forever.
            return _unresolved(title, authors, year)
        result = fallback_result

    _CACHE[key] = result
    return result
