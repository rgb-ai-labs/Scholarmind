from typing import TYPE_CHECKING

import httpx

from scholarmind.config import get_settings
from scholarmind.discovery.models import Candidate, DiscoverySourceError

if TYPE_CHECKING:
    from scholarmind.config import Settings

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_USER_AGENT = "ScholarMind/0.1 (https://github.com/scholarmind; mailto:contact@example.com)"
_FIELDS = "title,authors,year,venue,abstract,externalIds,openAccessPdf,url"


def _headers(settings: "Settings | None") -> dict[str, str]:
    headers = {"User-Agent": _USER_AGENT}
    if settings and settings.s2_api_key:
        headers["x-api-key"] = settings.s2_api_key
    return headers


def _get(url: str, params: dict, settings: "Settings | None", timeout: float) -> dict:
    try:
        response = httpx.get(url, params=params, headers=_headers(settings), timeout=timeout)
        if response.status_code == 429:
            raise DiscoverySourceError(
                "semantic_scholar", "rate limited — try again shortly, or set S2_API_KEY"
            )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DiscoverySourceError("semantic_scholar", str(exc)) from exc
    return response.json()


def _to_candidate(item: dict) -> Candidate:
    external_ids = item.get("externalIds") or {}
    open_access = item.get("openAccessPdf") or {}
    return Candidate(
        title=item.get("title"),
        authors=[a.get("name", "") for a in (item.get("authors") or []) if a.get("name")],
        year=item.get("year"),
        venue=item.get("venue") or None,
        abstract=item.get("abstract"),
        doi=external_ids.get("DOI"),
        url=item.get("url"),
        pdf_url=open_access.get("url"),
        source="semantic_scholar",
        external_id=item.get("paperId") or "",
    )


def search_semantic_scholar(
    query: str,
    limit: int = 10,
    settings: "Settings | None" = None,
    timeout: float = 10.0,
) -> list[Candidate]:
    settings = settings or get_settings()
    data = _get(
        f"{_BASE_URL}/paper/search",
        {"query": query, "limit": limit, "fields": _FIELDS},
        settings,
        timeout,
    )
    return [_to_candidate(item) for item in data.get("data", []) if item]


def get_references(
    paper_id: str,
    limit: int = 25,
    settings: "Settings | None" = None,
    timeout: float = 10.0,
) -> list[Candidate]:
    # Backward: what `paper_id` cites.
    settings = settings or get_settings()
    data = _get(
        f"{_BASE_URL}/paper/{paper_id}/references",
        {"fields": _FIELDS, "limit": limit},
        settings,
        timeout,
    )
    return [
        _to_candidate(item["citedPaper"])
        for item in data.get("data", [])
        if item.get("citedPaper")
    ]


def get_citations(
    paper_id: str,
    limit: int = 25,
    settings: "Settings | None" = None,
    timeout: float = 10.0,
) -> list[Candidate]:
    # Forward: what cites `paper_id`.
    settings = settings or get_settings()
    data = _get(
        f"{_BASE_URL}/paper/{paper_id}/citations",
        {"fields": _FIELDS, "limit": limit},
        settings,
        timeout,
    )
    return [
        _to_candidate(item["citingPaper"])
        for item in data.get("data", [])
        if item.get("citingPaper")
    ]


def resolve_paper_id(
    doi: str | None = None,
    title: str | None = None,
    settings: "Settings | None" = None,
    timeout: float = 10.0,
) -> str | None:
    # Best-effort lookup of an S2 paperId, so a citation graph can be requested for a library
    # paper that has no S2 id stored. A DOI match is exact; a title match takes the top search
    # hit and is not guaranteed to be the same paper.
    settings = settings or get_settings()

    if doi:
        try:
            response = httpx.get(
                f"{_BASE_URL}/paper/DOI:{doi}",
                params={"fields": "paperId"},
                headers=_headers(settings),
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise DiscoverySourceError("semantic_scholar", str(exc)) from exc
        if response.status_code == 429:
            raise DiscoverySourceError(
                "semantic_scholar", "rate limited — try again shortly, or set S2_API_KEY"
            )
        if response.status_code != 404:
            response.raise_for_status()
            paper_id = response.json().get("paperId")
            if paper_id:
                return paper_id

    if title:
        data = _get(
            f"{_BASE_URL}/paper/search",
            {"query": title, "limit": 1, "fields": "paperId"},
            settings,
            timeout,
        )
        results = data.get("data") or []
        if results:
            return results[0].get("paperId")

    return None
