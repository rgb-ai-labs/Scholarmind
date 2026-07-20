from typing import TYPE_CHECKING

import httpx

from scholarmind.discovery.models import Candidate, DiscoverySourceError

if TYPE_CHECKING:
    from scholarmind.config import Settings

_BASE_URL = "https://api.openalex.org/works"
_USER_AGENT = "ScholarMind/0.1 (https://github.com/scholarmind; mailto:contact@example.com)"


def _headers() -> dict[str, str]:
    return {"User-Agent": _USER_AGENT}


def _get(url: str, params: dict, timeout: float) -> dict:
    try:
        response = httpx.get(url, params=params, headers=_headers(), timeout=timeout)
        if response.status_code == 429:
            raise DiscoverySourceError("openalex", "rate limited — try again shortly")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DiscoverySourceError("openalex", str(exc)) from exc
    return response.json()


def _strip_doi_prefix(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.removeprefix("https://doi.org/")


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, indices in inverted_index.items():
        for index in indices:
            positions[index] = word
    return " ".join(positions[index] for index in sorted(positions)) or None


def _to_candidate(item: dict) -> Candidate:
    authorships = item.get("authorships") or []
    authors = [
        (a.get("author") or {}).get("display_name", "")
        for a in authorships
        if (a.get("author") or {}).get("display_name")
    ]

    primary_location = item.get("primary_location") or {}
    open_access = item.get("open_access") or {}
    pdf_url = primary_location.get("pdf_url")
    if not pdf_url and open_access.get("is_oa"):
        pdf_url = open_access.get("oa_url")

    source_info = primary_location.get("source") or {}
    work_id = item.get("id") or ""

    return Candidate(
        title=item.get("title") or item.get("display_name"),
        authors=authors,
        year=item.get("publication_year"),
        venue=source_info.get("display_name"),
        abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
        doi=_strip_doi_prefix(item.get("doi")),
        url=work_id or None,
        pdf_url=pdf_url,
        source="openalex",
        external_id=work_id.rsplit("/", 1)[-1] if work_id else "",
    )


def search_openalex(
    query: str,
    limit: int = 10,
    settings: "Settings | None" = None,  # unused; kept for a uniform call signature
    timeout: float = 10.0,
) -> list[Candidate]:
    data = _get(_BASE_URL, {"search": query, "per_page": limit}, timeout)
    return [_to_candidate(item) for item in data.get("results", [])]
