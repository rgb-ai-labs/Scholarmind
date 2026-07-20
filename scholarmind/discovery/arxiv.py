from typing import TYPE_CHECKING
from xml.etree import ElementTree

import httpx

from scholarmind.discovery.models import Candidate, DiscoverySourceError

if TYPE_CHECKING:
    from scholarmind.config import Settings

_API_URL = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_USER_AGENT = "ScholarMind/0.1 (https://github.com/scholarmind; mailto:contact@example.com)"


def search_arxiv(
    query: str,
    limit: int = 10,
    settings: "Settings | None" = None,  # unused; kept for a uniform call signature across sources
    timeout: float = 10.0,
) -> list[Candidate]:
    try:
        response = httpx.get(
            _API_URL,
            params={"search_query": f"all:{query}", "start": 0, "max_results": limit},
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DiscoverySourceError("arxiv", str(exc)) from exc

    try:
        root = ElementTree.fromstring(response.text)
    except ElementTree.ParseError as exc:
        raise DiscoverySourceError("arxiv", f"could not parse response: {exc}") from exc

    return [_to_candidate(entry) for entry in root.findall(f"{_ATOM_NS}entry")]


def _to_candidate(entry: ElementTree.Element) -> Candidate:
    title = (entry.findtext(f"{_ATOM_NS}title") or "").strip()
    title = " ".join(title.split()) or None

    abstract = (entry.findtext(f"{_ATOM_NS}summary") or "").strip()
    abstract = " ".join(abstract.split()) or None

    entry_url = (entry.findtext(f"{_ATOM_NS}id") or "").strip()
    external_id = entry_url.rsplit("/", 1)[-1] if entry_url else ""

    authors = [
        (author.findtext(f"{_ATOM_NS}name") or "").strip()
        for author in entry.findall(f"{_ATOM_NS}author")
    ]

    published = entry.findtext(f"{_ATOM_NS}published") or ""
    year = int(published[:4]) if published[:4].isdigit() else None

    doi = entry.findtext(f"{_ARXIV_NS}doi")

    pdf_url = None
    for link in entry.findall(f"{_ATOM_NS}link"):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = link.get("href")
            break

    return Candidate(
        title=title,
        authors=[name for name in authors if name],
        year=year,
        venue="arXiv",
        abstract=abstract,
        doi=doi,
        url=entry_url or None,
        pdf_url=pdf_url,
        source="arxiv",
        external_id=external_id,
    )
