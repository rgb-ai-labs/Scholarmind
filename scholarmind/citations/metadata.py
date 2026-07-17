from dataclasses import dataclass

import httpx

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
    key = (title, tuple(authors), year)
    if key in _CACHE:
        return _CACHE[key]

    if title is None or not title.strip():
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
        result = _unresolved(title, authors, year)

    _CACHE[key] = result
    return result
