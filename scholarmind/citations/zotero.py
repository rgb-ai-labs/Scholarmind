from dataclasses import dataclass

import httpx

from scholarmind.citations.metadata import NormalizedMetadata

_USER_AGENT = "ScholarMind/0.1 (https://github.com/scholarmind; mailto:contact@example.com)"
_BATCH_SIZE = 50  # Zotero's /items endpoint accepts at most 50 objects per request


class ZoteroError(Exception):
    pass


@dataclass
class ZoteroPushResult:
    pushed: int
    failed: int
    errors: list[str]


def _to_zotero_item(metadata: "NormalizedMetadata") -> dict:
    creators = []
    for author in metadata.authors:
        tokens = author.split()
        if len(tokens) >= 2:
            creators.append(
                {
                    "creatorType": "author",
                    "firstName": " ".join(tokens[:-1]),
                    "lastName": tokens[-1],
                }
            )
        elif tokens:
            creators.append({"creatorType": "author", "lastName": tokens[0]})

    item: dict = {
        "itemType": "journalArticle",
        "title": metadata.title or "Untitled",
        "creators": creators,
    }
    if metadata.year is not None:
        item["date"] = str(metadata.year)
    if metadata.venue:
        item["publicationTitle"] = metadata.venue
    if metadata.doi:
        item["DOI"] = metadata.doi
    return item


def push_references(
    references: list["NormalizedMetadata"],
    api_key: str,
    library_id: str,
    library_type: str = "user",
    timeout: float = 15.0,
) -> "ZoteroPushResult":
    if not api_key or not library_id:
        raise ZoteroError("Zotero is not configured — set an API key and library ID first.")
    if not references:
        return ZoteroPushResult(pushed=0, failed=0, errors=[])

    url = f"https://api.zotero.org/{library_type}s/{library_id}/items"
    headers = {
        "Zotero-API-Key": api_key,
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
    }

    pushed = 0
    failed = 0
    errors: list[str] = []

    for start in range(0, len(references), _BATCH_SIZE):
        batch = references[start : start + _BATCH_SIZE]
        items = [_to_zotero_item(metadata) for metadata in batch]
        try:
            response = httpx.post(url, json=items, headers=headers, timeout=timeout)
        except httpx.HTTPError as exc:
            raise ZoteroError(str(exc)) from exc

        if response.status_code == 403:
            raise ZoteroError("Zotero rejected the API key/library ID (403 Forbidden).")
        if response.status_code == 429:
            raise ZoteroError("Zotero rate limited this request — try again shortly.")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ZoteroError(f"Zotero request failed (status {response.status_code}).") from exc

        data = response.json()
        successful = data.get("successful") or {}
        failed_map = data.get("failed") or {}
        pushed += len(successful)
        failed += len(failed_map)
        for entry in failed_map.values():
            if isinstance(entry, dict):
                errors.append(entry.get("message", str(entry)))
            else:
                errors.append(str(entry))

    return ZoteroPushResult(pushed=pushed, failed=failed, errors=errors)
