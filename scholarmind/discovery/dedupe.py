from dataclasses import replace
from typing import TYPE_CHECKING

from scholarmind.discovery.models import Candidate
from scholarmind.retrieval.papers import list_papers, normalize_doi, normalize_title

if TYPE_CHECKING:
    from scholarmind.config import Settings

# Kept as local aliases so the rest of this module (and its tests) can keep referring to
# the short names; the implementations live in retrieval.papers so ingestion/pipeline.py
# can share them without importing this discovery-layer module.
_normalize_doi = normalize_doi
_normalize_title = normalize_title


def _richer(primary: Candidate, secondary: Candidate) -> Candidate:
    combined_source = primary.source
    if secondary.source not in combined_source.split("+"):
        combined_source = f"{combined_source}+{secondary.source}"
    return replace(
        primary,
        abstract=primary.abstract or secondary.abstract,
        doi=primary.doi or secondary.doi,
        pdf_url=primary.pdf_url or secondary.pdf_url,
        venue=primary.venue or secondary.venue,
        url=primary.url or secondary.url,
        source=combined_source,
    )


def merge_duplicate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    # Collapses the same paper surfaced by multiple sources (matched by DOI, else by a
    # normalized-title hash) into one record, keeping the fuller metadata and noting every
    # contributing source.
    def _field_score(candidate: Candidate) -> int:
        return sum(
            1 for field in (candidate.abstract, candidate.doi, candidate.pdf_url, candidate.venue)
            if field
        )

    merged: dict[str, Candidate] = {}
    order: list[str] = []
    for index, candidate in enumerate(candidates):
        key = (
            _normalize_doi(candidate.doi)
            or _normalize_title(candidate.title)
            or f"_unkeyed_{index}"
        )
        if key not in merged:
            merged[key] = candidate
            order.append(key)
            continue
        existing = merged[key]
        if _field_score(existing) >= _field_score(candidate):
            primary, secondary = existing, candidate
        else:
            primary, secondary = candidate, existing
        merged[key] = _richer(primary, secondary)

    return [merged[key] for key in order]


def mark_already_ingested(
    candidates: list[Candidate], settings: "Settings | None" = None
) -> list[Candidate]:
    papers = list_papers(settings)
    known_dois = {_normalize_doi(p.doi) for p in papers if p.doi}
    known_titles = {_normalize_title(p.label) for p in papers if p.label}

    result = []
    for candidate in candidates:
        doi_key = _normalize_doi(candidate.doi)
        title_key = _normalize_title(candidate.title)
        is_duplicate = (doi_key is not None and doi_key in known_dois) or (
            title_key is not None and title_key in known_titles
        )
        result.append(replace(candidate, already_ingested=is_duplicate))
    return result
