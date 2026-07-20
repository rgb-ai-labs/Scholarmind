from dataclasses import dataclass
from typing import TYPE_CHECKING

from scholarmind.config import get_settings
from scholarmind.discovery.arxiv import search_arxiv
from scholarmind.discovery.dedupe import mark_already_ingested, merge_duplicate_candidates
from scholarmind.discovery.models import Candidate, DiscoverySourceError
from scholarmind.discovery.openalex import search_openalex
from scholarmind.discovery.semantic_scholar import (
    get_citations,
    get_references,
    resolve_paper_id,
    search_semantic_scholar,
)

if TYPE_CHECKING:
    from scholarmind.config import Settings

_SEARCH_SOURCES = (
    ("arxiv", search_arxiv),
    ("semantic_scholar", search_semantic_scholar),
    ("openalex", search_openalex),
)


@dataclass
class DiscoveryResult:
    candidates: list[Candidate]
    errors: list[str]  # readable "<source>: ..." messages for any source that failed


def search_external(
    query: str, settings: "Settings | None" = None, limit_per_source: int = 10
) -> DiscoveryResult:
    settings = settings or get_settings()

    candidates: list[Candidate] = []
    errors: list[str] = []
    for _name, search_fn in _SEARCH_SOURCES:
        try:
            candidates.extend(search_fn(query, limit_per_source, settings))
        except DiscoverySourceError as exc:
            errors.append(str(exc))

    merged = merge_duplicate_candidates(candidates)
    flagged = mark_already_ingested(merged, settings)
    return DiscoveryResult(candidates=flagged, errors=errors)


@dataclass
class CitationGraphResult:
    paper_label: str
    s2_paper_id: str | None
    references: list[Candidate]  # backward: what the paper cites
    citing: list[Candidate]  # forward: what cites the paper
    errors: list[str]


def get_citation_graph(
    doi: str | None = None,
    s2_paper_id: str | None = None,
    title: str | None = None,
    settings: "Settings | None" = None,
    limit: int = 25,
) -> CitationGraphResult:
    settings = settings or get_settings()
    errors: list[str] = []
    label = title or doi or s2_paper_id or "unknown paper"

    paper_id = s2_paper_id
    if paper_id is None:
        try:
            paper_id = resolve_paper_id(doi=doi, title=title, settings=settings)
        except DiscoverySourceError as exc:
            errors.append(str(exc))

    if paper_id is None:
        errors.append(
            "Could not resolve this paper on Semantic Scholar "
            "(no DOI/S2 ID match found for a citation graph)."
        )
        return CitationGraphResult(
            paper_label=label, s2_paper_id=None, references=[], citing=[], errors=errors
        )

    references: list[Candidate] = []
    citing: list[Candidate] = []

    try:
        references = mark_already_ingested(
            get_references(paper_id, limit, settings), settings
        )
    except DiscoverySourceError as exc:
        errors.append(str(exc))

    try:
        citing = mark_already_ingested(get_citations(paper_id, limit, settings), settings)
    except DiscoverySourceError as exc:
        errors.append(str(exc))

    return CitationGraphResult(
        paper_label=label, s2_paper_id=paper_id, references=references, citing=citing, errors=errors
    )
