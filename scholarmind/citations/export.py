from dataclasses import replace
from typing import TYPE_CHECKING

from scholarmind.citations.formatter import BibTeXFormatter, unique_bibtex_key
from scholarmind.citations.metadata import NormalizedMetadata, normalize_metadata
from scholarmind.retrieval.papers import PaperSummary, persist_resolved_metadata

if TYPE_CHECKING:
    from scholarmind.config import Settings


def paper_to_metadata(
    paper: "PaperSummary", settings: "Settings | None" = None
) -> NormalizedMetadata:
    # A resolution already persisted from an earlier successful lookup (see below) is used
    # directly — no network call, and it still works fully offline. resolved_source is None
    # until a lookup has actually succeeded once; it's never set for a failed attempt, so an
    # unresolved paper keeps retrying live on every render exactly as before.
    if paper.resolved_source is not None:
        return NormalizedMetadata(
            doi=paper.resolved_doi or paper.doi,
            title=paper.label,
            authors=paper.resolved_authors or paper.authors,
            year=paper.resolved_year,
            venue=paper.resolved_venue or paper.venue,
            source=paper.resolved_source,
        )

    # Resolves against Crossref/OpenAlex/Semantic Scholar first (same as citations produced by
    # Ask/agent answers), so a paper whose PDF embeds no /Author (common for LaTeX-built PDFs)
    # still gets real author/year/venue instead of "Unknown Author (n.d.)". Falls back to the
    # paper's own DOI/venue if resolution didn't find one; falls back to the library's stored
    # title/authors/year (whatever the PDF itself had) if every external source fails.
    resolved = normalize_metadata(paper.label, paper.authors, paper.year)

    if resolved.source != "unresolved":
        persist_resolved_metadata(
            paper.paper_id,
            doi=resolved.doi,
            authors=resolved.authors,
            year=resolved.year,
            venue=resolved.venue,
            source=resolved.source,
            settings=settings,
        )

    return replace(
        resolved,
        doi=resolved.doi or paper.doi,
        venue=resolved.venue or paper.venue,
    )


def export_bibtex(papers: list["PaperSummary"], settings: "Settings | None" = None) -> str:
    formatter = BibTeXFormatter()
    used_keys: set[str] = set()
    entries = []
    for paper in papers:
        metadata = paper_to_metadata(paper, settings)
        key = unique_bibtex_key(metadata, used_keys)
        used_keys.add(key)
        entries.append(formatter.format(metadata, key=key))
    return "\n\n".join(entries)
