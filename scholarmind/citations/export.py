from dataclasses import replace

from scholarmind.citations.formatter import BibTeXFormatter, unique_bibtex_key
from scholarmind.citations.metadata import NormalizedMetadata, normalize_metadata
from scholarmind.retrieval.papers import PaperSummary


def paper_to_metadata(paper: "PaperSummary") -> NormalizedMetadata:
    # Resolves against Crossref/OpenAlex/Semantic Scholar first (same as citations produced by
    # Ask/agent answers), so a paper whose PDF embeds no /Author (common for LaTeX-built PDFs)
    # still gets real author/year/venue instead of "Unknown Author (n.d.)". Falls back to the
    # paper's own DOI/venue if resolution didn't find one; falls back to the library's stored
    # title/authors/year (whatever the PDF itself had) if every external source fails.
    resolved = normalize_metadata(paper.label, paper.authors, paper.year)
    return replace(
        resolved,
        doi=resolved.doi or paper.doi,
        venue=resolved.venue or paper.venue,
    )


def export_bibtex(papers: list["PaperSummary"]) -> str:
    formatter = BibTeXFormatter()
    used_keys: set[str] = set()
    entries = []
    for paper in papers:
        metadata = paper_to_metadata(paper)
        key = unique_bibtex_key(metadata, used_keys)
        used_keys.add(key)
        entries.append(formatter.format(metadata, key=key))
    return "\n\n".join(entries)
