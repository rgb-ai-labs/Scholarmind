from dataclasses import dataclass


@dataclass
class Candidate:
    title: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str | None
    doi: str | None
    url: str | None  # landing page / abstract page
    pdf_url: str | None  # None when no open-access PDF was found
    source: str  # "arxiv" | "semantic_scholar" | "openalex", or "+"-joined when merged
    external_id: str  # source-specific id (arXiv id, S2 paperId, OpenAlex work id)
    already_ingested: bool = False


class DiscoverySourceError(Exception):
    # Raised for any network/HTTP/rate-limit failure talking to an external source, so callers
    # (the UI) can show a readable "<source> is unavailable: ..." message instead of a traceback.
    def __init__(self, source: str, message: str) -> None:
        self.source = source
        self.message = message
        super().__init__(f"{source}: {message}")
