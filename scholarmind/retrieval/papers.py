import hashlib
import re
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from scholarmind.config import Settings, get_settings
from scholarmind.retrieval.dense import DenseResult


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.strip().lower().removeprefix("https://doi.org/")


def normalize_title(title: str | None) -> str | None:
    # Collapsed-alnum + hashed so two titles differing only in case/punctuation/whitespace
    # (e.g. "Attention Is All You Need" vs "attention is all you need!!") key identically.
    if not title:
        return None
    collapsed = re.sub(r"[^a-z0-9]+", "", title.lower())
    if not collapsed:
        return None
    return hashlib.sha256(collapsed.encode()).hexdigest()


@dataclass
class PaperSummary:
    paper_id: str
    label: str  # title if the parser captured one, else the source filename, else paper_id
    chunk_count: int
    ingested_at: float
    doi: str | None = None
    is_metadata_only: bool = False  # True for discovery records ingested without a PDF
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None


def list_papers(settings: "Settings | None" = None) -> list["PaperSummary"]:
    settings = settings or get_settings()

    client = QdrantClient(path=settings.qdrant_path)
    try:
        if not client.collection_exists(settings.qdrant_collection):
            return []

        payload_fields = [
            "paper_id",
            "title",
            "source_filename",
            "ingested_at",
            "doi",
            "is_metadata_only",
            "authors",
            "year",
            "venue",
        ]

        papers: dict[str, dict] = {}
        offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=settings.qdrant_collection,
                with_payload=payload_fields,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for point in batch:
                payload = point.payload
                paper_id = payload["paper_id"]
                entry = papers.setdefault(
                    paper_id,
                    {
                        "title": None,
                        "source_filename": None,
                        "chunk_count": 0,
                        "ingested_at": 0.0,
                        "doi": None,
                        "is_metadata_only": False,
                        "authors": [],
                        "year": None,
                        "venue": None,
                    },
                )
                entry["chunk_count"] += 1
                entry["title"] = entry["title"] or payload.get("title")
                entry["source_filename"] = entry["source_filename"] or payload.get(
                    "source_filename"
                )
                entry["ingested_at"] = max(entry["ingested_at"], payload.get("ingested_at") or 0.0)
                entry["doi"] = entry["doi"] or payload.get("doi")
                entry["is_metadata_only"] = entry["is_metadata_only"] or bool(
                    payload.get("is_metadata_only")
                )
                entry["authors"] = entry["authors"] or payload.get("authors") or []
                entry["year"] = entry["year"] if entry["year"] is not None else payload.get("year")
                entry["venue"] = entry["venue"] or payload.get("venue")
            if next_offset is None:
                break
            offset = next_offset

        results = [
            PaperSummary(
                paper_id=paper_id,
                label=data["title"] or data["source_filename"] or paper_id,
                chunk_count=data["chunk_count"],
                ingested_at=data["ingested_at"],
                doi=data["doi"],
                is_metadata_only=data["is_metadata_only"],
                authors=data["authors"],
                year=data["year"],
                venue=data["venue"],
            )
            for paper_id, data in papers.items()
        ]
        return sorted(results, key=lambda paper: paper.ingested_at, reverse=True)
    finally:
        client.close()


def get_paper_chunks(
    paper_id: str, settings: "Settings | None" = None
) -> list["DenseResult"]:
    settings = settings or get_settings()

    client = QdrantClient(path=settings.qdrant_path)
    try:
        if not client.collection_exists(settings.qdrant_collection):
            return []

        paper_filter = Filter(
            must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
        )

        points = []
        offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=paper_filter,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        results = [DenseResult(**point.payload, score=0.0) for point in points]
        return sorted(results, key=lambda result: result.chunk_index)
    finally:
        client.close()


def find_papers_by_identifier(
    identifier: str, settings: "Settings | None" = None
) -> list["PaperSummary"]:
    # Resolves a user-typed identifier to library papers, in a predictable priority order so a
    # deliberate delete never has to guess: (1) an exact paper_id, (2) a paper_id prefix (the
    # short IDs shown in listings like `dedupe`), (3) a case-insensitive title/label substring.
    # Higher tiers short-circuit — an exact ID match never also drags in title matches.
    ident = identifier.strip()
    if not ident:
        return []

    papers = list_papers(settings)

    exact = [p for p in papers if p.paper_id == ident]
    if exact:
        return exact

    prefix = [p for p in papers if p.paper_id.startswith(ident)]
    if prefix:
        return prefix

    lowered = ident.lower()
    return [p for p in papers if lowered in p.label.lower()]


def delete_papers(paper_ids: list[str], settings: "Settings | None" = None) -> int:
    # Removes every chunk belonging to the given paper_id(s) from the store and returns the
    # number of chunks actually removed. The source PDF (under the uploads folder) is left
    # untouched, so a deleted paper can always be re-ingested.
    settings = settings or get_settings()
    if not paper_ids:
        return 0

    client = QdrantClient(path=settings.qdrant_path)
    try:
        if not client.collection_exists(settings.qdrant_collection):
            return 0
        before = client.count(settings.qdrant_collection, exact=True).count
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=Filter(
                must=[FieldCondition(key="paper_id", match=MatchAny(any=paper_ids))]
            ),
        )
        after = client.count(settings.qdrant_collection, exact=True).count
        return before - after
    finally:
        client.close()
