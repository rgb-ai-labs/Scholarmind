from dataclasses import dataclass
from typing import TYPE_CHECKING

from scholarmind.config import get_settings
from scholarmind.retrieval.papers import (
    PaperSummary,
    delete_papers,
    list_papers,
    normalize_title,
)

if TYPE_CHECKING:
    from scholarmind.config import Settings

# delete_papers lives in retrieval.papers (alongside list_papers) as the general library-CRUD
# home; re-exported here so `scholarmind dedupe` and existing importers keep working unchanged.
__all__ = ["DuplicateGroup", "delete_papers", "find_duplicate_paper_groups"]


@dataclass
class DuplicateGroup:
    title: str
    keep: PaperSummary
    remove: list[PaperSummary]


def find_duplicate_paper_groups(settings: "Settings | None" = None) -> list[DuplicateGroup]:
    # Groups library papers by normalized title. A group with more than one paper_id means
    # the same paper was ingested from byte-different sources (e.g. a regenerated PDF, or two
    # separate downloads) — run_ingestion's content-hash paper_id can't catch that on its own.
    settings = settings or get_settings()
    papers = list_papers(settings)

    groups: dict[str, list[PaperSummary]] = {}
    for paper in papers:
        key = normalize_title(paper.label)
        if key is None:
            continue
        groups.setdefault(key, []).append(paper)

    duplicates: list[DuplicateGroup] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        # Keep the paper with the most chunks (most complete ingest); tie-break by earliest
        # ingestion, then paper_id, so the choice is deterministic.
        keeper = min(members, key=lambda p: (-p.chunk_count, p.ingested_at, p.paper_id))
        remove = [p for p in members if p.paper_id != keeper.paper_id]
        duplicates.append(DuplicateGroup(title=keeper.label, keep=keeper, remove=remove))

    return duplicates
