import re
from dataclasses import dataclass

from scholarmind.retrieval.dense import DenseResult

_MARKER_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Citation:
    index: int
    paper_id: str
    title: str | None
    authors: list[str]
    year: int | None
    section: str | None
    page_start: int
    page_end: int
    text: str


@dataclass
class VerifiedAnswer:
    text: str
    citations: list[Citation]
    invalid_citation_markers: list[int]


def build_source_block(sources: list["DenseResult"]) -> str:
    paragraphs = []
    for index, source in enumerate(sources, start=1):
        parenthetical = ""
        if source.authors:
            author_text = ", ".join(source.authors)
            if source.year is not None:
                parenthetical = f" ({author_text}, {source.year})"
            else:
                parenthetical = f" ({author_text})"
        elif source.year is not None:
            parenthetical = f" ({source.year})"

        section = f"{source.section}: " if source.section else ""
        paragraphs.append(f"[{index}]{parenthetical} {section}{source.text}")

    return "\n\n".join(paragraphs)


def extract_citation_markers(text: str) -> list[int]:
    markers: list[int] = []
    for match in _MARKER_PATTERN.finditer(text):
        for part in match.group(1).split(","):
            marker = int(part.strip())
            if marker not in markers:
                markers.append(marker)
    return markers


def verify_citations(text: str, sources: list["DenseResult"]) -> VerifiedAnswer:
    markers = extract_citation_markers(text)
    citations: list[Citation] = []
    invalid_citation_markers: list[int] = []

    for marker in markers:
        if 1 <= marker <= len(sources):
            source = sources[marker - 1]
            citations.append(
                Citation(
                    index=marker,
                    paper_id=source.paper_id,
                    title=source.title,
                    authors=source.authors,
                    year=source.year,
                    section=source.section,
                    page_start=source.page_start,
                    page_end=source.page_end,
                    text=source.text,
                )
            )
        else:
            invalid_citation_markers.append(marker)

    return VerifiedAnswer(
        text=text,
        citations=citations,
        invalid_citation_markers=invalid_citation_markers,
    )
