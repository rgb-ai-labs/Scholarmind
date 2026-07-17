import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scholarmind.ingestion.loader import RawDocument

HEADING_RE = re.compile(r"^(Abstract|[0-9]+\.\s+\S.*)$")


@dataclass
class ParsedSection:
    heading: str | None  # None for text before the first detected heading
    text: str  # cleaned, joined text for this section
    page_start: int  # 1-indexed
    page_end: int  # 1-indexed


@dataclass
class ParsedDocument:
    paper_id: str  # stable content hash of the source file, hex string
    title: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    sections: list[ParsedSection]


@dataclass
class _SectionBuilder:
    heading: str | None
    page_start: int
    lines: list[str] = field(default_factory=list)
    page_end: int = 0


def _join_lines(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip() == "":
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(" ".join(line.split()))
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _split_authors(raw_author: str | None) -> list[str]:
    if not raw_author or not raw_author.strip():
        return []
    parts = re.split(r"[,;]", raw_author)
    return [p.strip() for p in parts if p.strip()]


def _extract_year(subject: str | None) -> int | None:
    if subject is None:
        return None
    subject = subject.strip()
    if re.fullmatch(r"[0-9]{4}", subject):
        return int(subject)
    return None


def parse_document(raw: "RawDocument") -> ParsedDocument:
    paper_id = hashlib.sha256(raw.source_path.read_bytes()).hexdigest()

    metadata = raw.pdf_metadata or {}
    title = metadata.get("/Title") or None
    authors = _split_authors(metadata.get("/Author"))
    year = _extract_year(metadata.get("/Subject"))
    venue = None

    sections: list[_SectionBuilder] = []
    current = _SectionBuilder(heading=None, page_start=1)

    num_pages = len(raw.pages)
    for page_num, page_text in enumerate(raw.pages, start=1):
        for line in page_text.splitlines():
            if HEADING_RE.match(line.strip()):
                if current.heading is not None or any(entry.strip() for entry in current.lines):
                    current.page_end = page_num
                    sections.append(current)
                current = _SectionBuilder(heading=line.strip(), page_start=page_num)
            else:
                current.lines.append(line)

    current.page_end = num_pages if num_pages else current.page_start
    if current.heading is not None or any(entry.strip() for entry in current.lines):
        sections.append(current)

    parsed_sections = [
        ParsedSection(
            heading=s.heading,
            text=_join_lines(s.lines),
            page_start=s.page_start,
            page_end=s.page_end,
        )
        for s in sections
    ]

    return ParsedDocument(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        sections=parsed_sections,
    )
