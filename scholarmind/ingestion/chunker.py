from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scholarmind.ingestion.parser import ParsedDocument


@dataclass
class Chunk:
    paper_id: str
    title: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    section: str | None
    page_start: int
    page_end: int
    chunk_index: int  # 0-based, increasing across the whole document
    text: str


def _split_words(paragraph: str, chunk_size: int) -> list[str]:
    words = paragraph.split()
    pieces: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if current and len(candidate) > chunk_size:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _segments_for_section(text: str, chunk_size: int) -> list[str]:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    segments: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            segments.extend(_split_words(paragraph, chunk_size))
        else:
            segments.append(paragraph)
    return segments


def _pack_segments(segments: list[str], chunk_size: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for segment in segments:
        candidate = segment if not current else f"{current}\n\n{segment}"
        if current and len(candidate) > chunk_size:
            chunks.append(current)
            current = segment
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _apply_overlap(base_chunks: list[str], chunk_overlap: int) -> list[str]:
    final_texts: list[str] = []
    for i, base in enumerate(base_chunks):
        if i == 0 or chunk_overlap <= 0:
            final_texts.append(base)
            continue
        prev_final = final_texts[i - 1]
        overlap_text = prev_final[-chunk_overlap:]
        final_texts.append(overlap_text + base)
    return final_texts


def chunk_document(
    doc: "ParsedDocument",
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0

    for section in doc.sections:
        if not section.text.strip():
            continue

        segments = _segments_for_section(section.text, chunk_size)
        base_chunks = _pack_segments(segments, chunk_size)
        final_texts = _apply_overlap(base_chunks, chunk_overlap)

        for text in final_texts:
            chunks.append(
                Chunk(
                    paper_id=doc.paper_id,
                    title=doc.title,
                    authors=doc.authors,
                    year=doc.year,
                    venue=doc.venue,
                    section=section.heading,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    chunk_index=chunk_index,
                    text=text,
                )
            )
            chunk_index += 1

    return chunks
