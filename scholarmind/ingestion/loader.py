import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from scholarmind.ingestion.multimodal import (
    ExtractedFigure,
    ExtractedTable,
    extract_tables_and_figures,
)

logger = logging.getLogger("scholarmind.ingestion.loader")


@dataclass
class RawDocument:
    source_path: Path
    pages: list[str]  # raw extracted text per page, in page order
    pdf_metadata: dict  # raw pypdf metadata dict (e.g. reader.metadata), may be empty
    content_hash: str = ""  # sha256 of the source bytes; also used downstream as paper_id
    tables: list[ExtractedTable] = field(default_factory=list)
    figures: list[ExtractedFigure] = field(default_factory=list)


def load_pdf(path: Path, images_dir: Path | None = None) -> RawDocument:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]

    if all(not page.strip() for page in pages):
        logger.warning(
            "PDF appears to be scanned/image-only (no extractable text): %s", path
        )
        pages = ["" for _ in pages]

    metadata = dict(reader.metadata) if reader.metadata else {}
    content_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    tables: list[ExtractedTable] = []
    figures: list[ExtractedFigure] = []
    if images_dir is not None:
        # Never let table/figure extraction failure block the (already working) text pipeline.
        try:
            tables, figures = extract_tables_and_figures(
                Path(path), images_dir / content_hash
            )
        except Exception:
            logger.warning("Table/figure extraction failed for %s — continuing text-only", path)

    return RawDocument(
        source_path=Path(path),
        pages=pages,
        pdf_metadata=metadata,
        content_hash=content_hash,
        tables=tables,
        figures=figures,
    )


def load_path(path: Path, images_dir: Path | None = None) -> list[RawDocument]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_dir():
        pdf_files = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
        )
        return [load_pdf(p, images_dir=images_dir) for p in pdf_files]

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")

    return [load_pdf(path, images_dir=images_dir)]
