import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger("scholarmind.ingestion.loader")


@dataclass
class RawDocument:
    source_path: Path
    pages: list[str]  # raw extracted text per page, in page order
    pdf_metadata: dict  # raw pypdf metadata dict (e.g. reader.metadata), may be empty


def load_pdf(path: Path) -> RawDocument:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]

    if all(not page.strip() for page in pages):
        logger.warning(
            "PDF appears to be scanned/image-only (no extractable text): %s", path
        )
        pages = ["" for _ in pages]

    metadata = dict(reader.metadata) if reader.metadata else {}

    return RawDocument(source_path=Path(path), pages=pages, pdf_metadata=metadata)


def load_path(path: Path) -> list[RawDocument]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_dir():
        pdf_files = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
        )
        return [load_pdf(p) for p in pdf_files]

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")

    return [load_pdf(path)]
