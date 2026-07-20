import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("scholarmind.ingestion.multimodal")

_CAPTION_RE_TEMPLATE = r"^{prefix}\s+\d+[.:]?.*"
_EQUATION_NUMBER_RE = re.compile(r"\(\d+\)\s*$")
_MATH_SYMBOL_RE = re.compile(r"[=≈≠≤≥±∑∫∏√∂∇×÷θαβγδλμσΣΩ]")


@dataclass
class ExtractedTable:
    page: int
    caption: str | None
    markdown: str


@dataclass
class ExtractedFigure:
    page: int
    caption: str | None
    image_path: str


@dataclass
class ExtractedEquation:
    page: int
    text: str
    context: str


def _nearby_caption(page_text: str, prefix: str) -> str | None:
    # Heuristic only: the first line on the page starting with "Table N"/"Figure N", not a
    # true layout-proximity match to the specific table/image's bounding box. Good enough for
    # single-table/single-figure pages; a page with multiple tables/figures may mislabel which
    # caption belongs to which one.
    pattern = re.compile(_CAPTION_RE_TEMPLATE.format(prefix=prefix), re.IGNORECASE)
    for line in page_text.splitlines():
        stripped = line.strip()
        if pattern.match(stripped):
            return stripped
    return None


def extract_tables_and_figures(
    pdf_path: Path, images_dir: Path
) -> tuple[list["ExtractedTable"], list["ExtractedFigure"]]:
    # Robust to failure end-to-end: any problem here (missing/broken PyMuPDF, a corrupt PDF, a
    # single bad page/table/image) degrades to fewer extracted tables/figures, never raises —
    # the caller's plain-text ingestion must keep working regardless.
    try:
        import pymupdf
    except Exception:
        logger.warning("PyMuPDF unavailable — skipping table/figure extraction for %s", pdf_path)
        return [], []

    tables: list[ExtractedTable] = []
    figures: list[ExtractedFigure] = []

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception:
        logger.warning(
            "Could not open %s with PyMuPDF — skipping table/figure extraction", pdf_path
        )
        return [], []

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1

            try:
                page_text = page.get_text("text")
            except Exception:
                page_text = ""

            try:
                found_tables = page.find_tables().tables
            except Exception:
                logger.warning("Table detection failed on page %d of %s", page_num, pdf_path)
                found_tables = []

            for table in found_tables:
                try:
                    markdown = table.to_markdown()
                    if not markdown.strip():
                        continue
                    caption = _nearby_caption(page_text, "Table")
                    tables.append(
                        ExtractedTable(page=page_num, caption=caption, markdown=markdown)
                    )
                except Exception:
                    logger.warning(
                        "Failed to extract one table on page %d of %s", page_num, pdf_path
                    )

            try:
                image_infos = page.get_images(full=True)
            except Exception:
                logger.warning("Image detection failed on page %d of %s", page_num, pdf_path)
                image_infos = []

            for image_index, image_info in enumerate(image_infos):
                try:
                    xref = image_info[0]
                    pixmap = pymupdf.Pixmap(doc, xref)
                    if pixmap.n > 4:  # CMYK/other → normalize to RGB
                        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
                    images_dir.mkdir(parents=True, exist_ok=True)
                    image_path = images_dir / f"page{page_num}_img{image_index}.png"
                    pixmap.save(str(image_path))
                    caption = _nearby_caption(page_text, "Figure")
                    figures.append(
                        ExtractedFigure(page=page_num, caption=caption, image_path=str(image_path))
                    )
                except Exception:
                    logger.warning(
                        "Failed to extract one image on page %d of %s", page_num, pdf_path
                    )
    finally:
        doc.close()

    return tables, figures


def extract_equations(pages: list[str]) -> list["ExtractedEquation"]:
    # Heuristic, not true equation OCR/LaTeX recognition: flags lines that both end with a
    # parenthesized equation number ("... (3)") and contain a math-ish symbol. Deliberately
    # conservative (favors missing equations over flagging ordinary prose).
    equations: list[ExtractedEquation] = []
    for page_num, page_text in enumerate(pages, start=1):
        lines = page_text.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if _EQUATION_NUMBER_RE.search(stripped) and _MATH_SYMBOL_RE.search(stripped):
                context_lines = lines[max(0, i - 1) : i + 2]
                context = " ".join(entry.strip() for entry in context_lines if entry.strip())
                equations.append(ExtractedEquation(page=page_num, text=stripped, context=context))
    return equations
