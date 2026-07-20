"""Generate a small PDF with a table, an embedded image, and an equation-like line, used to
test multimodal (table/figure/equation) extraction.

Run once with `uv run python scripts/generate_sample_pdf_multimodal.py`; the output is
committed under tests/fixtures/sample_paper_multimodal.pdf so tests don't need to regenerate it.
"""

import io
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUTPUT = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_paper_multimodal.pdf"
)
IMAGE_PATH = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "_multimodal_figure.png"
)

TITLE = "Multimodal Extraction Benchmarks for Scholarly Documents"
AUTHORS = "Marie Curie, Rosalind Franklin"


def _make_figure_image() -> None:
    image = Image.new("RGB", (200, 120), color=(70, 130, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    IMAGE_PATH.write_bytes(buffer.getvalue())


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _make_figure_image()

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=letter, title=TITLE, author=AUTHORS)

    story = [
        Paragraph(TITLE, styles["Title"]),
        Paragraph(AUTHORS, styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Abstract", styles["Heading2"]),
        Paragraph(
            "This paper reports benchmark results across three retrieval configurations, "
            "illustrated in Figure 1 and summarized in Table 1, and derives a scoring "
            "equation used throughout the evaluation.",
            styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph("1. Results", styles["Heading2"]),
        Paragraph("Table 1: Retrieval precision and recall by configuration.", styles["Normal"]),
        Table(
            [
                ["Configuration", "Precision", "Recall"],
                ["Dense only", "0.71", "0.65"],
                ["Hybrid (dense + BM25)", "0.84", "0.79"],
                ["Hybrid + reranker", "0.91", "0.88"],
            ],
            style=TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ]
            ),
        ),
        Spacer(1, 12),
        Paragraph("Figure 1: Overview of the hybrid retrieval pipeline.", styles["Normal"]),
        RLImage(str(IMAGE_PATH), width=2 * inch, height=1.2 * inch),
        Spacer(1, 12),
        Paragraph("2. Scoring", styles["Heading2"]),
        Paragraph(
            "The final relevance score combines dense and sparse signals as follows.",
            styles["Normal"],
        ),
        Paragraph("score(q, d) = a * dense(q, d) + b * sparse(q, d) (1)", styles["Normal"]),
        Paragraph(
            "where a and b are tunable weights fit on a held-out validation split.",
            styles["Normal"],
        ),
    ]

    doc.build(story)
    IMAGE_PATH.unlink()
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
