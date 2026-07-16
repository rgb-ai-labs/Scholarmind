"""Generate the tiny 2-page sample PDF used by ingestion tests.

Run once with `uv run python scripts/generate_sample_pdf.py`; the output is
committed under tests/fixtures/sample_paper.pdf so tests don't need to
regenerate it.
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUTPUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_paper.pdf"

TITLE = "A Study of Retrieval-Augmented Generation for Scholarly Question Answering"
AUTHORS = "Ada Lovelace, Grace Hopper"

PAGE_1_LINES = [
    ("Title", TITLE),
    ("Authors", AUTHORS),
    ("", ""),
    ("heading", "Abstract"),
    ("body", "This paper investigates retrieval-augmented generation (RAG) as a"),
    ("body", "method for answering research questions grounded in a document"),
    ("body", "corpus. We evaluate hybrid retrieval and reranking strategies."),
    ("", ""),
    ("heading", "1. Introduction"),
    ("body", "Large language models are prone to hallucination when asked about"),
    ("body", "specific facts. Retrieval augmentation mitigates this by grounding"),
    ("body", "generation in retrieved source passages."),
]

PAGE_2_LINES = [
    ("heading", "2. Methodology"),
    ("body", "We chunk each paper into paragraph-sized units, embed them with a"),
    ("body", "local sentence-transformers model, and store vectors alongside"),
    ("body", "metadata in an embedded Qdrant collection."),
    ("", ""),
    ("heading", "3. Conclusion"),
    ("body", "Grounding every generated claim in a retrieved chunk is essential"),
    ("body", "for trustworthy scholarly assistants."),
]


def draw_page(c: canvas.Canvas, lines: list[tuple[str, str]]) -> None:
    y = 740
    for kind, text in lines:
        if not text:
            y -= 14
            continue
        if kind == "Title":
            c.setFont("Helvetica-Bold", 16)
        elif kind == "Authors":
            c.setFont("Helvetica", 11)
        elif kind == "heading":
            c.setFont("Helvetica-Bold", 13)
        else:
            c.setFont("Helvetica", 10)
        c.drawString(72, y, text)
        y -= 18
    c.showPage()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=letter)
    c.setTitle(TITLE)
    c.setAuthor(AUTHORS)
    c.setSubject("2024")
    draw_page(c, PAGE_1_LINES)
    draw_page(c, PAGE_2_LINES)
    c.save()
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
