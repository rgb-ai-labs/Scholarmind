"""Generate a second, distinct-topic sample PDF for the evaluation harness.

Run once with `uv run python scripts/generate_sample_pdf_2.py`; the output is
committed under tests/fixtures/sample_paper_2.pdf. Its topic (tidal energy)
shares no vocabulary with sample_paper.pdf (retrieval-augmented generation),
so paper-level retrieval precision/recall on the two-paper corpus is
non-degenerate.
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUTPUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_paper_2.pdf"

TITLE = "Tidal Energy Harvesting Efficiency in Coastal Turbine Arrays"
AUTHORS = "Marie Curie, Nikola Tesla"

PAGE_1_LINES = [
    ("Title", TITLE),
    ("Authors", AUTHORS),
    ("", ""),
    ("heading", "Abstract"),
    ("body", "This paper studies the efficiency of underwater turbines that convert"),
    ("body", "tidal currents into electricity across shallow coastal shelves. We"),
    ("body", "measure power output against tidal flow velocity and blade geometry."),
    ("", ""),
    ("heading", "1. Introduction"),
    ("body", "Coastal tides are a predictable renewable resource. Submerged turbine"),
    ("body", "arrays capture kinetic energy from the moving water column as tides"),
    ("body", "ebb and flood twice daily along the shoreline."),
]

PAGE_2_LINES = [
    ("heading", "2. Methodology"),
    ("body", "We deploy a scaled turbine array in a tidal channel and log rotor"),
    ("body", "torque against measured current speed over a full lunar cycle, then"),
    ("body", "compute the hydrodynamic capacity factor of each blade profile."),
    ("", ""),
    ("heading", "3. Conclusion"),
    ("body", "Optimally spaced coastal turbines sustain high capacity factors and"),
    ("body", "make tidal arrays a viable baseload marine energy source."),
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
    c.setSubject("2023")
    draw_page(c, PAGE_1_LINES)
    draw_page(c, PAGE_2_LINES)
    c.save()
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
