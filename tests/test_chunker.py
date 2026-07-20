import re

from scholarmind.ingestion.chunker import Chunk, chunk_document
from scholarmind.ingestion.multimodal import ExtractedEquation, ExtractedFigure, ExtractedTable
from scholarmind.ingestion.parser import ParsedDocument, ParsedSection


def _doc(
    sections: list[ParsedSection],
    tables: list[ExtractedTable] | None = None,
    equations: list[ExtractedEquation] | None = None,
    figures: list[ExtractedFigure] | None = None,
) -> ParsedDocument:
    return ParsedDocument(
        paper_id="paper-id-123",
        title="Sample Title",
        authors=["Ada Lovelace", "Grace Hopper"],
        year=2024,
        venue="ICML",
        sections=sections,
        tables=tables or [],
        equations=equations or [],
        figures=figures or [],
    )


def test_two_short_sections_produce_two_chunks_with_metadata_and_index():
    doc = _doc(
        [
            ParsedSection(
                heading="Abstract",
                text="Short abstract text that easily fits in one chunk.",
                page_start=1,
                page_end=1,
            ),
            ParsedSection(
                heading="1. Introduction",
                text="Short introduction text that also fits in one chunk.",
                page_start=1,
                page_end=2,
            ),
        ]
    )

    chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=0)

    assert len(chunks) == 2

    first, second = chunks
    assert first.section == "Abstract"
    assert first.chunk_index == 0
    assert first.page_start == 1
    assert first.page_end == 1
    assert first.text == doc.sections[0].text

    assert second.section == "1. Introduction"
    assert second.chunk_index == 1
    assert second.page_start == 1
    assert second.page_end == 2
    assert second.text == doc.sections[1].text

    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.paper_id == doc.paper_id
        assert chunk.title == doc.title
        assert chunk.authors == doc.authors
        assert chunk.year == doc.year
        assert chunk.venue == doc.venue


def test_paragraphs_exceeding_chunk_size_produce_overlapping_chunks():
    paragraphs = [
        f"Paragraph number {i} contains some extra padding text for length." for i in range(6)
    ]
    section = ParsedSection(
        heading="2. Methodology",
        text="\n\n".join(paragraphs),
        page_start=2,
        page_end=2,
    )
    doc = _doc([section])

    chunk_overlap = 15
    chunks = chunk_document(doc, chunk_size=80, chunk_overlap=chunk_overlap)

    assert len(chunks) > 1
    # chunk_index increases monotonically starting from 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    for prev_chunk, next_chunk in zip(chunks, chunks[1:], strict=False):
        assert prev_chunk.section == "2. Methodology"
        assert next_chunk.section == "2. Methodology"
        tail = prev_chunk.text[-chunk_overlap:]
        head = next_chunk.text[:chunk_overlap]
        assert tail == head


def test_single_oversized_paragraph_splits_on_word_boundaries_only():
    long_paragraph = " ".join(f"word{i}" for i in range(200))
    section = ParsedSection(
        heading="3. Results",
        text=long_paragraph,
        page_start=3,
        page_end=3,
    )
    doc = _doc([section])

    chunks = chunk_document(doc, chunk_size=50, chunk_overlap=0)

    assert len(chunks) > 1

    word_re = re.compile(r"^word\d+$")
    for chunk in chunks:
        assert len(chunk.text) <= 50
        for token in chunk.text.split():
            assert word_re.match(token), f"word split mid-boundary: {token!r}"
        # no leading/trailing whitespace left dangling by the split
        assert chunk.text == chunk.text.strip()


def test_empty_section_is_excluded_and_chunk_index_stays_contiguous():
    doc = _doc(
        [
            ParsedSection(heading="Empty", text="   \n\n  ", page_start=1, page_end=1),
            ParsedSection(heading="Real", text="Some real text here.", page_start=1, page_end=1),
        ]
    )

    chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=0)

    assert len(chunks) == 1
    assert chunks[0].section == "Real"
    assert chunks[0].chunk_index == 0


def test_text_chunks_default_to_chunk_type_text_with_no_image_path():
    doc = _doc(
        [ParsedSection(heading="Abstract", text="Some text.", page_start=1, page_end=1)]
    )

    [chunk] = chunk_document(doc, chunk_size=1000, chunk_overlap=0)

    assert chunk.chunk_type == "text"
    assert chunk.image_path is None


def test_table_produces_a_table_typed_chunk_continuing_the_index_sequence():
    doc = _doc(
        [ParsedSection(heading="Abstract", text="Some text.", page_start=1, page_end=1)],
        tables=[
            ExtractedTable(
                page=2, caption="Table 1: Results.", markdown="|A|B|\n|---|---|\n|1|2|\n"
            )
        ],
    )

    chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=0)

    assert len(chunks) == 2
    table_chunk = chunks[1]
    assert table_chunk.chunk_type == "table"
    assert table_chunk.chunk_index == 1  # continues the text chunk's index sequence
    assert table_chunk.page_start == 2
    assert table_chunk.page_end == 2
    assert table_chunk.section == "Table 1: Results."
    assert "Table 1: Results." in table_chunk.text
    assert "|A|B|" in table_chunk.text
    assert table_chunk.paper_id == doc.paper_id
    assert table_chunk.title == doc.title


def test_table_without_caption_still_produces_a_usable_chunk():
    doc = _doc(
        [],
        tables=[ExtractedTable(page=1, caption=None, markdown="|A|\n|---|\n|1|\n")],
    )

    [chunk] = chunk_document(doc, chunk_size=1000, chunk_overlap=0)

    assert chunk.chunk_type == "table"
    assert chunk.section == "Table (page 1)"
    assert "|A|" in chunk.text


def test_equation_produces_an_equation_typed_chunk_with_context():
    doc = _doc(
        [],
        equations=[
            ExtractedEquation(page=3, text="E = mc^2 (1)", context="Einstein derived E = mc^2 (1).")
        ],
    )

    [chunk] = chunk_document(doc, chunk_size=1000, chunk_overlap=0)

    assert chunk.chunk_type == "equation"
    assert chunk.page_start == 3
    assert chunk.page_end == 3
    assert "E = mc^2 (1)" in chunk.text
    assert "Einstein derived" in chunk.text


def test_figure_produces_a_figure_typed_chunk_with_image_path():
    doc = _doc(
        [],
        figures=[
            ExtractedFigure(page=4, caption="Figure 2: Overview.", image_path="/tmp/fig2.png")
        ],
    )

    [chunk] = chunk_document(doc, chunk_size=1000, chunk_overlap=0)

    assert chunk.chunk_type == "figure"
    assert chunk.image_path == "/tmp/fig2.png"
    assert chunk.page_start == 4
    assert "Figure 2: Overview." in chunk.text


def test_figure_without_caption_uses_placeholder_text():
    doc = _doc([], figures=[ExtractedFigure(page=1, caption=None, image_path="/tmp/f.png")])

    [chunk] = chunk_document(doc, chunk_size=1000, chunk_overlap=0)

    assert "(no caption detected)" in chunk.text
    assert chunk.section == "Figure (page 1)"


def test_tables_equations_figures_all_coexist_with_text_chunks_and_unique_indices():
    doc = _doc(
        [ParsedSection(heading="Abstract", text="Some text.", page_start=1, page_end=1)],
        tables=[ExtractedTable(page=1, caption="Table 1.", markdown="|A|\n|---|\n|1|\n")],
        equations=[ExtractedEquation(page=1, text="x = y (1)", context="x = y (1)")],
        figures=[ExtractedFigure(page=1, caption="Figure 1.", image_path="/tmp/f.png")],
    )

    chunks = chunk_document(doc, chunk_size=1000, chunk_overlap=0)

    assert len(chunks) == 4
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]
    assert [c.chunk_type for c in chunks] == ["text", "table", "equation", "figure"]
