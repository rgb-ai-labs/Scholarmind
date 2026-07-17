import hashlib
from pathlib import Path

from scholarmind.ingestion.loader import RawDocument, load_pdf
from scholarmind.ingestion.parser import ParsedDocument, parse_document

FIXTURE = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


def _parsed_fixture() -> ParsedDocument:
    raw = load_pdf(FIXTURE)
    return parse_document(raw)


def test_parse_document_paper_id_is_sha256_of_source_bytes():
    parsed = _parsed_fixture()
    expected = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert parsed.paper_id == expected


def test_parse_document_extracts_title_authors_year():
    parsed = _parsed_fixture()
    assert parsed.title == "A Study of Retrieval-Augmented Generation for Scholarly Question Answering"
    assert parsed.authors == ["Ada Lovelace", "Grace Hopper"]
    assert parsed.year == 2024
    assert parsed.venue is None


def test_parse_document_detects_sections_in_order():
    parsed = _parsed_fixture()
    headings = [s.heading for s in parsed.sections]
    assert "Abstract" in headings
    assert "1. Introduction" in headings
    assert "2. Methodology" in headings
    assert "3. Conclusion" in headings

    expected_order = ["Abstract", "1. Introduction", "2. Methodology", "3. Conclusion"]
    filtered = [h for h in headings if h in expected_order]
    assert filtered == expected_order


def test_parse_document_methodology_and_conclusion_are_on_page_two():
    parsed = _parsed_fixture()
    by_heading = {s.heading: s for s in parsed.sections}

    methodology = by_heading["2. Methodology"]
    assert methodology.page_start == 2
    assert methodology.page_end == 2

    conclusion = by_heading["3. Conclusion"]
    assert conclusion.page_start == 2
    assert conclusion.page_end == 2


def test_parse_document_abstract_section_text_is_nonempty_and_clean():
    parsed = _parsed_fixture()
    by_heading = {s.heading: s for s in parsed.sections}
    abstract = by_heading["Abstract"]
    assert abstract.page_start == 1
    assert "retrieval-augmented generation" in abstract.text
    assert "  " not in abstract.text  # no double spaces from raw line joins


def test_parse_document_multiple_authors_split_on_comma():
    raw = RawDocument(
        source_path=FIXTURE,
        pages=["Abstract\nsome text"],
        pdf_metadata={"/Author": "Alice Smith; Bob Jones"},
    )
    parsed = parse_document(raw)
    assert parsed.authors == ["Alice Smith", "Bob Jones"]


def test_parse_document_missing_metadata_yields_none_values():
    raw = RawDocument(source_path=FIXTURE, pages=["Abstract\nsome text"], pdf_metadata={})
    parsed = parse_document(raw)
    assert parsed.title is None
    assert parsed.authors == []
    assert parsed.year is None
    assert parsed.venue is None


def test_parse_document_subject_non_year_does_not_set_year():
    raw = RawDocument(
        source_path=FIXTURE,
        pages=["Abstract\nsome text"],
        pdf_metadata={"/Subject": "Machine Learning"},
    )
    parsed = parse_document(raw)
    assert parsed.year is None
