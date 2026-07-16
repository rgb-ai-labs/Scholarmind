from scholarmind.citations.verify import (
    build_source_block,
    extract_citation_markers,
    verify_citations,
)
from scholarmind.retrieval.dense import DenseResult


def _make_result(
    paper_id: str = "paper-1",
    title: str | None = "Some Paper",
    authors: list[str] | None = None,
    year: int | None = 2024,
    venue: str | None = "Some Venue",
    section: str | None = "Abstract",
    page_start: int = 1,
    page_end: int = 2,
    chunk_index: int = 0,
    text: str = "This is the chunk text.",
    score: float = 0.9,
) -> DenseResult:
    return DenseResult(
        paper_id=paper_id,
        title=title,
        authors=authors if authors is not None else ["Ada Lovelace", "Grace Hopper"],
        year=year,
        venue=venue,
        section=section,
        page_start=page_start,
        page_end=page_end,
        chunk_index=chunk_index,
        text=text,
        score=score,
    )


def test_build_source_block_with_two_sources():
    sources = [
        _make_result(paper_id="p1", authors=["Ada Lovelace", "Grace Hopper"], year=2024, text="First chunk text."),
        _make_result(paper_id="p2", authors=["Alan Turing"], year=2020, text="Second chunk text."),
    ]

    block = build_source_block(sources)

    assert "[1]" in block
    assert "[2]" in block
    assert "Ada Lovelace, Grace Hopper" in block
    assert "Alan Turing" in block
    assert "First chunk text." in block
    assert "Second chunk text." in block


def test_build_source_block_omits_none_year_and_empty_authors():
    sources = [_make_result(authors=[], year=None, text="Chunk with no authors or year.")]

    block = build_source_block(sources)

    assert "None" not in block
    assert "()" not in block
    assert "Chunk with no authors or year." in block


def test_extract_citation_markers_dedupes_first_seen_order():
    markers = extract_citation_markers("... [1] ... [2] ... [1] again")

    assert markers == [1, 2]


def test_verify_citations_with_valid_markers():
    sources = [
        _make_result(paper_id="p1"),
        _make_result(paper_id="p2"),
        _make_result(paper_id="p3"),
    ]

    result = verify_citations("Claim one [1]. Claim three [3].", sources)

    assert len(result.citations) == 2
    assert result.citations[0].paper_id == "p1"
    assert result.citations[0].index == 1
    assert result.citations[1].paper_id == "p3"
    assert result.citations[1].index == 3
    assert result.invalid_citation_markers == []
    assert result.text == "Claim one [1]. Claim three [3]."


def test_verify_citations_with_out_of_range_marker():
    sources = [_make_result(paper_id="p1"), _make_result(paper_id="p2")]

    result = verify_citations("Claim one [1]. Bad claim [5].", sources)

    assert len(result.citations) == 1
    assert result.citations[0].paper_id == "p1"
    assert result.invalid_citation_markers == [5]


def test_verify_citations_with_no_markers():
    sources = [_make_result(paper_id="p1")]

    result = verify_citations("No citations here at all.", sources)

    assert result.citations == []
    assert result.invalid_citation_markers == []
