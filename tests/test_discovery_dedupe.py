from scholarmind.discovery import dedupe as dedupe_module
from scholarmind.discovery.dedupe import mark_already_ingested, merge_duplicate_candidates
from scholarmind.discovery.models import Candidate
from scholarmind.retrieval.papers import PaperSummary


def _candidate(**overrides) -> Candidate:
    defaults = dict(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        year=2017,
        venue=None,
        abstract=None,
        doi=None,
        url=None,
        pdf_url=None,
        source="arxiv",
        external_id="1706.03762",
    )
    defaults.update(overrides)
    return Candidate(**defaults)


def test_merge_duplicate_candidates_collapses_same_doi():
    a = _candidate(source="arxiv", doi="10.1234/abc", abstract=None)
    b = _candidate(source="semantic_scholar", doi="10.1234/abc", abstract="A summary.")

    merged = merge_duplicate_candidates([a, b])

    assert len(merged) == 1
    assert merged[0].abstract == "A summary."
    assert set(merged[0].source.split("+")) == {"arxiv", "semantic_scholar"}


def test_merge_duplicate_candidates_collapses_same_title_when_no_doi():
    a = _candidate(source="arxiv", title="Attention Is All You Need", pdf_url="http://x/pdf")
    b = _candidate(source="openalex", title="attention is all you need!!", pdf_url=None)

    merged = merge_duplicate_candidates([a, b])

    assert len(merged) == 1
    assert merged[0].pdf_url == "http://x/pdf"


def test_merge_duplicate_candidates_keeps_distinct_papers_separate():
    a = _candidate(title="Paper One", doi="10.1/one")
    b = _candidate(title="Paper Two", doi="10.1/two")

    merged = merge_duplicate_candidates([a, b])

    assert len(merged) == 2


def test_merge_duplicate_candidates_keeps_unkeyed_candidates_separate():
    a = _candidate(title=None, doi=None, external_id="1")
    b = _candidate(title=None, doi=None, external_id="2")

    merged = merge_duplicate_candidates([a, b])

    assert len(merged) == 2


def test_mark_already_ingested_flags_by_doi(monkeypatch):
    monkeypatch.setattr(
        dedupe_module,
        "list_papers",
        lambda settings=None: [
            PaperSummary(
                paper_id="p1", label="Some Paper", chunk_count=3, ingested_at=1.0, doi="10.1/xyz"
            )
        ],
    )

    candidate = _candidate(doi="10.1/XYZ", title="A different title")
    [result] = mark_already_ingested([candidate])

    assert result.already_ingested is True


def test_mark_already_ingested_flags_by_title_when_no_doi(monkeypatch):
    monkeypatch.setattr(
        dedupe_module,
        "list_papers",
        lambda settings=None: [
            PaperSummary(
                paper_id="p1",
                label="Attention Is All You Need",
                chunk_count=3,
                ingested_at=1.0,
                doi=None,
            )
        ],
    )

    candidate = _candidate(doi=None, title="attention is ALL you need")
    [result] = mark_already_ingested([candidate])

    assert result.already_ingested is True


def test_mark_already_ingested_leaves_new_papers_unflagged(monkeypatch):
    monkeypatch.setattr(dedupe_module, "list_papers", lambda settings=None: [])

    candidate = _candidate()
    [result] = mark_already_ingested([candidate])

    assert result.already_ingested is False
