import httpx

from scholarmind.config import Settings
from scholarmind.discovery import ingest as ingest_module
from scholarmind.discovery.ingest import ingest_candidate
from scholarmind.discovery.models import Candidate
from scholarmind.ingestion.pipeline import IngestResult


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)


def _candidate(**overrides) -> Candidate:
    defaults = dict(
        title="A Paper",
        authors=["Someone"],
        year=2020,
        venue="A Venue",
        abstract="An abstract.",
        doi="10.1/xyz",
        url="https://example.org/paper",
        pdf_url=None,
        source="arxiv",
        external_id="1234.5678",
    )
    defaults.update(overrides)
    return Candidate(**defaults)


def test_ingest_candidate_with_valid_pdf_downloads_and_ingests(tmp_path, monkeypatch):
    settings = Settings(qdrant_path=str(tmp_path / "qdrant"))

    monkeypatch.setattr(
        ingest_module.httpx,
        "get",
        lambda *a, **kw: _FakeResponse(b"%PDF-1.4 fake pdf bytes"),
    )

    captured_path = {}

    def _fake_run_ingestion(path, settings_arg):
        captured_path["path"] = path
        return IngestResult(papers_ingested=1, chunks_created=3, collection_name="x")

    monkeypatch.setattr(ingest_module, "run_ingestion", _fake_run_ingestion)

    candidate = _candidate(pdf_url="https://example.org/paper.pdf")
    result = ingest_candidate(candidate, settings)

    assert result.papers_ingested == 1
    assert captured_path["path"].suffix == ".pdf"
    assert captured_path["path"].read_bytes().startswith(b"%PDF")


def test_ingest_candidate_falls_back_to_metadata_record_when_pdf_download_fails(
    tmp_path, monkeypatch
):
    settings = Settings(qdrant_path=str(tmp_path / "qdrant"))

    def _raise(*args, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(ingest_module.httpx, "get", _raise)

    captured = {}

    def _fake_ingest_metadata_record(**kwargs):
        captured.update(kwargs)
        return IngestResult(papers_ingested=1, chunks_created=1, collection_name="x")

    monkeypatch.setattr(ingest_module, "ingest_metadata_record", _fake_ingest_metadata_record)

    candidate = _candidate(pdf_url="https://example.org/paper.pdf")
    result = ingest_candidate(candidate, settings)

    assert result.papers_ingested == 1
    assert captured["title"] == "A Paper"
    assert captured["doi"] == "10.1/xyz"
    assert captured["text"] == "An abstract."


def test_ingest_candidate_falls_back_when_response_is_not_actually_a_pdf(tmp_path, monkeypatch):
    settings = Settings(qdrant_path=str(tmp_path / "qdrant"))

    monkeypatch.setattr(
        ingest_module.httpx, "get", lambda *a, **kw: _FakeResponse(b"<html>not a pdf</html>")
    )

    captured = {}
    monkeypatch.setattr(
        ingest_module,
        "ingest_metadata_record",
        lambda **kwargs: captured.update(kwargs)
        or IngestResult(papers_ingested=1, chunks_created=1, collection_name="x"),
    )

    candidate = _candidate(pdf_url="https://example.org/landing-page")
    ingest_candidate(candidate, settings)

    assert captured["title"] == "A Paper"


def test_ingest_candidate_without_pdf_url_ingests_metadata_record_directly(tmp_path, monkeypatch):
    settings = Settings(qdrant_path=str(tmp_path / "qdrant"))

    captured = {}
    monkeypatch.setattr(
        ingest_module,
        "ingest_metadata_record",
        lambda **kwargs: captured.update(kwargs)
        or IngestResult(papers_ingested=1, chunks_created=1, collection_name="x"),
    )

    candidate = _candidate(pdf_url=None, abstract=None)
    ingest_candidate(candidate, settings)

    assert captured["text"] == "(no abstract available from this source)"
    assert captured["source_filename"] == "arxiv:1234.5678"


def test_ingest_candidate_paper_id_is_stable_for_same_identity():
    a = _candidate(doi="10.1/same")
    b = _candidate(doi="10.1/same", title="Different title entirely")

    assert ingest_module._candidate_paper_id(a) == ingest_module._candidate_paper_id(b)


def test_ingest_candidate_metadata_only_record_is_searchable_end_to_end(tmp_path, monkeypatch):
    # Real (not mocked) ingest_metadata_record + embedding pipeline, to prove a candidate
    # with no open PDF actually becomes a retrievable, labeled library entry.
    settings = Settings(qdrant_path=str(tmp_path / "qdrant"))

    def _raise(*args, **kwargs):
        raise httpx.ConnectError("no pdf host")

    monkeypatch.setattr(ingest_module.httpx, "get", _raise)

    candidate = _candidate(
        title="Transformer Circuits for Interpretability",
        abstract="We study transformer internals via circuit analysis.",
        doi="10.9999/circuits",
        pdf_url="https://example.org/unreachable.pdf",
    )

    result = ingest_candidate(candidate, settings)

    assert result.papers_ingested == 1
    assert result.chunks_created >= 1

    from scholarmind.retrieval.papers import list_papers

    [paper] = list_papers(settings)
    assert paper.label == "Transformer Circuits for Interpretability"
    assert paper.doi == "10.9999/circuits"
    assert paper.is_metadata_only is True
