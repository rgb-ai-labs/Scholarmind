from pathlib import Path

import pytest

from scholarmind.ingestion.loader import RawDocument, load_path, load_pdf

FIXTURE = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


def test_load_pdf_returns_raw_document_with_two_pages():
    doc = load_pdf(FIXTURE)
    assert isinstance(doc, RawDocument)
    assert doc.source_path == FIXTURE
    assert len(doc.pages) == 2
    assert all(page.strip() for page in doc.pages)
    assert "/Title" in doc.pdf_metadata


def test_load_path_single_pdf_file_returns_list_of_one():
    docs = load_path(FIXTURE)
    assert isinstance(docs, list)
    assert len(docs) == 1
    assert isinstance(docs[0], RawDocument)


def test_load_path_directory_returns_list_of_matching_pdfs(tmp_path):
    import shutil

    shutil.copy(FIXTURE, tmp_path / "sample_paper.pdf")
    (tmp_path / "notes.txt").write_text("not a pdf")

    docs = load_path(tmp_path)
    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0].source_path.name == "sample_paper.pdf"


def test_load_path_directory_with_no_pdfs_returns_empty_list(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    docs = load_path(tmp_path)
    assert docs == []


def test_load_path_nonexistent_path_raises_file_not_found_error(tmp_path):
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError):
        load_path(missing)


def test_load_path_non_pdf_file_raises_value_error(tmp_path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hello")
    with pytest.raises(ValueError):
        load_path(txt_file)


def test_load_pdf_scanned_image_only_pdf_logs_warning_and_returns_empty_pages(tmp_path, caplog, monkeypatch):
    import scholarmind.ingestion.loader as loader_module

    class _FakePage:
        def extract_text(self):
            return "   "

    class _FakeReader:
        def __init__(self, _path):
            self.pages = [_FakePage(), _FakePage()]
            self.metadata = {}

    monkeypatch.setattr(loader_module, "PdfReader", _FakeReader)

    fake_path = tmp_path / "scanned.pdf"
    fake_path.write_bytes(b"%PDF-1.4 fake")

    with caplog.at_level("WARNING", logger="scholarmind.ingestion.loader"):
        doc = loader_module.load_pdf(fake_path)

    assert doc.pages == ["", ""]
    assert any("scanned" in record.message.lower() or "image" in record.message.lower() for record in caplog.records)
