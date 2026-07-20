from pathlib import Path

from typer.testing import CliRunner

from scholarmind.cli import app
from scholarmind.config import Settings
from scholarmind.ingestion.dedupe import delete_papers, find_duplicate_paper_groups
from scholarmind.ingestion.pipeline import ingest_metadata_record, run_ingestion
from scholarmind.retrieval.papers import list_papers

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"
RAG_TITLE = "A Study of Retrieval-Augmented Generation for Scholarly Question Answering"


def _settings(tmp_path: Path, name: str) -> Settings:
    return Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection=name)


def test_find_duplicate_paper_groups_finds_same_title_different_paper_id(tmp_path: Path):
    settings = _settings(tmp_path, "test_dedupe_find_chunks")

    run_ingestion(FIXTURE_PATH, settings)
    ingest_metadata_record(
        paper_id="a-different-hash-same-title",
        title=RAG_TITLE,
        authors=["Someone Else"],
        year=2023,
        venue=None,
        text="A shorter, unrelated abstract that happens to share the same title.",
        settings=settings,
    )

    groups = find_duplicate_paper_groups(settings)

    assert len(groups) == 1
    group = groups[0]
    assert group.title == RAG_TITLE
    # The real PDF has more chunks than the one-paragraph metadata-only record, so it's kept.
    assert group.keep.paper_id != "a-different-hash-same-title"
    assert [p.paper_id for p in group.remove] == ["a-different-hash-same-title"]


def test_find_duplicate_paper_groups_ignores_distinct_titles(tmp_path: Path):
    settings = _settings(tmp_path, "test_dedupe_distinct_chunks")

    run_ingestion(FIXTURE_PATH, settings)

    assert find_duplicate_paper_groups(settings) == []


def test_delete_papers_removes_only_the_targeted_paper_ids(tmp_path: Path):
    settings = _settings(tmp_path, "test_dedupe_delete_chunks")

    run_ingestion(FIXTURE_PATH, settings)
    ingest_metadata_record(
        paper_id="a-different-hash-same-title",
        title=RAG_TITLE,
        authors=[],
        year=None,
        venue=None,
        text="Duplicate metadata-only record.",
        settings=settings,
    )

    [group] = find_duplicate_paper_groups(settings)
    deleted = delete_papers([p.paper_id for p in group.remove], settings)

    assert deleted > 0
    papers = list_papers(settings)
    assert len(papers) == 1
    assert papers[0].paper_id == group.keep.paper_id


def test_delete_papers_with_no_ids_is_a_noop(tmp_path: Path):
    settings = _settings(tmp_path, "test_dedupe_noop_chunks")
    run_ingestion(FIXTURE_PATH, settings)

    assert delete_papers([], settings) == 0
    assert len(list_papers(settings)) == 1


def test_run_ingestion_flags_duplicate_title_from_a_byte_different_pdf(tmp_path: Path):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    settings = _settings(tmp_path, "test_dedupe_warning_chunks")
    run_ingestion(FIXTURE_PATH, settings)

    # A second, byte-different PDF with the exact same title as the fixture just ingested —
    # simulates a user re-downloading/regenerating "the same paper" as a different file.
    styles = getSampleStyleSheet()
    duplicate_pdf = tmp_path / "duplicate.pdf"
    doc = SimpleDocTemplate(str(duplicate_pdf), pagesize=letter, title=RAG_TITLE)
    doc.build([Paragraph(RAG_TITLE, styles["Title"]), Paragraph("Different body text.", styles["Normal"])])

    result = run_ingestion(duplicate_pdf, settings)

    assert result.duplicate_title_warnings == [RAG_TITLE]
    assert len(list_papers(settings)) == 2  # still ingested — the warning is advisory, not a block


def test_cli_dedupe_dry_run_lists_without_deleting(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "test_dedupe_cli_dry_chunks")
    run_ingestion(FIXTURE_PATH, settings)
    ingest_metadata_record(
        paper_id="a-different-hash-same-title",
        title=RAG_TITLE,
        authors=[],
        year=None,
        venue=None,
        text="Duplicate metadata-only record.",
        settings=settings,
    )

    monkeypatch.setenv("QDRANT_PATH", settings.qdrant_path)
    monkeypatch.setenv("QDRANT_COLLECTION", settings.qdrant_collection)

    runner = CliRunner()
    result = runner.invoke(app, ["dedupe"])

    assert result.exit_code == 0
    assert "Dry run" in result.stdout
    assert len(list_papers(settings)) == 2  # nothing deleted


def test_cli_dedupe_apply_removes_duplicates(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "test_dedupe_cli_apply_chunks")
    run_ingestion(FIXTURE_PATH, settings)
    ingest_metadata_record(
        paper_id="a-different-hash-same-title",
        title=RAG_TITLE,
        authors=[],
        year=None,
        venue=None,
        text="Duplicate metadata-only record.",
        settings=settings,
    )

    monkeypatch.setenv("QDRANT_PATH", settings.qdrant_path)
    monkeypatch.setenv("QDRANT_COLLECTION", settings.qdrant_collection)

    runner = CliRunner()
    result = runner.invoke(app, ["dedupe", "--apply"])

    assert result.exit_code == 0
    assert "Removed" in result.stdout
    assert len(list_papers(settings)) == 1
