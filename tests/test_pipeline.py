from pathlib import Path

from typer.testing import CliRunner

from scholarmind.cli import app
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import IngestResult, run_ingestion

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


def _settings(collection_name: str) -> Settings:
    return Settings(qdrant_path=":memory:", qdrant_collection=collection_name)


def test_run_ingestion_end_to_end_and_idempotent():
    settings = _settings("test_pipeline_chunks")

    result = run_ingestion(FIXTURE_PATH, settings)

    assert isinstance(result, IngestResult)
    assert result.papers_ingested == 1
    assert result.chunks_created > 0
    assert result.collection_name == "test_pipeline_chunks"

    second_result = run_ingestion(FIXTURE_PATH, settings)

    assert second_result.papers_ingested == 1
    assert second_result.chunks_created == result.chunks_created


def test_cli_ingest_command_prints_summary(monkeypatch):
    monkeypatch.setenv("QDRANT_PATH", ":memory:")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_cli_pipeline_chunks")

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", str(FIXTURE_PATH)])

    assert result.exit_code == 0
    assert "Ingested" in result.stdout
    assert "test_cli_pipeline_chunks" in result.stdout
