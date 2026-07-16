from pathlib import Path

from qdrant_client import QdrantClient
from typer.testing import CliRunner

from scholarmind.cli import app
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import IngestResult, run_ingestion

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


def test_run_ingestion_end_to_end_and_idempotent(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_pipeline_chunks",
    )

    result = run_ingestion(FIXTURE_PATH, settings)

    assert isinstance(result, IngestResult)
    assert result.papers_ingested == 1
    assert result.chunks_created > 0
    assert result.collection_name == "test_pipeline_chunks"

    second_result = run_ingestion(FIXTURE_PATH, settings)

    assert second_result.papers_ingested == 1
    assert second_result.papers_ingested == result.papers_ingested
    assert second_result.chunks_created == result.chunks_created

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    count = client.count("test_pipeline_chunks", exact=True).count
    assert count == result.chunks_created


def test_cli_ingest_command_prints_summary(monkeypatch):
    monkeypatch.setenv("QDRANT_PATH", ":memory:")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_cli_pipeline_chunks")

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", str(FIXTURE_PATH)])

    assert result.exit_code == 0
    assert "Ingested" in result.stdout
    assert "test_cli_pipeline_chunks" in result.stdout
