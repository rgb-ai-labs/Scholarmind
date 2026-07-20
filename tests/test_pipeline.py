from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient
from typer.testing import CliRunner

from scholarmind.cli import app
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import IngestResult, run_ingestion
from scholarmind.retrieval.papers import get_paper_chunks, list_papers

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"
MULTIMODAL_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper_multimodal.pdf"


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


def test_run_ingestion_same_pdf_twice_creates_only_one_paper_id(tmp_path: Path):
    # Regression test: the paper picker showed the same paper twice in production because two
    # ingestions of visually-identical content ended up under different paper_ids. Re-ingesting
    # the literal same file must not be one of the ways that happens — content-hash paper_id
    # plus upsert-on-paper_id:chunk_index should keep this idempotent.
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_pipeline_dupe_chunks",
    )

    run_ingestion(FIXTURE_PATH, settings)
    run_ingestion(FIXTURE_PATH, settings)

    papers = list_papers(settings)
    assert len(papers) == 1


def test_run_ingestion_end_to_end_extracts_tables_equations_and_figures(tmp_path: Path):
    # Regression test: the Figures & tables panel showed nothing for a paper that genuinely has
    # a table, a figure, and an equation, because the only prior end-to-end ingestion test used a
    # text-only fixture and never asserted on chunk_type — a real extraction gap would have
    # passed silently.
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_pipeline_multimodal_chunks",
    )

    result = run_ingestion(MULTIMODAL_FIXTURE_PATH, settings)
    assert result.papers_ingested == 1

    [paper] = list_papers(settings)
    chunks = get_paper_chunks(paper.paper_id, settings)
    counts = Counter(chunk.chunk_type for chunk in chunks)

    assert counts["table"] >= 1
    assert counts["figure"] >= 1
    assert counts["equation"] >= 1
    assert counts["text"] >= 1

    [figure_chunk] = [c for c in chunks if c.chunk_type == "figure"]
    assert figure_chunk.image_path is not None
    assert Path(figure_chunk.image_path).is_file()


def test_cli_ingest_command_prints_summary(monkeypatch):
    monkeypatch.setenv("QDRANT_PATH", ":memory:")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_cli_pipeline_chunks")

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", str(FIXTURE_PATH)])

    assert result.exit_code == 0
    assert "Ingested" in result.stdout
    assert "test_cli_pipeline_chunks" in result.stdout
