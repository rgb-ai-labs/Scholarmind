from pathlib import Path

from typer.testing import CliRunner

from scholarmind.cli import app
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.orchestrator.run import run

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        return self.response


def test_run_ingest_path(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_run_ingest_chunks",
    )

    result = run(f"ingest {FIXTURE_PATH}", settings=settings)

    assert result.intent == "ingest"
    assert result.ingest_result is not None
    assert result.ingest_result.papers_ingested == 1
    assert result.answer_result is None
    assert result.error is None


def test_run_ask_path(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_run_ask_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = FakeLLMClient("RAG grounds answers in sources [1].")

    result = run("What does this paper propose?", llm_client=fake_client, settings=settings)

    assert result.intent == "ask"
    assert result.answer_result is not None
    assert result.answer_result.sources_found > 0
    assert result.ingest_result is None


def test_run_ingest_path_error_surfaces_cleanly():
    result = run("ingest /nonexistent/path/that/does/not/exist.pdf")

    assert result.error is not None
    assert result.ingest_result is None


def test_cli_chat_ingest_command(monkeypatch):
    monkeypatch.setenv("QDRANT_PATH", ":memory:")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_cli_chat_ingest_chunks")

    runner = CliRunner()
    result = runner.invoke(app, ["chat", f"ingest {FIXTURE_PATH}"])

    assert result.exit_code == 0
    assert "Ingested" in result.stdout
