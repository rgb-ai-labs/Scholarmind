from pathlib import Path

from typer.testing import CliRunner

from scholarmind.agents.llm_client import OpenRouterClient
from scholarmind.agents.qa import AnswerResult, answer_question
from scholarmind.cli import app
from scholarmind.config import Settings, get_settings
from scholarmind.ingestion.pipeline import run_ingestion

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        return self.response


def test_answer_question_returns_none_when_no_sources_and_never_calls_llm(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_qa_empty_chunks",
    )
    fake_client = FakeLLMClient("should never be returned")

    result = answer_question("anything", fake_client, settings)

    assert isinstance(result, AnswerResult)
    assert result.sources_found == 0
    assert result.answer is None
    assert fake_client.call_count == 0


def test_answer_question_with_valid_citation(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_qa_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = FakeLLMClient("RAG grounds answers in sources [1].")

    result = answer_question("What does RAG do?", fake_client, settings)

    assert result.sources_found > 0
    assert result.answer is not None
    assert result.answer.text == "RAG grounds answers in sources [1]."
    assert len(result.answer.citations) == 1
    assert result.answer.citations[0].index == 1
    assert result.answer.citations[0].title is not None
    assert result.answer.citations[0].authors
    assert result.answer.invalid_citation_markers == []


def test_answer_question_with_out_of_range_citation(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_qa_invalid_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = FakeLLMClient("See [1] and [99].")

    result = answer_question("What does RAG do?", fake_client, settings)

    assert result.sources_found > 0
    assert result.answer is not None
    assert result.answer.invalid_citation_markers == [99]
    assert len(result.answer.citations) == 1
    assert result.answer.citations[0].index == 1


def test_answer_question_real_end_to_end(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_qa_real_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    app_settings = get_settings()
    client = OpenRouterClient(
        api_key=app_settings.llm_api_key,
        base_url=app_settings.llm_base_url,
        model=app_settings.llm_model,
        max_tokens=app_settings.llm_max_tokens,
    )

    result = answer_question(
        "What does this paper propose for grounding LLM answers?", client, settings
    )

    assert result.sources_found > 0
    assert result.answer is not None
    assert result.answer.text.strip() != ""


def test_cli_ask_command_no_sources_found(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("QDRANT_COLLECTION", "test_cli_qa_empty_chunks")

    runner = CliRunner()
    result = runner.invoke(app, ["ask", "anything at all"])

    assert result.exit_code == 0
    assert "No relevant sources found" in result.stdout
