from pathlib import Path

from scholarmind.agents.base import AgentResult
from scholarmind.agents.summarization import SYSTEM_PROMPT, summarize
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0
        self.last_system_prompt = None
        self.last_user_prompt = None

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.response


def test_summarize_with_sources(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_summarization_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = FakeLLMClient("RAG grounds generation in retrieved passages.")

    result = summarize("What is retrieval-augmented generation?", fake_client, settings)

    assert isinstance(result, AgentResult)
    assert result.text == "RAG grounds generation in retrieved passages."
    assert result.sources_found > 0
    assert len(result.sources) == result.sources_found
    assert fake_client.call_count == 1
    assert fake_client.last_system_prompt == SYSTEM_PROMPT
    assert "Sources:" in fake_client.last_user_prompt


def test_summarize_refuses_without_sources(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_summarization_empty_chunks",
    )

    fake_client = FakeLLMClient("should never be returned")

    result = summarize("anything", fake_client, settings=settings)

    assert result.text == ""
    assert result.sources_found == 0
    assert fake_client.call_count == 0
