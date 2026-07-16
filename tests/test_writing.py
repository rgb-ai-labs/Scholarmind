from pathlib import Path

from scholarmind.agents.base import AgentResult
from scholarmind.agents.writing import SYSTEM_PROMPT, draft_section
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0
        self.last_system_prompt = ""
        self.last_user_prompt = ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.response


def test_draft_section_returns_grounded_draft_with_sources(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_writing_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = FakeLLMClient(
        "Retrieval-augmented generation grounds outputs in retrieved passages [1]."
    )

    result = draft_section("retrieval augmented generation", fake_client, settings)

    assert isinstance(result, AgentResult)
    assert result.text == "Retrieval-augmented generation grounds outputs in retrieved passages [1]."
    assert result.sources_found > 0
    assert len(result.sources) == result.sources_found
    assert fake_client.call_count == 1
    assert fake_client.last_system_prompt == SYSTEM_PROMPT
    assert "Sources:" in fake_client.last_user_prompt


def test_draft_section_refuses_when_no_sources_and_never_calls_llm(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant_empty"),
        qdrant_collection="test_writing_empty_chunks",
    )
    fake_client = FakeLLMClient("should never be returned")

    result = draft_section("anything", fake_client, settings)

    assert result.text == ""
    assert result.sources_found == 0
    assert fake_client.call_count == 0
