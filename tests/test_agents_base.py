from pathlib import Path

from scholarmind.agents.base import AgentResult, grounded_generate
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


class CapturingFakeLLMClient:
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


def test_grounded_generate_uses_sources_and_prompts(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_agents_base_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    client = CapturingFakeLLMClient("A grounded summary.")
    result = grounded_generate(
        "What is retrieval-augmented generation?",
        "SYSTEM",
        "Do the task.",
        client,
        settings,
    )

    assert isinstance(result, AgentResult)
    assert result.text == "A grounded summary."
    assert result.sources_found > 0
    assert len(result.sources) == result.sources_found
    assert client.call_count == 1
    assert client.last_system_prompt == "SYSTEM"
    assert "Do the task." in client.last_user_prompt
    assert "Sources:" in client.last_user_prompt


def test_grounded_generate_refuses_without_sources(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_agents_base_empty_chunks",
    )

    client = CapturingFakeLLMClient("should not be called")
    result = grounded_generate(
        "anything at all",
        "SYSTEM",
        "Do the task.",
        client,
        settings,
    )

    assert result.text == ""
    assert result.sources == []
    assert result.sources_found == 0
    assert client.call_count == 0
