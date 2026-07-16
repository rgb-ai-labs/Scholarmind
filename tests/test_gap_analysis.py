from pathlib import Path

from scholarmind.agents.gap_analysis import SYSTEM_PROMPT, analyze_gaps
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


def test_analyze_gaps_returns_grounded_result(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "q"),
        qdrant_collection="test_gap_analysis_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    client = FakeLLMClient("- Limited evaluation on long documents.")
    result = analyze_gaps("retrieval augmented generation", client, settings)

    assert result.text == "- Limited evaluation on long documents."
    assert result.sources_found > 0
    assert len(result.sources) == result.sources_found
    assert client.call_count == 1
    assert client.last_system_prompt == SYSTEM_PROMPT
    assert "Sources:" in client.last_user_prompt


def test_analyze_gaps_refuses_without_sources(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "q_empty"),
        qdrant_collection="test_gap_analysis_empty_chunks",
    )

    client = FakeLLMClient("should not be called")
    result = analyze_gaps("anything", client, settings=settings)

    assert result.text == ""
    assert result.sources == []
    assert result.sources_found == 0
    assert client.call_count == 0
