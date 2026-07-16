from pathlib import Path

from scholarmind.agents.methodology import SYSTEM_PROMPT, extract_methodology
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.response


def test_extract_methodology_returns_grounded_comparison(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_methodology_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = FakeLLMClient(
        "The paper chunks and embeds documents, then retrieves and reranks."
    )

    result = extract_methodology("retrieval augmented generation", fake_client, settings)

    assert result.text == "The paper chunks and embeds documents, then retrieves and reranks."
    assert result.sources_found > 0
    assert len(result.sources) == result.sources_found
    assert fake_client.call_count == 1
    assert fake_client.last_system_prompt == SYSTEM_PROMPT
    assert fake_client.last_user_prompt is not None
    assert "Sources:" in fake_client.last_user_prompt


def test_extract_methodology_refuses_when_no_sources(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant_empty"),
        qdrant_collection="test_methodology_empty_chunks",
    )
    fake_client = FakeLLMClient("should never be returned")

    result = extract_methodology("anything", fake_client, settings)

    assert result.text == ""
    assert result.sources_found == 0
    assert fake_client.call_count == 0
