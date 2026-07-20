from pathlib import Path

from scholarmind.agents.base import AgentResult
from scholarmind.agents.summarization import (
    _MAP_SYSTEM_PROMPT,
    _PAPER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    summarize,
)
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.retrieval.papers import list_papers

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0
        self.last_system_prompt = None
        self.last_user_prompt = None
        self.system_prompts: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.system_prompts.append(system_prompt)
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


def test_summarize_with_paper_id_uses_all_that_papers_chunks_not_a_topic_search(
    tmp_path,
):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_summarization_paper_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)
    paper_id = list_papers(settings)[0].paper_id

    fake_client = FakeLLMClient("Overview: ... [1]")

    # The query text is irrelevant to a completely unrelated topic — since paper_id
    # scoping bypasses topic search entirely, the summary must still be produced from
    # the paper's own chunks.
    result = summarize(
        "an off-topic query that matches nothing",
        fake_client,
        settings,
        paper_id=paper_id,
    )

    assert isinstance(result, AgentResult)
    assert result.sources_found > 0
    assert all(source.paper_id == paper_id for source in result.sources)
    assert fake_client.call_count == 1
    assert fake_client.last_system_prompt == _PAPER_SYSTEM_PROMPT


def test_summarize_with_paper_id_returns_empty_for_unknown_paper(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_summarization_unknown_paper_chunks",
    )

    fake_client = FakeLLMClient("should never be returned")

    result = summarize("anything", fake_client, settings, paper_id="does-not-exist")

    assert result.text == ""
    assert result.sources_found == 0
    assert fake_client.call_count == 0


def test_summarize_with_paper_id_map_reduces_long_papers(tmp_path, monkeypatch):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_summarization_map_reduce_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)
    paper_id = list_papers(settings)[0].paper_id

    # Force the map-reduce path (rather than the single-call path) without needing a
    # huge fixture paper: shrink the batch size to 1 chunk per map call.
    monkeypatch.setattr("scholarmind.agents.summarization._MAP_BATCH_SIZE", 1)

    fake_client = FakeLLMClient("Overview: ... [1]")

    result = summarize("irrelevant", fake_client, settings, paper_id=paper_id)

    chunk_count = result.sources_found
    assert chunk_count > 1  # otherwise this test can't exercise multiple map batches

    # One map call per chunk, plus one reduce call.
    assert fake_client.call_count == chunk_count + 1
    assert fake_client.system_prompts[:-1] == [_MAP_SYSTEM_PROMPT] * chunk_count
    assert fake_client.system_prompts[-1] == _PAPER_SYSTEM_PROMPT
