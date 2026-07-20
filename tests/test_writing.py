from pathlib import Path

import pytest

from scholarmind.agents.base import AgentResult
from scholarmind.agents.writing import SECTION_TYPES, SYSTEM_PROMPT, draft_section
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.retrieval.papers import list_papers

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"
TIDAL_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper_2.pdf"


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


def test_draft_section_strips_sentences_without_citation_markers(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_writing_strip_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = FakeLLMClient(
        "RAG grounds outputs in retrieved passages [1]. "
        "This sentence has no citation and must be dropped. "
        "A second grounded claim follows [1]."
    )

    result = draft_section("retrieval augmented generation", fake_client, settings)

    assert "no citation and must be dropped" not in result.text
    assert "grounds outputs in retrieved passages [1]." in result.text
    assert "A second grounded claim follows [1]." in result.text


def test_draft_section_all_uncited_produces_empty_text(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_writing_all_uncited_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = FakeLLMClient("This sentence cites nothing at all.")

    result = draft_section("retrieval augmented generation", fake_client, settings)

    assert result.text == ""
    assert result.sources_found > 0  # sources WERE retrieved; everything was just uncited


@pytest.mark.parametrize("section_type", SECTION_TYPES)
def test_draft_section_each_named_section_type_uses_a_distinct_prompt(
    tmp_path: Path, section_type: str
):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection=f"test_writing_section_{section_type}_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = FakeLLMClient("Grounded claim [1].")

    result = draft_section(
        "retrieval augmented generation", fake_client, settings, section_type=section_type
    )

    assert result.sources_found > 0
    assert fake_client.last_system_prompt != SYSTEM_PROMPT


def test_draft_section_unknown_section_type_raises(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_writing_unknown_section_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = FakeLLMClient("should never be returned")

    with pytest.raises(ValueError, match="Unknown section type"):
        draft_section(
            "retrieval augmented generation", fake_client, settings, section_type="haiku"
        )

    assert fake_client.call_count == 0


def test_draft_section_voice_notes_are_folded_into_the_prompt(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_writing_voice_notes_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = FakeLLMClient("Grounded claim [1].")

    draft_section(
        "retrieval augmented generation",
        fake_client,
        settings,
        voice_notes="keep it formal and concise",
    )

    assert "keep it formal and concise" in fake_client.last_user_prompt


def test_draft_section_scoped_to_paper_ids_only_retrieves_those_papers(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_writing_scoped_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)
    run_ingestion(TIDAL_FIXTURE_PATH, settings)

    rag_paper_id = next(
        p.paper_id for p in list_papers(settings) if "Retrieval-Augmented Generation" in p.label
    )

    fake_client = FakeLLMClient("Grounded claim [1].")

    # Off-topic query relative to the scoped paper: proves the paper_ids filter — not the
    # query — decides which papers are eligible, same regression shape as Summarize/Ask.
    result = draft_section(
        "tidal energy coastal turbines",
        fake_client,
        settings,
        paper_ids=[rag_paper_id],
    )

    assert all(source.paper_id == rag_paper_id for source in result.sources)
