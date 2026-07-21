from pathlib import Path

import pytest
from typer.testing import CliRunner

from scholarmind.agents.llm_client import OpenRouterClient
from scholarmind.agents.qa import (
    AnswerResult,
    answer_question,
    answer_question_streaming,
    finalize_streamed_answer,
)
from scholarmind.citations.verify import Citation, VerifiedAnswer
from scholarmind.cli import _print_answer_result, app
from scholarmind.config import Settings, get_settings
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.retrieval.papers import list_papers

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"
TIDAL_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper_2.pdf"

_has_llm_key = bool(get_settings().llm_api_key)


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        return self.response


class FakeStreamingLLMClient:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.stream_call_count = 0

    def stream(self, system_prompt: str, user_prompt: str):
        self.stream_call_count += 1
        yield from self.tokens


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
    assert result.answer.citations[0].title == "A Study of Retrieval-Augmented Generation for Scholarly Question Answering"
    assert result.answer.citations[0].authors == ["Ada Lovelace", "Grace Hopper"]
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


def test_answer_question_with_paper_ids_scopes_to_chosen_papers(tmp_path: Path):
    # Regression test for multi-paper scoping in the Ask flow: paper_ids should behave like
    # search()'s own paper_ids param, restricting sources to the chosen subset — here, an
    # on-topic query scoped to just the RAG paper out of a two-paper library.
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_qa_paper_ids_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)
    run_ingestion(TIDAL_FIXTURE_PATH, settings)

    rag_paper_id = next(
        p.paper_id for p in list_papers(settings) if "Retrieval-Augmented Generation" in p.label
    )

    fake_client = FakeLLMClient("RAG grounds answers in sources [1].")
    result = answer_question(
        "What does RAG do?", fake_client, settings, paper_ids=[rag_paper_id]
    )

    assert result.sources_found > 0
    assert result.answer is not None
    assert all(c.paper_id == rag_paper_id for c in result.answer.citations)


def test_answer_question_streaming_with_paper_ids_scopes_to_chosen_papers(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_qa_stream_paper_ids_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)
    run_ingestion(TIDAL_FIXTURE_PATH, settings)

    rag_paper_id = next(
        p.paper_id for p in list_papers(settings) if "Retrieval-Augmented Generation" in p.label
    )

    streaming = answer_question_streaming(
        "What does RAG do?",
        FakeStreamingLLMClient(["irrelevant"]),
        settings,
        paper_ids=[rag_paper_id],
    )

    assert streaming is not None
    assert len(streaming.sources) > 0
    assert all(source.paper_id == rag_paper_id for source in streaming.sources)


def test_answer_question_streaming_returns_none_without_calling_llm_when_no_sources(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_qa_stream_empty_chunks",
    )
    fake_client = FakeStreamingLLMClient(["never used"])

    result = answer_question_streaming("anything", fake_client, settings)

    assert result is None
    assert fake_client.stream_call_count == 0


def test_answer_question_streaming_streams_tokens_then_finalizes_with_citations(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_qa_stream_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    # The tokens together reproduce the same shape answer_question() would verify in one shot.
    fake_client = FakeStreamingLLMClient(["RAG grounds ", "answers in sources ", "[1]."])

    streaming = answer_question_streaming("What does RAG do?", fake_client, settings)

    assert streaming is not None
    assert len(streaming.sources) > 0

    full_text = "".join(streaming.tokens)  # what st.write_stream returns to the caller
    assert full_text == "RAG grounds answers in sources [1]."

    answer_result = finalize_streamed_answer("What does RAG do?", full_text, streaming.sources)

    assert answer_result.answer is not None
    assert answer_result.answer.text == full_text  # replay renders the identical streamed text
    assert answer_result.sources_found == len(streaming.sources)
    assert len(answer_result.answer.citations) == 1
    assert answer_result.answer.citations[0].index == 1
    assert answer_result.answer.invalid_citation_markers == []


def test_streaming_and_non_streaming_paths_produce_equivalent_verified_answers(tmp_path: Path):
    # The streaming split (answer_question_streaming + finalize_streamed_answer) must be
    # equivalent to the one-shot answer_question() for the same model output.
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_qa_stream_equiv_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    text = "RAG grounds answers in retrieved sources [1]."
    one_shot = answer_question("What does RAG do?", FakeLLMClient(text), settings)

    streaming = answer_question_streaming(
        "What does RAG do?", FakeStreamingLLMClient([text]), settings
    )
    assert streaming is not None
    streamed = finalize_streamed_answer(
        "What does RAG do?", "".join(streaming.tokens), streaming.sources
    )

    assert one_shot.answer is not None and streamed.answer is not None
    assert streamed.answer.text == one_shot.answer.text
    assert streamed.sources_found == one_shot.sources_found
    assert [c.index for c in streamed.answer.citations] == [
        c.index for c in one_shot.answer.citations
    ]
    assert streamed.answer.invalid_citation_markers == one_shot.answer.invalid_citation_markers


@pytest.mark.skipif(not _has_llm_key, reason="LLM_API_KEY not configured")
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


def test_print_answer_result_handles_unicode_and_missing_metadata_without_crashing(capsys):
    # Regression test: a real "Attention Is All You Need" answer crashed the CLI with
    # UnicodeEncodeError on Windows (the author "Łukasz Kaiser" — cp1252 can't encode "Ł") and,
    # separately, printed the literal string "None" for a citation with no year/section (common:
    # many PDFs' /Subject isn't a plain 4-digit year), instead of degrading like the web app does.
    citation = Citation(
        index=1,
        paper_id="p1",
        title="Attention is All you Need",
        authors=["Łukasz Kaiser"],
        year=None,
        section=None,
        page_start=1,
        page_end=1,
        text="...",
    )
    answer = VerifiedAnswer(
        text="The Transformer uses self-attention [1].",
        citations=[citation],
        invalid_citation_markers=[],
    )
    result = AnswerResult(question="q", answer=answer, sources_found=1)

    _print_answer_result(result, "q")  # must not raise

    captured = capsys.readouterr()
    assert "Łukasz Kaiser" in captured.out
    assert "(n.d.)" in captured.out
    assert "None" not in captured.out


def test_print_answer_result_falls_back_for_fully_empty_citation_metadata(capsys):
    citation = Citation(
        index=1,
        paper_id="p1",
        title=None,
        authors=[],
        year=None,
        section=None,
        page_start=1,
        page_end=1,
        text="...",
    )
    answer = VerifiedAnswer(text="An answer [1].", citations=[citation], invalid_citation_markers=[])
    result = AnswerResult(question="q", answer=answer, sources_found=1)

    _print_answer_result(result, "q")

    captured = capsys.readouterr()
    assert "Untitled" in captured.out
    assert "Unknown author" in captured.out
    assert "(n.d.)" in captured.out
