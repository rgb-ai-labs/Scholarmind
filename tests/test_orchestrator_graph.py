from pathlib import Path

from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.orchestrator.graph import build_graph, classify_intent

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        return self.response


class RoutingFakeLLMClient:
    def __init__(self, answer_response: str, verification_response: str) -> None:
        self.answer_response = answer_response
        self.verification_response = verification_response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if "fact-checking" in system_prompt:
            return self.verification_response
        return self.answer_response


def test_graph_final_node_formats_and_verifies_citations(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_graph_formatted_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = RoutingFakeLLMClient(
        answer_response="RAG grounds answers in sources [1].",
        verification_response="yes\nThe passage directly supports this.",
    )
    graph = build_graph(fake_client, settings)

    final_state = graph.invoke({"request": "What does this paper propose?"})

    assert "formatted_answer" in final_state
    formatted = final_state["formatted_answer"]
    assert len(formatted.references) == 1
    assert formatted.references[0].apa
    assert formatted.verification_report.unsupported_count == 0


class VerificationFailingLLMClient:
    def __init__(self, answer_response: str) -> None:
        self.answer_response = answer_response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if "fact-checking" in system_prompt:
            raise RuntimeError("verification service unavailable")
        return self.answer_response


def test_graph_formatting_failure_preserves_answer(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_graph_formatting_fail_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = VerificationFailingLLMClient("RAG grounds answers in sources [1].")
    graph = build_graph(fake_client, settings)

    final_state = graph.invoke({"request": "What does this paper propose?"})

    assert "answer_result" in final_state
    assert "error" not in final_state
    assert final_state["formatting_error"] == "verification service unavailable"
    assert "formatted_answer" not in final_state


def test_classify_intent_ingest_with_prefix():
    assert classify_intent("ingest tests/fixtures/sample_paper.pdf") == (
        "ingest",
        "tests/fixtures/sample_paper.pdf",
    )


def test_classify_intent_ingest_via_pdf_suffix():
    assert classify_intent("tests/fixtures/sample_paper.pdf") == (
        "ingest",
        "tests/fixtures/sample_paper.pdf",
    )


def test_classify_intent_ask():
    assert classify_intent("What does this paper propose?") == (
        "ask",
        "What does this paper propose?",
    )


def test_classify_intent_ingest_alone_falls_through_to_ask():
    assert classify_intent("ingest") == ("ask", "ingest")


def test_graph_ingest_path(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_graph_ingest_chunks",
    )
    fake_client = FakeLLMClient("unused")
    graph = build_graph(fake_client, settings)

    final_state = graph.invoke(
        {"request": f"ingest {FIXTURE_PATH}"}
    )

    assert "ingest_result" in final_state
    assert not final_state.get("answer_result")


def test_graph_ask_path(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_graph_ask_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = FakeLLMClient("RAG grounds answers in sources [1].")
    graph = build_graph(fake_client, settings)

    final_state = graph.invoke({"request": "What does this paper propose?"})

    assert "answer_result" in final_state
    assert "ingest_result" not in final_state


def test_graph_messages_accumulate(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_graph_messages_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = FakeLLMClient("RAG grounds answers in sources [1].")
    graph = build_graph(fake_client, settings)

    final_state = graph.invoke({"request": "What does this paper propose?"})

    assert len(final_state["messages"]) > 1
