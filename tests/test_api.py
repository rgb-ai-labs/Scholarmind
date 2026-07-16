from pathlib import Path

from fastapi.testclient import TestClient

from scholarmind.api.app import (
    app,
    create_app,
    get_llm_client_dependency,
    get_settings_dependency,
)
from scholarmind.config import Settings

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


class RoutingFakeLLMClient:
    def __init__(self, answer_response: str, verification_response: str) -> None:
        self.answer_response = answer_response
        self.verification_response = verification_response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if "fact-checking" in system_prompt:
            return self.verification_response
        return self.answer_response


def test_health_returns_ok():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_with_path(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_api_ingest_chunks",
    )
    api = create_app()
    api.dependency_overrides[get_settings_dependency] = lambda: settings

    client = TestClient(api)
    response = client.post("/ingest", data={"path": str(FIXTURE_PATH)})

    assert response.status_code == 200
    body = response.json()
    assert body["papers_ingested"] == 1
    assert body["chunks_created"] > 0
    assert body["collection_name"] == "test_api_ingest_chunks"


def test_ingest_without_path_or_file_returns_400():
    client = TestClient(app)

    response = client.post("/ingest", data={})

    assert response.status_code == 400


def test_ask_returns_expected_schema_with_mocked_llm(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_api_ask_chunks",
    )
    from scholarmind.ingestion.pipeline import run_ingestion

    run_ingestion(FIXTURE_PATH, settings)

    fake_client = RoutingFakeLLMClient(
        answer_response="RAG grounds answers in retrieved passages [1].",
        verification_response="yes\nThe passage directly supports this.",
    )

    api = create_app()
    api.dependency_overrides[get_settings_dependency] = lambda: settings
    api.dependency_overrides[get_llm_client_dependency] = lambda: fake_client

    client = TestClient(api)
    response = client.post("/ask", json={"question": "What does this paper propose?"})

    assert response.status_code == 200
    body = response.json()

    assert body["intent"] == "ask"
    assert body["answer"] == "RAG grounds answers in retrieved passages [1]."
    assert body["sources_found"] > 0
    assert len(body["sources"]) >= 1
    source = body["sources"][0]
    assert set(source.keys()) == {
        "index",
        "paper_id",
        "title",
        "authors",
        "year",
        "section",
        "page_start",
        "page_end",
        "text",
    }
    assert len(body["references"]) >= 1
    reference = body["references"][0]
    assert set(reference.keys()) == {"citation_index", "apa", "bibtex"}
    assert body["verification_report"] is not None
    assert body["verification_report"]["unsupported_count"] == 0
    assert body["error"] is None


def test_ask_no_sources_returns_empty_answer(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_api_ask_empty_chunks",
    )

    fake_client = RoutingFakeLLMClient("unused", "unused")

    api = create_app()
    api.dependency_overrides[get_settings_dependency] = lambda: settings
    api.dependency_overrides[get_llm_client_dependency] = lambda: fake_client

    client = TestClient(api)
    response = client.post("/ask", json={"question": "Anything at all?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is None
    assert body["sources_found"] == 0
    assert body["sources"] == []
    assert body["references"] == []
