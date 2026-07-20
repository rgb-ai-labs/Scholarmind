from pathlib import Path

from scholarmind.agents import novelty as novelty_module
from scholarmind.agents.novelty import SYSTEM_PROMPT, NoveltyCheckResult, check_novelty
from scholarmind.config import Settings
from scholarmind.discovery.models import Candidate
from scholarmind.discovery.service import DiscoveryResult
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


def _candidate(**overrides) -> Candidate:
    defaults = dict(
        title="An External Paper",
        authors=["Someone Else"],
        year=2021,
        venue="A Venue",
        abstract="An external abstract.",
        doi=None,
        url=None,
        pdf_url=None,
        source="arxiv",
        external_id="ext-1",
    )
    defaults.update(overrides)
    return Candidate(**defaults)


def test_check_novelty_returns_library_overlaps_and_assessment(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_novelty_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    fake_client = FakeLLMClient("This overlaps with [1]; the novel angle is X; consider Y.")

    result = check_novelty("A passage about retrieval augmented generation.", fake_client, settings)

    assert isinstance(result, NoveltyCheckResult)
    assert result.sources_found > 0
    assert len(result.library_overlaps) == result.sources_found
    assert result.assessment == "This overlaps with [1]; the novel angle is X; consider Y."
    assert result.external_overlaps == []
    assert fake_client.call_count == 1
    assert fake_client.last_system_prompt == SYSTEM_PROMPT
    assert "Passage:" in fake_client.last_user_prompt


def test_check_novelty_no_matches_returns_empty_without_calling_llm(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant_empty"),
        qdrant_collection="test_novelty_empty_chunks",
    )
    fake_client = FakeLLMClient("should never be returned")

    result = check_novelty("anything at all", fake_client, settings)

    assert result.library_overlaps == []
    assert result.sources_found == 0
    assert result.assessment == ""
    assert fake_client.call_count == 0


def test_check_novelty_external_search_disabled_by_default(tmp_path: Path, monkeypatch):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_novelty_no_external_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    def _fail(*args, **kwargs):
        raise AssertionError("search_external should not be called by default")

    monkeypatch.setattr(novelty_module, "search_external", _fail)

    fake_client = FakeLLMClient("assessment")
    result = check_novelty("retrieval augmented generation", fake_client, settings)

    assert result.external_overlaps == []


def test_check_novelty_with_external_search_includes_candidates_in_prompt(
    tmp_path: Path, monkeypatch
):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_novelty_external_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)

    monkeypatch.setattr(
        novelty_module,
        "search_external",
        lambda query, settings: DiscoveryResult(candidates=[_candidate()], errors=["arxiv: down"]),
    )

    fake_client = FakeLLMClient("assessment")
    result = check_novelty(
        "retrieval augmented generation", fake_client, settings, include_external_search=True
    )

    assert len(result.external_overlaps) == 1
    assert result.external_search_errors == ["arxiv: down"]
    assert "[external] An External Paper" in fake_client.last_user_prompt


def test_check_novelty_uses_only_external_overlaps_when_library_is_empty(monkeypatch, tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant_empty"),
        qdrant_collection="test_novelty_external_only_chunks",
    )

    monkeypatch.setattr(
        novelty_module,
        "search_external",
        lambda query, settings: DiscoveryResult(candidates=[_candidate()], errors=[]),
    )

    fake_client = FakeLLMClient("assessment")
    result = check_novelty("anything", fake_client, settings, include_external_search=True)

    assert result.library_overlaps == []
    assert len(result.external_overlaps) == 1
    assert fake_client.call_count == 1
    assert result.assessment == "assessment"
