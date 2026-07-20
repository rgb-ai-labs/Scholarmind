from pathlib import Path

from typer.testing import CliRunner

from scholarmind.cli import app
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import ingest_metadata_record, run_ingestion
from scholarmind.retrieval.papers import (
    delete_papers,
    find_papers_by_identifier,
    list_papers,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"
RAG_TITLE = "A Study of Retrieval-Augmented Generation for Scholarly Question Answering"


def _settings(tmp_path: Path, name: str) -> Settings:
    return Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection=name)


def _ingest_extra_metadata_record(settings: Settings, paper_id: str, title: str) -> None:
    ingest_metadata_record(
        paper_id=paper_id,
        title=title,
        authors=[],
        year=None,
        venue=None,
        text="A short metadata-only abstract.",
        settings=settings,
    )


# --- find_papers_by_identifier ------------------------------------------------


def test_find_by_exact_paper_id(tmp_path: Path):
    settings = _settings(tmp_path, "test_delete_find_id_chunks")
    run_ingestion(FIXTURE_PATH, settings)
    [paper] = list_papers(settings)

    matches = find_papers_by_identifier(paper.paper_id, settings)

    assert [m.paper_id for m in matches] == [paper.paper_id]


def test_find_by_paper_id_prefix(tmp_path: Path):
    settings = _settings(tmp_path, "test_delete_find_prefix_chunks")
    run_ingestion(FIXTURE_PATH, settings)
    [paper] = list_papers(settings)

    matches = find_papers_by_identifier(paper.paper_id[:12], settings)

    assert [m.paper_id for m in matches] == [paper.paper_id]


def test_find_by_title_substring_case_insensitive(tmp_path: Path):
    settings = _settings(tmp_path, "test_delete_find_title_chunks")
    run_ingestion(FIXTURE_PATH, settings)

    matches = find_papers_by_identifier("retrieval-augmented", settings)

    assert len(matches) == 1
    assert matches[0].label == RAG_TITLE


def test_find_returns_all_ambiguous_title_matches(tmp_path: Path):
    settings = _settings(tmp_path, "test_delete_find_ambiguous_chunks")
    run_ingestion(FIXTURE_PATH, settings)  # title contains "Retrieval"
    _ingest_extra_metadata_record(settings, "meta-retrieval-1", "Retrieval Methods in Practice")

    matches = find_papers_by_identifier("retrieval", settings)

    assert len(matches) == 2


def test_find_no_match_returns_empty(tmp_path: Path):
    settings = _settings(tmp_path, "test_delete_find_none_chunks")
    run_ingestion(FIXTURE_PATH, settings)

    assert find_papers_by_identifier("nonexistent-xyz", settings) == []


def test_find_empty_identifier_returns_empty(tmp_path: Path):
    settings = _settings(tmp_path, "test_delete_find_empty_chunks")
    run_ingestion(FIXTURE_PATH, settings)

    assert find_papers_by_identifier("   ", settings) == []


def test_exact_id_match_short_circuits_title_matches(tmp_path: Path):
    # A paper whose ID is given exactly must not also drag in a second paper that merely shares
    # a title word — the exact-ID tier wins outright.
    settings = _settings(tmp_path, "test_delete_find_shortcircuit_chunks")
    run_ingestion(FIXTURE_PATH, settings)
    _ingest_extra_metadata_record(settings, "meta-retrieval-2", "Retrieval Methods in Practice")
    rag = next(p for p in list_papers(settings) if p.label == RAG_TITLE)

    matches = find_papers_by_identifier(rag.paper_id, settings)

    assert [m.paper_id for m in matches] == [rag.paper_id]


# --- delete_papers ------------------------------------------------------------


def test_delete_papers_removes_only_the_target(tmp_path: Path):
    settings = _settings(tmp_path, "test_delete_fn_chunks")
    run_ingestion(FIXTURE_PATH, settings)
    _ingest_extra_metadata_record(settings, "keep-me", "An Unrelated Paper")
    rag = next(p for p in list_papers(settings) if p.label == RAG_TITLE)

    removed = delete_papers([rag.paper_id], settings)

    assert removed == rag.chunk_count
    remaining = list_papers(settings)
    assert [p.paper_id for p in remaining] == ["keep-me"]


# --- CLI: scholarmind delete --------------------------------------------------


def _runner_env(monkeypatch, settings: Settings) -> CliRunner:
    monkeypatch.setenv("QDRANT_PATH", settings.qdrant_path)
    monkeypatch.setenv("QDRANT_COLLECTION", settings.qdrant_collection)
    return CliRunner()


def test_cli_delete_dry_run_does_not_delete(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "test_delete_cli_dry_chunks")
    run_ingestion(FIXTURE_PATH, settings)
    runner = _runner_env(monkeypatch, settings)

    result = runner.invoke(app, ["delete", "retrieval-augmented"])

    assert result.exit_code == 0
    assert "Dry run" in result.stdout
    assert len(list_papers(settings)) == 1  # nothing deleted


def test_cli_delete_with_yes_removes_the_paper(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "test_delete_cli_yes_chunks")
    run_ingestion(FIXTURE_PATH, settings)
    runner = _runner_env(monkeypatch, settings)

    result = runner.invoke(app, ["delete", "retrieval-augmented", "--yes"])

    assert result.exit_code == 0
    assert "Deleted" in result.stdout
    assert list_papers(settings) == []


def test_cli_delete_no_match_exits_nonzero(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "test_delete_cli_nomatch_chunks")
    run_ingestion(FIXTURE_PATH, settings)
    runner = _runner_env(monkeypatch, settings)

    result = runner.invoke(app, ["delete", "does-not-exist", "--yes"])

    assert result.exit_code == 1
    assert "No paper matched" in result.stdout
    assert len(list_papers(settings)) == 1  # untouched


def test_cli_delete_ambiguous_match_refuses(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "test_delete_cli_ambiguous_chunks")
    run_ingestion(FIXTURE_PATH, settings)  # title contains "Retrieval"
    _ingest_extra_metadata_record(settings, "meta-retrieval-3", "Retrieval Methods in Practice")
    runner = _runner_env(monkeypatch, settings)

    result = runner.invoke(app, ["delete", "retrieval", "--yes"])

    assert result.exit_code == 1
    assert "matched 2 papers" in result.stdout
    assert len(list_papers(settings)) == 2  # nothing deleted on ambiguity
