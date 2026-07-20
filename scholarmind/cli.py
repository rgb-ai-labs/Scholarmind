import tempfile
from pathlib import Path

import typer

from scholarmind.agents.base import AgentResult
from scholarmind.agents.llm_client import OpenRouterClient
from scholarmind.agents.qa import AnswerResult, answer_question
from scholarmind.citations.service import FormattedAndVerifiedAnswer
from scholarmind.config import Settings, get_settings
from scholarmind.eval.runner import Scorecard, load_eval_set, run_eval
from scholarmind.ingestion.pipeline import IngestResult, run_ingestion
from scholarmind.orchestrator import run as orchestrator_run

app = typer.Typer(help="ScholarMind: a multi-agent RAG research assistant for PhD students.")


def _print_ingest_result(result: "IngestResult") -> None:
    typer.echo(
        f"Ingested {result.papers_ingested} paper(s), {result.chunks_created} chunk(s) "
        f"into collection '{result.collection_name}'."
    )
    for title in result.duplicate_title_warnings:
        typer.echo(
            f"Warning: '{title}' already appears to be in your library under a different "
            "ID — this may be a duplicate. Run `scholarmind dedupe` to review."
        )


def _print_answer_result(result: "AnswerResult", question: str) -> None:
    if result.answer is None:
        typer.echo(f"No relevant sources found for: {question}")
        return

    typer.echo(result.answer.text)
    typer.echo("")
    typer.echo("Sources:")
    for citation in result.answer.citations:
        authors = ", ".join(citation.authors)
        typer.echo(
            f"[{citation.index}] {citation.title} — {authors} ({citation.year}), "
            f"{citation.section}, pp. {citation.page_start}-{citation.page_end}"
        )

    if result.answer.invalid_citation_markers:
        markers = ", ".join(f"[{m}]" for m in result.answer.invalid_citation_markers)
        typer.echo(
            f"Note: the model referenced source(s) {markers} which do not exist in the "
            "sources list above and were not included."
        )


def _print_formatted_answer(formatted: "FormattedAndVerifiedAnswer") -> None:
    typer.echo("")
    typer.echo("References:")
    for reference in formatted.references:
        typer.echo(f"[{reference.citation_index}] {reference.apa}")
        typer.echo(reference.bibtex)

    unsupported = [v for v in formatted.verification_report.verifications if not v.supported]
    if unsupported:
        typer.echo("")
        typer.echo("Warning: the following claims could not be verified against their sources:")
        for verification in unsupported:
            typer.echo(
                f"[{verification.citation_index}] {verification.claim} — {verification.reason}"
            )


def _print_agent_result(result: "AgentResult", request: str) -> None:
    if result.sources_found == 0:
        typer.echo(f"No relevant sources found for: {request}")
        return
    typer.echo(result.text)
    typer.echo("")
    typer.echo(f"(grounded in {result.sources_found} retrieved source(s))")


@app.command(help="Ingest a PDF or a directory of PDFs into the knowledge base.")
def ingest(
    path: str = typer.Argument(..., help="Path to a document or directory to ingest."),
) -> None:
    result = run_ingestion(Path(path))
    _print_ingest_result(result)


@app.command(help="Answer a question grounded only in ingested sources, with citations.")
def ask(question: str = typer.Argument(..., help="Research question to ask.")) -> None:
    settings = get_settings()
    client = OpenRouterClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    )

    result = answer_question(question, client, settings)
    _print_answer_result(result, question)


@app.command(help="Route a request: ingest a path, ask a question, or run a domain agent.")
def chat(
    request: str = typer.Argument(
        ..., help="A question, 'ingest <path>', or '<summarize|gaps|methods|write> <topic>'."
    ),
) -> None:
    result = orchestrator_run.run(request)

    if result.error is not None:
        typer.echo(f"Error: {result.error}")
    elif result.ingest_result is not None:
        _print_ingest_result(result.ingest_result)
    elif result.answer_result is not None:
        _print_answer_result(result.answer_result, result.answer_result.question)
        if result.formatted_answer is not None:
            _print_formatted_answer(result.formatted_answer)
        elif result.formatting_error is not None:
            typer.echo("")
            typer.echo(
                f"Note: references could not be formatted/verified ({result.formatting_error}); "
                "the answer above is unchanged."
            )
    elif result.agent_result is not None:
        _print_agent_result(result.agent_result, request)
    else:
        typer.echo("No result produced.")


@app.command(help="Start the FastAPI server (uvicorn) exposing /ingest, /ask, /health.")
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind the API server to."),
    port: int = typer.Option(8000, help="Port to bind the API server to."),
) -> None:
    import uvicorn

    uvicorn.run("scholarmind.api.app:app", host=host, port=port)


@app.command(
    name="app",
    help="Launch the Streamlit web UI (upload PDFs, ask questions) at http://localhost:8501.",
)
def launch_webapp() -> None:
    import subprocess
    import sys

    # Embedded Qdrant is single-process: don't run this alongside `scholarmind serve`
    # against the same QDRANT_PATH at the same time.
    webapp_path = Path(__file__).resolve().parent / "webapp" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(webapp_path)])


def _print_scorecard(scorecard: "Scorecard") -> None:
    typer.echo(f"Eval scorecard (k={scorecard.k}, cases={scorecard.num_cases})")
    typer.echo(f"  mean precision@{scorecard.k}: {scorecard.mean_precision_at_k:.3f}")
    typer.echo(f"  mean recall@{scorecard.k}:    {scorecard.mean_recall_at_k:.3f}")
    typer.echo(f"  mean faithfulness:   {scorecard.mean_faithfulness:.3f}")
    typer.echo("")
    typer.echo("Per-case results:")
    for case in scorecard.per_case:
        typer.echo(
            f"  - {case.question!r} (expected: {case.expected_title}) "
            f"precision@{scorecard.k}={case.precision_at_k:.3f} "
            f"recall@{scorecard.k}={case.recall_at_k:.3f} "
            f"faithfulness={case.faithfulness:.3f}"
        )


@app.command(
    help=(
        "Find papers that are duplicated under different IDs (same normalized title, "
        "different content hash) and remove the extra copies. Dry-run by default."
    )
)
def dedupe(
    apply: bool = typer.Option(
        False, "--apply", help="Actually delete the duplicate papers' chunks (default: dry-run)."
    ),
) -> None:
    from scholarmind.ingestion.dedupe import delete_papers, find_duplicate_paper_groups

    settings = get_settings()
    groups = find_duplicate_paper_groups(settings)

    if not groups:
        typer.echo("No duplicate papers found.")
        return

    for group in groups:
        typer.echo(f"'{group.title}':")
        typer.echo(f"  keep:   {group.keep.paper_id} ({group.keep.chunk_count} chunk(s))")
        for paper in group.remove:
            typer.echo(f"  remove: {paper.paper_id} ({paper.chunk_count} chunk(s))")

    if not apply:
        typer.echo("")
        typer.echo(
            f"Dry run - {len(groups)} duplicate group(s) found. Re-run with --apply to remove them."
        )
        return

    remove_ids = [paper.paper_id for group in groups for paper in group.remove]
    deleted = delete_papers(remove_ids, settings)
    typer.echo("")
    typer.echo(f"Removed {deleted} chunk(s) across {len(remove_ids)} duplicate paper(s).")


@app.command(
    help=(
        "Delete one ingested paper from the library, matched by paper ID, ID prefix, or a "
        "distinctive part of its title. Dry-run by default; the source PDF is left in place."
    )
)
def delete(
    identifier: str = typer.Argument(
        ..., help="A paper ID, an ID prefix, or a distinctive part of the title."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Actually delete (default: dry-run preview)."
    ),
) -> None:
    from scholarmind.retrieval.papers import delete_papers, find_papers_by_identifier

    settings = get_settings()
    matches = find_papers_by_identifier(identifier, settings)

    if not matches:
        typer.echo(f"No paper matched '{identifier}'. Nothing to delete.")
        raise typer.Exit(code=1)

    if len(matches) > 1:
        typer.echo(
            f"'{identifier}' matched {len(matches)} papers — be more specific, or use an exact "
            "paper ID:"
        )
        for paper in matches:
            typer.echo(f"  {paper.paper_id}  {paper.label} ({paper.chunk_count} chunk(s))")
        raise typer.Exit(code=1)

    paper = matches[0]
    typer.echo(f"{paper.paper_id}  {paper.label} ({paper.chunk_count} chunk(s))")

    if not yes:
        typer.echo("")
        typer.echo(
            "Dry run - re-run with --yes to delete this paper. (Its PDF in the uploads folder "
            "is kept, so you can re-ingest it later.)"
        )
        return

    removed = delete_papers([paper.paper_id], settings)
    typer.echo("")
    typer.echo(f"Deleted '{paper.label}' ({removed} chunk(s) removed).")


@app.command(help="Run the labelled eval set and print a precision/recall/faithfulness scorecard.")
def eval(k: int = typer.Option(5, help="Cut-off k for precision@k / recall@k.")) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    eval_set_path = repo_root / "tests" / "eval_set" / "eval_set.json"
    fixtures_dir = repo_root / "tests" / "fixtures"

    eval_qdrant_path = tempfile.mkdtemp(prefix="scholarmind_eval_")
    eval_settings = Settings(
        qdrant_path=eval_qdrant_path,
        qdrant_collection="scholarmind_eval_chunks",
    )

    run_ingestion(fixtures_dir / "sample_paper.pdf", eval_settings)
    run_ingestion(fixtures_dir / "sample_paper_2.pdf", eval_settings)

    settings = get_settings()
    client = OpenRouterClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    )

    cases = load_eval_set(eval_set_path)
    scorecard = run_eval(cases, client, eval_settings, k=k)
    _print_scorecard(scorecard)


if __name__ == "__main__":
    app()
