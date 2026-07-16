from pathlib import Path

import typer

from scholarmind.agents.llm_client import OpenRouterClient
from scholarmind.agents.qa import AnswerResult, answer_question
from scholarmind.citations.service import FormattedAndVerifiedAnswer
from scholarmind.config import get_settings
from scholarmind.ingestion.pipeline import IngestResult, run_ingestion
from scholarmind.orchestrator import run as orchestrator_run

app = typer.Typer(help="ScholarMind: a multi-agent RAG research assistant for PhD students.")


def _print_ingest_result(result: "IngestResult") -> None:
    typer.echo(
        f"Ingested {result.papers_ingested} paper(s), {result.chunks_created} chunk(s) "
        f"into collection '{result.collection_name}'."
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
            typer.echo(f"[{verification.citation_index}] {verification.claim} — {verification.reason}")


@app.command()
def ingest(path: str = typer.Argument(..., help="Path to a document or directory to ingest.")) -> None:
    result = run_ingestion(Path(path))
    _print_ingest_result(result)


@app.command()
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


@app.command()
def chat(request: str = typer.Argument(..., help="A request: a question, or a path/'ingest <path>' to ingest.")) -> None:
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
    else:
        typer.echo("No result produced.")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind the API server to."),
    port: int = typer.Option(8000, help="Port to bind the API server to."),
) -> None:
    import uvicorn

    uvicorn.run("scholarmind.api.app:app", host=host, port=port)


if __name__ == "__main__":
    app()
