from pathlib import Path

import typer

from scholarmind.agents.llm_client import OpenRouterClient
from scholarmind.agents.qa import answer_question
from scholarmind.config import get_settings
from scholarmind.ingestion.pipeline import run_ingestion

app = typer.Typer(help="ScholarMind: a multi-agent RAG research assistant for PhD students.")


@app.command()
def ingest(path: str = typer.Argument(..., help="Path to a document or directory to ingest.")) -> None:
    result = run_ingestion(Path(path))
    typer.echo(
        f"Ingested {result.papers_ingested} paper(s), {result.chunks_created} chunk(s) "
        f"into collection '{result.collection_name}'."
    )


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


if __name__ == "__main__":
    app()
