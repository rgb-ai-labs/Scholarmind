from pathlib import Path

import typer

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
    """Ask a research question against the ingested knowledge base."""
    typer.echo("not implemented")


if __name__ == "__main__":
    app()
