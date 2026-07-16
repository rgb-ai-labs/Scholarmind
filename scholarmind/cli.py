import typer

app = typer.Typer(help="ScholarMind: a multi-agent RAG research assistant for PhD students.")


@app.command()
def ingest(path: str = typer.Argument(..., help="Path to a document or directory to ingest.")) -> None:
    """Ingest documents into the knowledge base."""
    typer.echo("not implemented")


@app.command()
def ask(question: str = typer.Argument(..., help="Research question to ask.")) -> None:
    """Ask a research question against the ingested knowledge base."""
    typer.echo("not implemented")


if __name__ == "__main__":
    app()
