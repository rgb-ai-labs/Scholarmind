# ScholarMind

A multi-agent, RAG-powered research assistant for PhD students. ScholarMind ingests papers
into a local vector index and answers research questions with verified, citation-backed
answers — no cloud vector database, no Docker required.

## Architecture

ScholarMind is organized into four layers (see `CLAUDE.md` for full details):

1. **Interaction** — CLI (Typer) and API (FastAPI) entry points.
2. **Orchestration** — a LangGraph supervisor that routes work across specialized agents.
3. **Agent** — eight task-specific agents (discovery, ingestion, Q&A, summarization,
   gap-analysis, citation, methodology, writing).
4. **Knowledge & Retrieval** — LlamaIndex-driven ingestion and hybrid retrieval backed by a
   local, embedded Qdrant store.

## Quickstart

```bash
# Install dependencies (uv preferred, falls back to pip)
make setup

# Copy env template and fill in your LLM API key
cp .env.example .env

# Run tests
make test

# CLI
uv run scholarmind --help
```

## Status

This is the project scaffold only. Ingestion, retrieval, and agent logic are not yet
implemented — `ingest` and `ask` are stubs.
