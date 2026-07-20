# Contributing to ScholarMind

Thanks for your interest in contributing.

## Setup

```bash
uv sync --extra dev --extra webapp    # runtime + dev deps (ruff, pytest) + the Streamlit UI
cp .env.example .env                   # then add your OpenRouter/LLM key for live features
```

## Workflow

1. Open an issue describing the change before starting significant work.
2. Create a branch off `main`.
3. Write tests first where practical (see `tests/`).
4. Before opening a PR, run:
   - `uv run ruff check .`
   - `uv run pytest -m "not slow and not llm"` — the fast, offline suite (no models, no key)
5. Keep PRs scoped to one concern; note any architectural decisions in the PR description.

## Test markers

The suite is split so contributors and CI can run a fast offline subset:

- `@pytest.mark.slow` — tests that download/load HuggingFace embedding or reranker models.
- `@pytest.mark.llm` — tests that make a live LLM or network (OpenRouter/Crossref) call.

Markers are applied centrally in `tests/conftest.py`. If you add a test that loads a model
or hits the network, add it to the appropriate set there so the fast CI job stays fast and
offline. Live-LLM tests must also be `skipif`-guarded on `LLM_API_KEY` so a keyless run is green.

## Project conventions

- Python 3.11+, typed where reasonable. Modules are docstring-free — put non-obvious *why*
  in a short `#` comment, never a "what it does" docstring.
- Configuration lives only in `scholarmind/config.py` (pydantic-settings, `.env`-driven).
- No Docker, no external services required to run or test locally — Qdrant runs embedded.
- Never commit secrets, API keys, or ingested data (`.env` and `data/` are gitignored).
- See `ARCHITECTURE.md` for the architecture and build phases.

## Good first issues

These are self-contained and don't require touching the orchestration core:

- **New source connectors** — the ingestion loader only handles PDFs today. Add a loader for
  another format (plain text, HTML) that produces the same `RawDocument` shape
  (`scholarmind/ingestion/loader.py`). (arXiv/Semantic Scholar/OpenAlex discovery — search plus
  ingest — already exists in `scholarmind/discovery/`.)
- **A second citation-graph hop, or an OpenAlex citation-graph provider** — `scholarmind/discovery/service.get_citation_graph`
  is deliberately 1-hop and Semantic Scholar-only; expanding either is a self-contained addition
  to `scholarmind/discovery/`.
- **New citation styles** — `scholarmind/citations/formatter.py` uses a pluggable style
  registry (APA/MLA/Chicago/IEEE/Vancouver/BibTeX today). Add another (e.g. Harvard) by
  registering a new formatter — it's a one-line registration plus the formatter class and
  its tests.
- **Zotero read/import** — `scholarmind/citations/zotero.py` only pushes (write); adding a
  `GET /items` read path to import an existing Zotero library is a self-contained addition.
- **A sixth intent for the orchestrator** — the `chat` router uses a registry + dispatcher
  (`scholarmind/orchestrator/graph.py`); adding an intent is a documented, small change.
- **Stricter Crossref match confidence** — `citations/metadata.py`'s `_MIN_MATCH_SCORE` (15.0)
  can accept a false-positive match for a short/generic title; tightening the threshold or
  reusing the title-similarity check already added for the OpenAlex/Semantic Scholar fallback
  path is self-contained, with `tests/test_citation_metadata.py` as the pattern to follow.

## Code of conduct

Be respectful and constructive. Assume good faith.
