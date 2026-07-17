# Contributing to ScholarMind

Thanks for your interest in contributing.

## Setup

```bash
uv sync --extra dev     # install runtime + dev dependencies (ruff, pytest)
cp .env.example .env     # then add your OpenRouter/LLM key for live features
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
- See `CLAUDE.md` for the architecture and build phases.

## Good first issues

These are self-contained and don't require touching the orchestration core:

- **New source connectors** — the ingestion loader only handles PDFs today. Add a loader for
  another format (plain text, HTML, arXiv/DOI fetch) that produces the same `RawDocument`
  shape (`scholarmind/ingestion/loader.py`).
- **New citation styles** — `scholarmind/citations/formatter.py` uses a pluggable style
  registry (APA + BibTeX today). Add MLA, Chicago, or IEEE by registering a new formatter —
  it's a one-line registration plus the formatter class and its tests.
- **Wire up the Zotero sync hook** — there's a `# TODO: Zotero sync hook` marker in
  `formatter.py` for exporting formatted references via the Zotero Web API.
- **A sixth intent for the orchestrator** — the `chat` router uses a registry + dispatcher
  (`scholarmind/orchestrator/graph.py`); adding an intent is a documented, small change.

## Code of conduct

Be respectful and constructive. Assume good faith.
