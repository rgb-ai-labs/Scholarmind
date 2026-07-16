# Contributing to ScholarMind

Thanks for your interest in contributing.

## Setup

```bash
make setup      # uv sync (falls back to pip if uv is unavailable)
cp .env.example .env
```

## Workflow

1. Open an issue describing the change before starting significant work.
2. Create a branch off `main`.
3. Write tests first where practical (see `tests/`).
4. Run `make test` and `make lint` before opening a PR.
5. Keep PRs scoped to one concern; note any architectural decisions in the PR description.

## Project conventions

- Python 3.11+, typed where reasonable.
- Configuration lives only in `scholarmind/config.py` (pydantic-settings, `.env`-driven).
- No Docker, no external services required to run or test locally — Qdrant runs embedded.
- See `CLAUDE.md` for architecture and the current build phase.

## Code of conduct

Be respectful and constructive. Assume good faith.
