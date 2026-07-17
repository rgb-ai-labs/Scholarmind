## What this changes

Short summary of the change and why.

Closes #

## Type

- [ ] Bug fix
- [ ] New feature (new agent, connector, citation style, …)
- [ ] Refactor / internal
- [ ] Docs / CI

## Checklist

- [ ] `uv run ruff check .` passes
- [ ] `uv run pytest -m "not slow and not llm"` passes (fast, offline suite)
- [ ] New model-download tests are marked `@pytest.mark.slow`; new live-LLM/network tests are marked `@pytest.mark.llm` (see `tests/conftest.py`)
- [ ] Code stays docstring-free and type-hinted, per project style
- [ ] No secrets, API keys, or ingested data committed
- [ ] If a new agent: grounded in retrieved sources only; refuses when retrieval is empty

## Notes for reviewers

Anything non-obvious, trade-offs, or follow-ups.
