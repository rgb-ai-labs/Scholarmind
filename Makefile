.PHONY: setup run test test-all lint fmt ingest ask

setup:
	uv sync --extra dev

run:
	uv run scholarmind serve

# Fast, offline, deterministic suite (no model downloads, no API key) — the default dev loop.
test:
	uv run pytest -m "not slow and not llm"

# Everything, including model-download and live-LLM/network tests.
test-all:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff check . --fix

ingest:
	uv run scholarmind ingest $(PATH)

ask:
	uv run scholarmind ask "$(Q)"
