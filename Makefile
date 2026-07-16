.PHONY: setup run test lint ingest ask

setup:
	uv sync

run:
	uv run uvicorn scholarmind.api:app --reload

test:
	uv run pytest

lint:
	uv run ruff check .

ingest:
	uv run scholarmind ingest $(PATH)

ask:
	uv run scholarmind ask "$(Q)"
