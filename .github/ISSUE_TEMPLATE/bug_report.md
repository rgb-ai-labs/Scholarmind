---
name: Bug report
about: Report something that is broken or behaving unexpectedly
title: "[bug] "
labels: bug
---

## What happened

A clear description of the bug.

## Expected behavior

What you expected instead.

## Steps to reproduce

1. Ingest `...`
2. Run `scholarmind ask "..."` (or the API call / code snippet)
3. See error

## Environment

- OS:
- Python version (`python --version`):
- ScholarMind version / commit:
- Embedding / reranker / LLM model (if non-default):

## Logs / traceback

```
paste the full error output here
```

## Notes

- Is retrieval finding sources, or is it the "no relevant sources" path?
- Does it reproduce on the committed fixture corpus (`tests/fixtures/`)?
