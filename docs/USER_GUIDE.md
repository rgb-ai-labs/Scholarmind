# ScholarMind — User Guide

ScholarMind is a local, multi-agent RAG research assistant: it ingests your PDFs into an
embedded vector index and answers research questions grounded **only** in what you ingested,
with inline citations and a verification pass that flags any claim its cited passage doesn't
support. Everything runs on your machine — embedded Qdrant, local embedding/reranker models,
and an OpenAI-compatible LLM API — with no Docker and no cloud vector database.

---

## Prerequisites

- **Python 3.11+** (the repo pins 3.13 via `.python-version`).
- **[uv](https://github.com/astral-sh/uv)** for dependency management (a `pip` fallback works too).
- Network access on first run — the embedding and reranker models download from Hugging Face
  and are then cached locally.
- An **OpenRouter** (or any OpenAI-compatible) API key for the answer/verification/agent steps.
  Not needed for ingestion or retrieval.

## Install

```bash
git clone <your-fork-url> scholarmind
cd scholarmind

uv sync --extra dev          # install runtime + dev dependencies

cp .env.example .env         # then edit .env and set LLM_API_KEY (see below)
```

Verify the install with the fast, offline test suite (no key, no model download needed):

```bash
uv run pytest -m "not slow and not llm" -q     # ~90 tests, ~11s
uv run scholarmind --help                       # lists all commands
```

---

## Configuration reference

All settings live in `scholarmind/config.py` (pydantic-settings) and are read from `.env`
(environment variables of the same UPPER_CASE name override the file).

| `.env` variable | Default | Purpose |
|---|---|---|
| `QDRANT_PATH` | `./data/qdrant` | Folder for the embedded Qdrant store (no server) |
| `QDRANT_COLLECTION` | `scholarmind_chunks` | Collection name holding chunk vectors |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model (query + chunks) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used to rerank candidates |
| `LLM_API_KEY` | *(empty)* | Your OpenRouter / OpenAI-compatible key |
| `LLM_MODEL` | `google/gemma-4-26b-a4b-it:free` | Model used for answers, verification, and agents |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | LLM API base URL — change to swap providers |
| `LLM_MAX_TOKENS` | `512` | Max tokens per generation |
| `CHUNK_SIZE` | `800` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `150` | Character overlap between consecutive chunks |
| `RETRIEVAL_CANDIDATE_K` | `20` | Candidates fetched (dense + sparse) before reranking |
| `RETRIEVAL_TOP_K` | `5` | Results kept after reranking |
| `RETRIEVAL_MIN_RERANK_SCORE` | `-7.0` | Cross-encoder score below which a chunk is dropped (drives the low-confidence refusal) |

## Getting an LLM key

- **Provider:** any OpenAI-compatible chat-completions API. The default is
  [OpenRouter](https://openrouter.ai), which offers free-tier models (the default
  `google/gemma-4-26b-a4b-it:free` needs no payment, only a free account + key).
- **Where to put it:** `LLM_API_KEY` in `.env`. To use a different provider, also set
  `LLM_BASE_URL` and `LLM_MODEL`.
- **Where it's used:** the Q&A answer step (`scholarmind/agents/qa.py`), the claim verifier
  (`scholarmind/citations/verifier.py`), and the domain agents (`scholarmind/agents/`). Ingestion,
  retrieval, and metadata lookup do **not** use it.

---

## Command reference

Run any command with `uv run scholarmind <command>` (or `python -m scholarmind.cli <command>`).
`uv run scholarmind --help` lists them; `--help` after a command shows its options.

### `ingest` — add documents to the knowledge base

```
scholarmind ingest PATH
```
- `PATH` (required) — a single PDF file, or a directory of PDFs (scanned non-recursively).
- No API key required. Re-ingesting the same file updates it in place (idempotent, keyed by a
  content hash) rather than duplicating.

**Example**
```bash
uv run scholarmind ingest papers/rag_survey.pdf
```
**Output shape**
```
Ingested 1 paper(s), 5 chunk(s) into collection 'scholarmind_chunks'.
```

### `ask` — grounded, cited answer

```
scholarmind ask QUESTION
```
- `QUESTION` (required) — a natural-language research question.
- **Requires `LLM_API_KEY`.** Answers only from ingested sources; if retrieval finds nothing
  relevant it refuses instead of guessing.

**Example**
```bash
uv run scholarmind ask "What does this paper propose for grounding LLM answers?"
```
**Output shape (answer found)**
```
<answer text with inline [1] [2] citation markers>

Sources:
[1] <title> — <authors> (<year>), <section>, pp. <start>-<end>
[2] ...

References:
[1] <APA-formatted reference>
@article{key, author={...}, title={...}, year={...}, ...}    # BibTeX

Warning: the following claims could not be verified against their sources:   # only if any
[2] <claim sentence> — <reason the passage doesn't support it>
```
**Output shape (nothing relevant)**
```
No relevant sources found for: <question>
```

### `chat` — route through the orchestrator

```
scholarmind chat REQUEST
```
- `REQUEST` (required) — the supervisor classifies it and routes to the right handler:
  - `"ingest <path>"` or a bare `<path>.pdf` → ingestion (no key needed)
  - `"summarize <topic>"`, `"gaps <topic>"`, `"methods <topic>"`, `"write <topic>"`,
    `"discover <topic>"` → the matching domain agent (needs key, except `discover` which is
    retrieval-only)
  - anything else → a grounded `ask`
- Unlike `ask`, a failure in the answer/agent path is caught and printed as `Error: ...` rather
  than crashing.

**Example**
```bash
uv run scholarmind chat "summarize retrieval-augmented generation"
```
**Output shape (a domain agent)**
```
<grounded summary / gap list / draft ...>

(grounded in 5 retrieved source(s))
```

### `serve` — HTTP API

```
scholarmind serve [--host 127.0.0.1] [--port 8000]
```
- Starts a FastAPI (uvicorn) server exposing `GET /health`, `POST /ingest`, `POST /ask`,
  backed by the same entry points as the CLI. Interactive docs at `/docs`.

### `eval` — quality scorecard

```
scholarmind eval [--k 5]
```
- `--k` (default `5`) — the cut-off for precision@k / recall@k.
- Ingests the two committed fixture papers into a scratch collection, runs a labelled question
  set, and prints retrieval precision/recall plus citation faithfulness. **Requires `LLM_API_KEY`**
  (faithfulness runs the verifier). Never touches your real `./data/qdrant`.

**Output shape**
```
Eval scorecard (k=5, cases=4)
  mean precision@5: 1.000
  mean recall@5:    1.000
  mean faithfulness:   1.000

Per-case results:
  - '<question>' (expected: <paper title>) precision@5=1.000 recall@5=1.000 faithfulness=1.000
  ...
```

---

## Full walkthrough

```bash
# 0. one-time: install + configure
uv sync --extra dev
cp .env.example .env          # set LLM_API_KEY=sk-or-...

# 1. Ingest a paper
uv run scholarmind ingest papers/rag_survey.pdf
#   -> Ingested 1 paper(s), 12 chunk(s) into collection 'scholarmind_chunks'.

# 2. Ask a grounded question
uv run scholarmind ask "How does the paper reduce hallucination?"
#   -> answer with [1]/[2] markers, a Sources list, formatted References,
#      and (if any) a Warning listing unverified claims.

# 3. Use the orchestrator (routes to the summarization agent)
uv run scholarmind chat "summarize the paper's evaluation methodology"

# 4. Serve the API and call it
uv run scholarmind serve            # in one terminal
```
```bash
# ... in another terminal:
curl http://127.0.0.1:8000/health
#   {"status":"ok"}

# ingest by path (multipart form field) or by file upload
curl -F "path=papers/rag_survey.pdf" http://127.0.0.1:8000/ingest
curl -F "file=@papers/rag_survey.pdf" http://127.0.0.1:8000/ingest
#   {"papers_ingested":1,"chunks_created":12,"collection_name":"scholarmind_chunks"}

# ask (JSON body)
curl -H "Content-Type: application/json" \
     -d '{"question":"How does the paper reduce hallucination?"}' \
     http://127.0.0.1:8000/ask
#   {"intent":"ask","answer":"...[1]...","sources_found":5,
#    "sources":[{"index":1,"title":...,"authors":[...],...}],
#    "invalid_citation_markers":[],
#    "references":[{"citation_index":1,"apa":"...","bibtex":"..."}],
#    "verification_report":{"verifications":[...],"unsupported_count":0},
#    "formatting_error":null,"error":null}
```
```bash
# 5. Score retrieval + citation quality
uv run scholarmind eval
```

---

## How citations & verification look to you

- **`[N]` markers** in the answer text point to entry `[N]` in the **Sources** list below it.
  Each source resolves to a real ingested chunk (title, authors, year, section, page range).
- Every source also appears in **References** formatted in APA and BibTeX (BibTeX is copy-paste
  ready for a `.bib` file).
- After generation, each cited claim is re-checked against its passage by a second model pass.
  Any claim the passage does **not** support is listed under **"Warning: the following claims
  could not be verified against their sources"** with the reason — the answer text is left
  intact, but you're told which parts to distrust.
- If the model cites a number that was never provided (a fabricated source), you'll see a
  **"Note: the model referenced source(s) [N] which do not exist…"** line, and that marker is
  excluded from the sources/references.

---

## Troubleshooting

- **`OpenAIError: Missing credentials` when running `ask`.** `LLM_API_KEY` is empty. Set it in
  `.env`. (`ingest`, `discover`, and the fast test suite work without a key; `chat` prints a
  clean `Error: ...` instead of crashing.)
- **"No relevant sources found".** Retrieval returned nothing above the confidence threshold —
  either you haven't ingested anything on that topic, or the question is off-corpus. Ingest the
  relevant paper, or lower `RETRIEVAL_MIN_RERANK_SCORE` (default `-7.0`) if it's too strict.
- **Slow first run.** The first ingestion/ask/eval downloads the embedding + reranker models
  from Hugging Face (hundreds of MB). Subsequent runs use the local cache and are fast.
- **`Storage folder ... already accessed by another instance`.** Embedded Qdrant is
  single-process: only one process can open a given `QDRANT_PATH` at a time. Don't run `serve`
  and a CLI command against the same store simultaneously (or point them at different
  `QDRANT_PATH`s).
- **Free-tier rate limits.** The default OpenRouter free model is shared and rate-limited; a
  `429` means retry shortly or configure your own key/model.

## FAQ

- **Do I need Docker or a Qdrant server?** No. Qdrant runs embedded from a local folder.
- **Does my data leave my machine?** Only the LLM calls (your question + retrieved passages) go
  to the configured LLM provider. Ingestion, embedding, and retrieval are fully local. Metadata
  lookup queries Crossref (public API) by paper title.
- **Can I use OpenAI / a local model instead of OpenRouter?** Yes — set `LLM_BASE_URL`,
  `LLM_MODEL`, and `LLM_API_KEY` to any OpenAI-compatible endpoint.
- **How do I add another citation style or file format?** See `CONTRIBUTING.md` — the citation
  formatter uses a pluggable registry, and the ingestion loader is the place for new formats.
- **Where's the architecture overview?** [`ARCHITECTURE.md`](../ARCHITECTURE.md).
