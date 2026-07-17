# ScholarMind

**A multi-agent, RAG-powered research assistant for PhD students.** ScholarMind ingests your
papers into a local vector index and answers research questions with **verified, citation-backed**
answers — every claim is grounded in a retrieved passage, and a second model pass checks that the
cited passage actually supports the claim. It runs entirely on your machine: embedded Qdrant, no
cloud vector database, and **no Docker**.

<!-- ![ScholarMind architecture](docs/architecture.png) -->
<!-- TODO: add an architecture diagram image at docs/architecture.png and uncomment the line above. -->

## Why

Large language models hallucinate citations. ScholarMind's cardinal rule is **"never cite from
memory, always verify"**: the LLM only ever sees numbered passages retrieved from *your* ingested
corpus, every citation marker is validated against that passage list, and unsupported claims are
flagged back to you rather than presented as fact.

## Architecture

Four layers (see [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design):

1. **Interaction** — CLI (Typer) and HTTP API (FastAPI).
2. **Orchestration** — a [LangGraph](https://github.com/langchain-ai/langgraph) supervisor that
   classifies a request and routes it to the right agent.
3. **Agent** — a Q&A/citation agent plus five domain agents (discovery, summarization,
   gap-analysis, methodology, writing), all grounded in retrieval.
4. **Knowledge & retrieval** — a **hand-rolled hybrid-RAG layer over embedded Qdrant**: PDF
   parsing (pypdf), section/paragraph-aware chunking, local `sentence-transformers` embeddings,
   dense + BM25 sparse retrieval fused with Reciprocal Rank Fusion, and a cross-encoder reranker.
   *(No LlamaIndex — the retrieval stack is explicit and self-contained.)*

## Features

- **Ingestion** — PDF → clean text + section structure + bibliographic metadata → chunked,
  metadata-tagged, embedded, and stored idempotently (keyed by a content hash, so re-ingesting a
  paper updates rather than duplicates).
- **Hybrid retrieval** — dense vector search + real BM25 sparse search, RRF-fused, then
  cross-encoder reranked; results carry full citation metadata.
- **Grounded Q&A** — answers with inline `[N]` citation markers, a resolved sources list, and a
  low-confidence refusal ("no relevant sources found") when retrieval comes up empty or weak.
- **Citation & verification** — Crossref metadata normalization, APA + BibTeX formatting
  (pluggable styles), and a per-claim verifier that flags any citation whose passage doesn't
  support the claim.
- **Orchestrator** — `chat` routes `ingest`, `ask`, and the five domain agents through one graph.
- **HTTP API** — `POST /ingest`, `POST /ask`, `GET /health`, served by `scholarmind serve`.
- **Eval harness** — precision@k / recall@k / citation-faithfulness scoring over a labelled set
  (`scholarmind eval`), with centralized guardrails (confidence threshold, cite-only-ingested).

**149 tests** — 91 run fully offline with no API key (fast CI gate); the remaining 58 download
models or exercise live LLM/network paths.

## Quickstart

```bash
# 1. Install dependencies (uv; falls back to pip)
uv sync --extra dev

# 2. Configure — copy the template and add an OpenRouter (or any OpenAI-compatible) key
cp .env.example .env
# then edit .env: set LLM_API_KEY=sk-or-...   (a free OpenRouter model is the default)

# 3. Ingest a PDF (or a directory of PDFs)
uv run scholarmind ingest path/to/paper.pdf

# 4. Ask a question, grounded in what you ingested
uv run scholarmind ask "What does this paper propose for grounding LLM answers?"

# Or route everything through the orchestrator
uv run scholarmind chat "summarize retrieval-augmented generation"

# Serve the HTTP API
uv run scholarmind serve
```

Run `uv run scholarmind --help` to see all commands (`ingest`, `ask`, `chat`, `serve`, `eval`).
See the [User Guide](docs/USER_GUIDE.md) for the full command/config reference and a walkthrough.

## Configuration

All settings live in `scholarmind/config.py` (pydantic-settings) and are read from `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_PATH` | `./data/qdrant` | Embedded Qdrant storage folder (no server) |
| `QDRANT_COLLECTION` | `scholarmind_chunks` | Collection name for chunk vectors |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `LLM_API_KEY` | *(empty)* | OpenRouter / OpenAI-compatible API key |
| `LLM_MODEL` | `google/gemma-4-26b-a4b-it:free` | Generation model |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | LLM API base URL (swap the provider here) |
| `LLM_MAX_TOKENS` | `512` | Max tokens per generation |
| `CHUNK_SIZE` | `800` | Target chunk size (characters) |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks (characters) |
| `RETRIEVAL_CANDIDATE_K` | `20` | Candidates fetched before reranking |
| `RETRIEVAL_TOP_K` | `5` | Results returned after reranking |
| `RETRIEVAL_MIN_RERANK_SCORE` | `-7.0` | Cross-encoder score below which a chunk is refused |

No API key is needed to ingest, retrieve, or run the offline test suite — only the LLM-backed
answer/verification/agent steps require one.

## Roadmap / build phases

Phases 1–8 are implemented and tested; phase 9 (this pass) is packaging for release.

1. ✅ Project scaffold (config, CLI, packaging)
2. ✅ Ingestion pipeline (PDF → chunks → embedded Qdrant, idempotent)
3. ✅ Hybrid retrieval (dense + BM25, RRF, cross-encoder rerank)
4. ✅ Q&A RAG agent with inline citations + low-confidence refusal
5. ✅ LangGraph orchestrator (`chat`)
6. ✅ Citation & verification agent (Crossref, APA/BibTeX, claim verifier)
7. ✅ FastAPI + CLI
8. ✅ Eval harness & guardrails
9. 🔜 Docs & release polish

Future ideas: additional source connectors (HTML, arXiv/DOI), more citation styles, an OpenAlex
fallback for metadata, and a real Zotero export.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) — including the test-marker convention and a list of
good first issues (new source connectors, new citation styles).

## License

[Apache-2.0](LICENSE).
