# ScholarMind

[User Guide](USER_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Contributing](CONTRIBUTING.md) · [License](LICENSE)

**A multi-agent, RAG-powered research assistant for PhD students.** ScholarMind ingests your
papers into a local vector index and answers research questions with **verified, citation-backed**
answers — every claim is grounded in a retrieved passage, and a second model pass checks that the
cited passage actually supports the claim.

> **Runs 100% locally — no Pinecone, no cloud vector database, no Docker, no vector-store account
> or API key.** The RAG stack is fully on-device: embeddings run locally (`sentence-transformers`)
> and the vector store is **Qdrant in embedded mode** — think *SQLite for vectors*: it lives
> in-process and writes to a plain local folder (`./data/qdrant`) that's created automatically on
> first ingest. Your papers and their embeddings never leave your machine. The only network call
> is the final LLM step (any OpenAI-compatible API), plus optional public metadata lookups. Nothing
> to provision, no infrastructure to stand up — just `uv sync` and go.

<!-- ![ScholarMind architecture](architecture.png) -->
<!-- TODO: add an architecture diagram image at the repo root and uncomment the line above. -->

## Why

Large language models hallucinate citations. ScholarMind's cardinal rule is **"never cite from
memory, always verify"**: the LLM only ever sees numbered passages retrieved from *your* ingested
corpus, every citation marker is validated against that passage list, and unsupported claims are
flagged back to you rather than presented as fact.

## Architecture

Four layers (see [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design):

1. **Interaction** — CLI (Typer), HTTP API (FastAPI), and a Streamlit web app — all thin layers
   over the same engine functions.
2. **Orchestration** — a [LangGraph](https://github.com/langchain-ai/langgraph) supervisor that
   classifies a request and routes it to the right agent.
3. **Agent** — a Q&A/citation agent plus domain agents (discovery, summarization, gap-analysis,
   methodology, writing, novelty), all grounded in retrieval. The orchestrator's `chat` router
   only wires up the original five (`discover`/`summarize`/`gaps`/`methods`/`write`); novelty is
   reached through the web app only, since a single free-text `chat` intent doesn't fit its
   "check this passage" shape.
4. **Knowledge & retrieval** — a **hand-rolled hybrid-RAG layer over embedded Qdrant**: PDF
   parsing (pypdf), section/paragraph-aware chunking, local `sentence-transformers` embeddings,
   dense + BM25 sparse retrieval fused with Reciprocal Rank Fusion, and a cross-encoder reranker.
   *(No LlamaIndex — the retrieval stack is explicit and self-contained.)*

## Features

- **Ingestion & library management** — PDF → clean text + section structure + bibliographic
  metadata → chunked, metadata-tagged, embedded, and stored idempotently (keyed by a content hash,
  so re-ingesting a paper updates rather than duplicates). A byte-different copy of a paper you
  already have (e.g. re-downloaded or regenerated) is flagged with a warning rather than silently
  duplicated; `scholarmind dedupe` finds and removes any duplicates, and `scholarmind delete`
  removes a single paper by ID or title (also a per-paper delete button in the web app).
- **Hybrid retrieval** — dense vector search + real BM25 sparse search, RRF-fused, then
  cross-encoder reranked; results carry full citation metadata.
- **Grounded Q&A** — answers with inline `[N]` citation markers, a resolved sources list, and a
  low-confidence refusal ("no relevant sources found") when retrieval comes up empty or weak.
- **Citation & verification** — metadata normalization via Crossref, falling back to OpenAlex
  then Semantic Scholar when Crossref has nothing (with a title-similarity check before trusting
  a fallback match), APA/MLA/Chicago/IEEE/Vancouver/BibTeX formatting (pluggable style registry),
  and a per-claim verifier that flags any citation whose passage doesn't support the claim. A
  successful resolution is persisted into the library (a payload-only update, no re-embedding) so
  it's instant and works offline afterward; a failed one is never persisted, so it keeps
  retrying on later renders instead of getting stuck.
- **Reference management & export** — bulk BibTeX export (whole library or selected papers,
  collision-safe keys), a Zotero push (`scholarmind/citations/zotero.py`), and a LaTeX/Overleaf
  export for Writing-agent drafts ([N] markers → `\cite{}`, plus a matching .bib).
- **Structured writing & novelty check** — the writing agent now takes a section-type
  (Related Work/Introduction/Discussion/Abstract), a multi-paper scope, and voice notes,
  and guarantees no uncited sentence reaches the output; a separate novelty/overlap agent
  (`scholarmind/agents/novelty.py`) gives an advisory (not plagiarism-verdict) read on how a
  draft overlaps with retrieved content, from the library and optionally external literature.
- **Orchestrator** — `chat` routes `ingest`, `ask`, and the five domain agents through one graph.
- **HTTP API** — `POST /ingest`, `POST /ask`, `GET /health`, served by `scholarmind serve`.
- **Web app** — a Streamlit UI (`scholarmind app`) for uploading PDFs and asking questions in a
  chat interface, built entirely on the same engine functions as the CLI/API.
- **Literature discovery & citation graph** — searches arXiv, Semantic Scholar, and OpenAlex by
  topic (`scholarmind/discovery/`), deduping against your library and across sources, with an
  ingest action (PDF when open-access, else a title+abstract record); a 1-hop citation graph
  (references/citing papers) via Semantic Scholar for any library paper, DOI, or S2 paper ID.
- **Multimodal ingestion** — tables, equations, and figures are extracted alongside body text
  (PyMuPDF layout-aware table/image detection, a regex equation heuristic), each stored as its
  own retrievable, citable chunk; figures keep an on-disk image path. A gated, configurable
  **Figure Q&A** (`scholarmind/agents/figures.py`) can send a figure's actual image to a
  vision-capable model, falling back to caption-only when none is configured.
- **Eval harness** — precision@k / recall@k / citation-faithfulness scoring over a labelled set
  (`scholarmind eval`), with centralized guardrails (confidence threshold, cite-only-ingested).

**326 tests** — 207 run fully offline with no API key (fast CI gate); the remaining 119 download
models or exercise live LLM/network paths.

## Quickstart

Works the same way on **macOS (incl. Apple Silicon), Linux, and Windows** — pure Python, no
platform-specific setup. (macOS: `brew install uv` or the curl installer below both work.)

```bash
# 1. Install dependencies (uv; falls back to pip). No uv yet?
#    macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
#    Windows:     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync --extra dev --extra webapp

# 2. Configure — copy the template and add an OpenRouter (or any OpenAI-compatible) key
cp .env.example .env
# then edit .env: set LLM_API_KEY=sk-or-...   (a free OpenRouter model is the default)

# 3. Ingest a PDF (or a directory of PDFs) — no PDF handy? use the bundled sample:
uv run scholarmind ingest tests/fixtures/sample_paper.pdf

# 4. Ask a question, grounded in what you ingested
uv run scholarmind ask "What does this paper propose for grounding LLM answers?"

# Or route everything through the orchestrator
uv run scholarmind chat "summarize retrieval-augmented generation"

# Serve the HTTP API
uv run scholarmind serve
```

Run `uv run scholarmind --help` to see all commands (`ingest`, `dedupe`, `delete`, `ask`, `chat`,
`serve`, `app`, `eval`).
See the [User Guide](USER_GUIDE.md) for the full command/config reference, macOS/Linux/Windows
install details, and a walkthrough.

### Web app

Prefer a browser to the CLI? `uv run scholarmind app` launches a Streamlit UI at
`http://localhost:8501` — a clean, themed, **multi-page** app (top navigation, shared sidebar)
over the same engine functions the CLI and API use — no separate logic. Ask questions in a chat
interface with a prominent green/red **verification badge**, sources, formatted references, and
flagged/unsupported claims shown inline.

> Embedded Qdrant is single-process: don't run the web app and `scholarmind serve` against the
> same `QDRANT_PATH` at the same time.

#### The pages

Five pages across the top nav, each focused on one intent:

| Page | What's on it |
|---|---|
| **Ask** | Chat, scoped to all papers or one. The answer **streams in token-by-token**, then a prominent verification badge and expandable Sources / Verification details / References appear once each cited claim has been checked. |
| **Library** | Upload & ingest, a two-step per-paper delete, and Figures & tables browsing (extracted tables/equations/figures, with per-figure Q&A that uses `VISION_MODEL` if configured). |
| **Analyze** | Tabs: **Summarize** (one paper in full, or by topic), **Gap analysis** (themes/contradictions across the library), **Methodology** (study-design/stats advice). |
| **Write** | Tabs: **Writing** (section-type picker, multi-paper scope, voice notes, never-uncited guarantee, + a novelty check), **References & export** (APA/MLA/Chicago/IEEE/Vancouver, `.bib`, Zotero, LaTeX bundle), **Citation / verify** (verify a pasted `[N]`-marked draft). |
| **Discover** | Tabs: **My library** (browse ingested papers), **Literature search** (arXiv + Semantic Scholar + OpenAlex, with ingest), **Citation graph** (1-hop references/citing via Semantic Scholar). |

**To summarize one paper, pick it from the dropdown — don't type the filename.** The engine has
no way to match a filename against paper content; a bare topic/filename search runs
library-wide semantic retrieval and can pull in unrelated chunks. The Ask panel has the same
single-paper picker as an optional scope filter (default: all papers); Writing has a
multi-paper picker instead (`paper_id` `MatchAny` over the selection) — Gap analysis and
Methodology remain library-wide only.

Summarize/Gap analysis/Methodology/Writing all show the same Sources and Verification panels as
Ask — the web app composes `verify_citations` + `format_and_verify` around each agent's raw
output itself, since the orchestrator's automatic citation verification only applies to the Ask
path.

**Writing never returns an uncited claim** — any sentence without a `[N]` marker is stripped
from the draft before it's shown (`writing._strip_uncited_sentences`), not just discouraged by
the prompt. A **Check novelty of this draft** button then runs an advisory (not
plagiarism-verdict) overlap check via the same retrieval+rerank pipeline as everything else,
optionally also searching arXiv/Semantic Scholar/OpenAlex.

**Discovery results merge duplicates across sources (by DOI, else normalized title) and flag
anything already in your library.** Ingesting a result downloads its open-access PDF when one
exists; otherwise it stores a lightweight title+abstract record instead of failing. Semantic
Scholar's free tier is rate-limited — set `S2_API_KEY` in `.env` to raise it. A source that's
down or rate-limited shows a warning next to whatever the other sources still returned, never a
traceback.

**Table/figure/equation extraction never blocks ingestion.** Every step (opening the PDF with
PyMuPDF, one table, one image, PyMuPDF being missing entirely) is individually wrapped, so a
paper with none of these — or a PDF PyMuPDF can't parse — still ingests with its body text
exactly as before, just with fewer extra chunks. Figure Q&A is opt-in: set `VISION_MODEL` to an
actual multimodal model (not a text-only one) to send the image itself; leave it empty and it
always answers from the caption alone, explicitly told not to guess at what it can't see.

**Zotero push writes only** (no read/import) and needs an API key + library ID, configured in
the sidebar for the current session or via `.env` — never commit a real key. An unconfigured
Zotero panel shows a note instead of a button; a rejected key/library ID shows a readable
"403 Forbidden" message.

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
| `S2_API_KEY` | *(empty)* | Optional Semantic Scholar key — raises the shared rate limit for literature discovery/citation-graph lookups |
| `ZOTERO_API_KEY` | *(empty)* | Optional — only needed to push references to Zotero |
| `ZOTERO_LIBRARY_ID` | *(empty)* | Your Zotero user or group library ID |
| `ZOTERO_LIBRARY_TYPE` | `user` | `user` or `group` |
| `VISION_MODEL` | *(empty)* | Optional vision-capable model for Figure Q&A to use the actual figure image instead of caption-only |

No API key is needed to ingest, retrieve, or run the offline test suite — only the LLM-backed
answer/verification/agent steps require one. `S2_API_KEY`, the `ZOTERO_*` variables, and
`VISION_MODEL` are all optional; each feature works without them, or is simply unavailable (or
falls back) in the UI until configured.

## Roadmap / build phases

All phases below are implemented and tested.

1. ✅ Project scaffold (config, CLI, packaging)
2. ✅ Ingestion pipeline (PDF → chunks → embedded Qdrant, idempotent)
3. ✅ Hybrid retrieval (dense + BM25, RRF, cross-encoder rerank)
4. ✅ Q&A RAG agent with inline citations + low-confidence refusal
5. ✅ LangGraph orchestrator (`chat`)
6. ✅ Citation & verification agent (Crossref, APA/BibTeX, claim verifier)
7. ✅ FastAPI + CLI
8. ✅ Eval harness & guardrails
9. ✅ Docs & release polish
10. ✅ Streamlit web app (`scholarmind app`)
11. ✅ Literature discovery (arXiv/Semantic Scholar/OpenAlex search + ingest) and a 1-hop
    citation graph (Semantic Scholar references/citations + ingest)
12. ✅ Reference management & export (MLA/Chicago/IEEE/Vancouver styles, bulk BibTeX export,
    Zotero push, LaTeX/Overleaf draft export)
13. ✅ Structured writing (section types, multi-paper scope, voice notes, no-uncited-claims
    guarantee) and an advisory novelty/overlap check
14. ✅ Multimodal ingestion (tables, equations, figures extracted alongside text, each a
    retrievable/citable chunk) and gated, configurable Figure Q&A
15. ✅ Library management — duplicate-paper detection (`scholarmind dedupe`), per-paper deletion
    (`scholarmind delete` + a web-app delete panel), and a Crossref → OpenAlex → Semantic Scholar
    metadata fallback chain (with self-healing, non-permanent negative caching) for references the
    PDF's own metadata can't supply
16. ✅ Performance — a shared, process-wide cache for the embedding/reranker models (loaded once,
    not on every ingest/search call) and persisted reference-metadata resolution (written back to
    the library on first success, so later use is instant and works offline)
17. ✅ Web-app UX overhaul — a multi-page layout (`st.navigation`: Ask / Library / Analyze /
    Write / Discover) instead of one long scroll, a prominent citation-verification badge,
    token-by-token streamed answers on the Ask page, a light professional theme
    (`.streamlit/config.toml`), and friendlier empty states

Future ideas: additional source connectors (HTML, DOI-direct fetch), a Zotero read/import path,
expanding the citation graph past one hop, a real web-search backend for the novelty check
(today it optionally checks arXiv/Semantic Scholar/OpenAlex, not the general web), and true
equation OCR/LaTeX recognition (today's equation extraction is a numbered-line + math-symbol
heuristic, not an ML-based recognizer).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) — including the test-marker convention and a list of
good first issues (new source connectors, new citation styles).

## License

[Apache-2.0](LICENSE).
