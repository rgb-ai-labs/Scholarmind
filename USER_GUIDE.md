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

ScholarMind is pure Python with cross-platform dependencies (PyTorch, PyMuPDF, Qdrant, Streamlit
all ship macOS/Linux/Windows wheels) and an embedded, serverless vector store — there's nothing
platform-specific to install. It runs the same way on **macOS (including Apple Silicon), Linux,
and Windows**.

## Install

### macOS / Linux

```bash
# Install uv if you don't have it (skip if `uv --version` already works)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <your-fork-url> scholarmind
cd scholarmind

uv sync --extra dev --extra webapp    # runtime + dev deps + the Streamlit web app

cp .env.example .env                   # then edit .env and set LLM_API_KEY (see below)
```

On macOS you can also install uv with Homebrew: `brew install uv`. Apple Silicon (M-series) and
Intel Macs both work — `uv sync` picks the correct wheels automatically; there's no Rosetta step
and nothing to compile.

### Windows

```powershell
# Install uv if you don't have it (skip if `uv --version` already works)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

git clone <your-fork-url> scholarmind
cd scholarmind

uv sync --extra dev --extra webapp
copy .env.example .env
```

(Omit `--extra webapp` on either platform if you only want the CLI/API and not the browser UI.)

Verify the install with the fast, offline test suite (no key, no model download needed):

```bash
uv run pytest -m "not slow and not llm" -q     # ~207 tests, ~20s
uv run scholarmind --help                       # lists all commands
```

**Don't have a PDF handy?** The repo ships a small sample paper for exactly this — try the
[walkthrough](#full-walkthrough) below with `tests/fixtures/sample_paper.pdf` before hunting
down a real one.

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
| `S2_API_KEY` | *(empty)* | Optional Semantic Scholar key — raises the shared rate limit for discovery/citation-graph lookups |
| `ZOTERO_API_KEY` | *(empty)* | Optional — needed only to push references to Zotero from the web app |
| `ZOTERO_LIBRARY_ID` | *(empty)* | Your Zotero user or group library ID |
| `ZOTERO_LIBRARY_TYPE` | `user` | `user` or `group`, matching the library ID above |
| `VISION_MODEL` | *(empty)* | Optional vision-capable model for Figure Q&A to use the actual figure image, not just its caption — see [Multimodal content](#multimodal-content-tables-equations-and-figures) below |

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
- No API key required. Re-ingesting the *same* file updates it in place (idempotent, keyed by a
  content hash) rather than duplicating — safe to re-run any time, including just to backfill
  table/figure/equation chunks for a paper ingested before that feature existed.
- Also extracts tables, equations, and figures (saved as images) as their own retrievable,
  citable chunks — see [Multimodal content](#multimodal-content-tables-equations-and-figures).
  This never blocks ingestion: a paper with no extractable tables/figures/equations, or a
  broken/unsupported one, still ingests text-only exactly as before.
- **Duplicate-title warning:** if the paper's title normalizes to match a paper already in your
  library under a *different* ID (e.g. a re-downloaded or regenerated PDF whose bytes differ
  from the copy you already have), ingestion still proceeds but prints a warning — content-hash
  identity alone can't catch "same paper, different file." See [`dedupe`](#dedupe--find-and-remove-duplicate-papers)
  below to review and clean these up.

**Example**
```bash
uv run scholarmind ingest papers/rag_survey.pdf
```
**Output shape**
```
Ingested 1 paper(s), 5 chunk(s) into collection 'scholarmind_chunks'.
```
**Output shape (possible duplicate detected)**
```
Ingested 1 paper(s), 5 chunk(s) into collection 'scholarmind_chunks'.
Warning: 'A Study of Retrieval-Augmented Generation' already appears to be in your library under
a different ID — this may be a duplicate. Run `scholarmind dedupe` to review.
```

### `dedupe` — find and remove duplicate papers

```
scholarmind dedupe [--apply]
```
- No API key required. Groups every paper in your library by a normalized title (case/
  punctuation/whitespace-insensitive) and reports any group with more than one paper ID — this
  is the cleanup counterpart to the warning `ingest` prints when it detects one.
- **Dry-run by default** — prints which paper would be kept (the one with the most chunks;
  ties broken by earliest ingestion, then paper ID) and which would be removed, without deleting
  anything.
- `--apply` — actually deletes the redundant papers' chunks from the store. This is
  irreversible (the papers' vectors/text are gone; the original PDF files under your uploads
  folder are untouched, so you can always re-ingest).

**Example**
```bash
uv run scholarmind dedupe          # see what would change
uv run scholarmind dedupe --apply  # actually remove the duplicates
```
**Output shape**
```
'A Study of Retrieval-Augmented Generation for Scholarly Question Answering':
  keep:   128a0946c4d4... (5 chunk(s))
  remove: 89541c6b53df... (5 chunk(s))

Dry run - 1 duplicate group(s) found. Re-run with --apply to remove them.
```
```
Removed 5 chunk(s) across 1 duplicate paper(s).   # after --apply
```
**No duplicates found**
```
No duplicate papers found.
```

### `delete` — remove one ingested paper

```
scholarmind delete IDENTIFIER [--yes]
```
- No API key required. Removes a single paper from your library by `IDENTIFIER`, resolved in a
  predictable order: an **exact paper ID** first, then a **paper-ID prefix** (the short IDs shown
  by `dedupe`/`delete`), then a **case-insensitive substring of the title**.
- **Dry-run by default** — prints the matched paper without deleting. Add `--yes` (or `-y`) to
  actually remove it. Deletion takes out the paper's chunks from the search index; the **original
  PDF in your uploads folder is left untouched**, so you can always re-ingest it.
- **Safe on ambiguity** — if the identifier matches more than one paper, it lists them and deletes
  nothing, asking you to be more specific or use an exact ID. If it matches nothing, it says so
  and exits non-zero.
- Prefer a browser? The web app has a **"Your library — manage & delete"** panel with a per-paper
  delete button and a confirmation step (see [Web app](#web-app) below).

**Example**
```bash
uv run scholarmind delete "retrieval-augmented"        # dry-run preview by title match
uv run scholarmind delete 89541c6b53df --yes           # delete by ID prefix, for real
```
**Output shape (dry-run)**
```
89541c6b53df...  A Study of Retrieval-Augmented Generation for Scholarly Question Answering (5 chunk(s))

Dry run - re-run with --yes to delete this paper. (Its PDF in the uploads folder is kept, so you
can re-ingest it later.)
```
**Output shape (deleted)**
```
Deleted 'A Study of Retrieval-Augmented Generation for Scholarly Question Answering' (5 chunk(s) removed).
```
**Output shape (ambiguous — nothing deleted)**
```
'retrieval' matched 2 papers — be more specific, or use an exact paper ID:
  89541c6b53df...  A Study of Retrieval-Augmented Generation... (5 chunk(s))
  a1b2c3d4e5f6...  Retrieval Methods in Practice (3 chunk(s))
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

### `app` — web UI

```
scholarmind app
```
- Launches a Streamlit web app at `http://localhost:8501` (runs `streamlit run
  scholarmind/webapp/app.py` under the hood). Upload PDFs and ask questions from a browser — it
  calls the same `run_ingestion`, `answer_question`, and `format_and_verify` functions the CLI
  uses, so it always sees the same library as `ingest`/`ask`/`chat`.
- No flags. Stop it with Ctrl-C in the terminal it's running in.
- **Embedded Qdrant is single-process** — don't run `app` and `serve` (or two `app`/`serve`
  instances) against the same `QDRANT_PATH` at the same time; the second one will fail to open
  the store.

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
uv sync --extra dev --extra webapp
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

## Web app

Prefer a browser? Same engine, no separate setup:

```bash
uv sync --extra dev --extra webapp   # once, to install Streamlit
uv run scholarmind app                # -> opens http://localhost:8501
```

The app is organized as **five pages** across a top navigation bar, with a shared sidebar. Pick
a page from the top nav; the sidebar (config + library stats + Zotero) stays visible everywhere.

- **Sidebar** — the app name/brand, your current config (LLM model, embedding model, Qdrant
  path), how many papers/chunks are indexed, and Zotero settings. If `LLM_API_KEY` isn't set,
  paste one here for the session (used only in memory, never written to `.env`).
- **Ask** (default page) — pick a **Scope** (default "All papers", or one specific ingested
  paper) and type a question in the chat box. After a brief "Searching your library…" while
  retrieval runs, the answer **streams in token-by-token** (with its inline `[N]` markers);
  then each cited claim is checked and a prominent **verification badge** appears (green "all N
  claims verified" or red "M unsupported"), alongside an expandable **Sources** panel,
  **Verification details**, and **References (APA/BibTeX)**. If retrieval finds nothing, you get
  a friendly "No relevant sources found" box, not an error.
- **Library** — everything about the papers you already have: **Upload & ingest** (drag PDFs,
  click ingest; each file gets a success/error line, never a raw traceback), a **manage & delete**
  expander (per-paper **Delete** with a deliberate two-step confirm — the first click arms it,
  only **Confirm delete** removes it; the PDF in the uploads folder is kept, so it's
  re-ingestable), and **Figures & tables** browsing.
- **Analyze** — reading/understanding tools as tabs: **Summarize**, **Gap analysis**,
  **Methodology**.
- **Write** — producing cited output as tabs: **Writing** (+ novelty check), **References &
  export**, **Citation / verify**.
- **Discover** — finding new papers as tabs: **My library** (browse ingested papers by topic),
  **Literature search** (arXiv/Semantic Scholar/OpenAlex), **Citation graph**.

The web app is a thin UI over `run_ingestion`, `answer_question`, and `format_and_verify` — the
exact same functions the CLI calls, reading and writing the exact same `QDRANT_PATH`, so
whatever you ingest via `scholarmind ingest` shows up here and vice versa.

> Embedded Qdrant is single-process — don't run the web app and `scholarmind serve` against the
> same `QDRANT_PATH` at the same time.

### The agent tools (Analyze / Write / Discover pages)

The agent tools live on the **Analyze**, **Write**, and **Discover** pages (as tabs within each).
All of them call the exact functions `chat`/`scholarmind/agents/` use — there's no separate UI
logic.

- **Summarize** (`summarize` / `summarize_paper`) — pick a paper from the dropdown to summarize
  **just that paper**: the engine pulls every one of its chunks in reading order (not a topic
  search) and produces a structured Overview/Methods/Findings/Limitations summary, cited only
  against that paper's own sources. **To summarize one paper, pick it from the dropdown — don't
  type its filename into a search box**; the engine has no way to match a filename against
  content, so typing one into a free-text field returns unrelated chunks. Switch to
  "Across whole library (by topic)" for the old free-text, topic-driven mode.
- **Gap analysis** (`analyze_gaps`) — synthesizes across your whole library for themes,
  contradictions, and open questions. Type e.g. `reinforcement learning from human feedback`.
- **Methodology** (`extract_methodology`) — free-text advice on study design/statistics. Type
  e.g. `What sample size justification do these papers use?`.
- **Writing** (`draft_section`) — see
  [Writing and the novelty check](#writing-and-the-novelty-check) below: a real section-type
  picker (Related Work / Introduction / Discussion / Abstract), a multi-paper scope, optional
  voice notes, and a "never emit an uncited claim" guarantee — plus an advisory novelty/overlap
  check on the resulting draft.
- **Discover** — three sub-tabs (see [Literature discovery and the citation graph](#literature-discovery-and-the-citation-graph)
  below): **My library** (unchanged — browses papers already ingested, no network call),
  **Literature search** (external arXiv/Semantic Scholar/OpenAlex search with an ingest
  action), and **Citation graph** (references/citing papers for one paper, one hop out).
- **Citation / verify** — paste any `[N]`-marked draft plus its numbered source list (one line
  each: `N | title | authors | year | section | pages`) to run the same `verify_citations` +
  `format_and_verify` pipeline Ask uses, without needing a fresh retrieval.
- **References & export** — see
  [Reference management and export](#reference-management-and-export) below: pick a citation
  style, export a `.bib` file, push references to Zotero, or export a Writing-agent draft as a
  LaTeX/Overleaf bundle.
- **Figures & tables** (on the **Library** page, not an agent tool) — see
  [Multimodal content](#multimodal-content-tables-equations-and-figures) below: browse one
  paper's extracted tables (rendered), equations, and figures (with thumbnails), and ask a
  question about a specific figure.

**Paper scoping:** Ask and Summarize show a single-paper picker; Writing shows a multi-paper
picker (see below). All are backed by `retrieval.papers.list_papers()`, which lists every
distinct `paper_id` ingested so far, labeled with the paper's title when the PDF had `/Title`
metadata, falling back to the source filename when it didn't (many PDFs — especially arXiv
downloads — have no title metadata, so you'll often see a filename like `1036663.pdf` in the
picker; that's expected, not a bug). Selecting paper(s) applies a real Qdrant payload filter
(`paper_id` match, or `paper_id` in a list for Writing's multi-select) before retrieval, so only
those papers' chunks are ever searched. Gap analysis, Methodology, and Discover are unscoped —
they always run across the whole library.

Summarize/Gap analysis/Methodology/Writing show the same **Sources** and **Verification**
expanders as Ask. The orchestrator's `chat`/`run()` path only auto-runs citation verification
for `ask` (see `orchestrator/graph.py`'s `final` node) — the web app calls `verify_citations`
then `format_and_verify` itself on each agent's raw output to match that behavior, reusing the
same functions rather than adding new citation logic.

### Literature discovery and the citation graph

`discover` (`scholarmind/agents/discovery.py`) only ever browses papers you've already
ingested — it makes no network call. Finding and pulling in *new* papers is a separate,
explicitly-named capability in `scholarmind/discovery/` (arXiv, Semantic Scholar, and
OpenAlex clients + dedupe + a Semantic Scholar–backed citation graph), exposed as two more
sub-tabs under **Discover**:

- **Literature search** — type a topic; it queries all three sources (arXiv's Atom API,
  Semantic Scholar's `/paper/search`, OpenAlex's `/works?search=`), merges results
  that are clearly the same paper (same DOI, or the same title after normalizing), and flags any
  result that matches a paper already in your library (by DOI, else by a normalized-title hash)
  with "Already in library" instead of an ingest button. Each remaining result shows its
  source(s) (arXiv/Semantic Scholar/OpenAlex), authors, year, venue, and an expandable abstract,
  with an **Ingest** button.
  - If the result has an open-access PDF URL, ingesting downloads and runs it through the same
    `run_ingestion` pipeline as an upload.
  - If it doesn't — or the PDF download fails or isn't actually a PDF — it's ingested as a
    **lightweight, metadata-only record** instead: just the title and abstract, chunked and
    embedded like any other paper, so it's fully searchable and citable, but visibly limited to
    what the abstract says. The library picker (used by Ask/Summarize) shows a
    `PaperSummary.is_metadata_only` flag for these.
- **Citation graph** — pick a library paper (its stored DOI is used automatically if present),
  or type a DOI or Semantic Scholar paper ID directly, then fetch **References** (what it cites)
  and **Cited by** (what cites it), one hop out, via Semantic Scholar's `/paper/{id}/references`
  and `/paper/{id}/citations` endpoints. If no S2 ID/DOI is given, it resolves one via a
  best-effort title search — treat that resolution as approximate, not exact. Each entry shows
  the same ingest action as Literature search. OpenAlex is not used for the citation graph itself
  (only for search) since Semantic Scholar's paperId-keyed endpoints are simpler and sufficient
  for a 1-hop graph.

**Rate limits:** arXiv and OpenAlex don't require a key for this usage. Semantic Scholar's
unauthenticated tier is shared and easy to rate-limit (`429`); set `S2_API_KEY` in `.env` (free,
from Semantic Scholar) to raise it. Any source that's unreachable or rate-limited shows a
readable warning next to the results from the sources that did work — one source failing never
blocks the others or surfaces a traceback.

### Writing and the novelty check

The **Writing** tab (`scholarmind/agents/writing.py`) drafts a section grounded ONLY in
retrieved chunks:

- **Section type** — a real picker with four options, each with its own prompt (not just a
  free-text hint folded into one generic query, like the other agents): **Related Work**
  (groups sources by theme, calls out disagreement between them), **Introduction**
  (frames background/context/open questions — never invents the user's own unstated
  contribution), **Discussion** (contrasts and interprets sources against each other), and
  **Abstract** (a concise, cited summary of what the retrieved literature says — this is a
  summary of *the retrieved sources*, not the user's own unpublished paper, since the engine
  has no access to results the user hasn't given it).
- **Scope** — a multi-select paper picker (leave empty for the whole library). Internally this
  is a real Qdrant `paper_id` filter using `MatchAny` over the selected papers (the same
  mechanism Ask/Summarize use for a single paper, extended to a list) — not a soft nudge.
- **Voice notes** (optional) — free text folded into the prompt as a style instruction (e.g.
  "keep it formal, under 200 words"), explicitly told not to come at the cost of dropping a
  citation.
- **Never emit an uncited claim** — after generation, every sentence is checked for a `[N]`
  marker; any sentence with none is silently dropped from what you see. This is an actual
  guarantee enforced in code (`writing._strip_uncited_sentences`), not just a prompt
  instruction — though the LLM is also instructed this way, since dropping bad output after
  the fact is a worse experience than the model getting it right. If everything gets dropped
  (the model didn't cite anything), you'll see a message asking you to rephrase or widen the
  scope rather than an empty draft with no explanation.
- **Verification, reused as-is** — the resulting draft (whatever survives the uncited-sentence
  filter) is run through the exact same `verify_citations` → `format_and_verify` pipeline as
  every other agent panel, showing **Sources** and **Verification** (which citations are
  actually supported by their passage, via `citations/verifier.py`'s existing claim verifier —
  no new verification logic).

**Novelty / overlap check** (`scholarmind/agents/novelty.py`) — once you have a draft, a
**Check novelty of this draft** button appears below it. This is explicitly **advisory, not a
plagiarism verdict** (labeled as such in the UI): it retrieves the most similar content to your
draft via the same hybrid retrieval + rerank pipeline (`retrieval.search`) used everywhere else
in the app — the reranker's relevance score doubles as the similarity signal, so no separate
embedding-similarity system was built — and asks the LLM for a three-part plain-language read:
what the draft overlaps with (naming the specific source), what looks like its distinct angle,
and one or two things to check or differentiate further. Check **"Also check external
literature"** to additionally search arXiv/Semantic Scholar/OpenAlex (the same discovery search
built for literature search) as an approximation of a web search — real general web search isn't
implemented, since this engine's only external search capability is over literature databases.
Overlaps are shown as an expandable list with sources (and, for external results, which
database they came from); any external source that's unavailable shows a warning without
blocking the rest of the check.

### Reference management and export

The **References & export** tab (`scholarmind/citations/`) covers formatting, bulk export, a
Zotero push, and a LaTeX/Overleaf export for a Writing-agent draft. Nothing here re-implements
citation logic — it all goes through `citations/formatter.py`'s existing pluggable style
registry (`format_reference(metadata, style)`), extended with four more styles.

**Metadata resolution (Crossref → OpenAlex → Semantic Scholar):** the first time a reference is
shown here, it's resolved *live* — not just from whatever the PDF itself had embedded (many
LaTeX-built PDFs, e.g. ACM/IEEE templates, embed a `/Title` but no `/Author` at all).
`citations/metadata.py::normalize_metadata` first queries Crossref by title; if that fails or
scores too low, it falls back to OpenAlex, then Semantic Scholar (the same clients literature
discovery uses), each checked against a title-similarity sanity check before being trusted.
- **A *successful* resolution is written back into the library** (`retrieval/papers.py::
  persist_resolved_metadata` — a payload-only update, no re-embedding), so every later render of
  that paper's reference is instant and needs no network call at all, including with no internet
  connection.
- **A failed lookup is never persisted** — a paper not yet indexed anywhere, or hit during a
  transient network hiccup, is retried live on every render rather than being stuck forever, so
  it self-heals on its own once the data becomes available, with no re-ingestion or restart
  needed.
- If every source genuinely has nothing, you still get the paper's own title (and any
  author/year the PDF itself had) rather than a hard "Unknown Author (n.d.)" wherever the PDF
  provided something usable.

- **Citation styles** — `APA` and `BibTeX` already existed; this adds **MLA**, **Chicago**
  (author-date), **IEEE**, and **Vancouver**, all in the same `_FORMATTERS` registry. Pick a
  style from the dropdown and every reference below re-renders in it. These are simplified,
  journal-article-shaped formatters (no special handling for books, conference proceedings,
  etc.) — the same level of simplification the original APA formatter already used.
- **BibTeX export** — check specific papers, or leave none checked to export your whole
  library, and click **Export .bib**. Keys are generated the same way as a single-reference
  BibTeX export (`familyname` + `year`, e.g. `vaswani2017`); when two papers would collide
  (same first author, same year), the export appends `a`, `b`, … to disambiguate so the file
  stays valid.
- **Zotero push** — configure a Zotero API key and library ID in the sidebar (session-only by
  default; set `ZOTERO_API_KEY`/`ZOTERO_LIBRARY_ID` in `.env` to make it permanent — **never
  commit real values to `.env`**). Selected (or all) references are pushed as `journalArticle`
  items via Zotero's Web API (`POST /users|groups/{id}/items`, batched at 50 items per
  request, the API's own limit). Read access isn't implemented — write is. If Zotero isn't
  configured, the push button is replaced with a note instead of failing; a bad key/library ID
  shows a readable "403 Forbidden" message, not a traceback.
- **LaTeX / Overleaf export** — after drafting something in the **Writing** tab, switch to
  **References & export**: it remembers the most recent draft's text and citations, replaces
  every `[N]` (and combined `[N, M]`) marker with `\cite{key}` (skipping any marker that
  didn't resolve to a real citation, rather than silently deleting it), and generates a
  matching `.bib` file with the same keys. Both files are zipped into one download —
  **upload both files from the zip to Overleaf** (or any local LaTeX install with `references.bib`
  next to the `.tex` file). The draft body is TeX-escaped first (`%`, `&`, `$`, `_`, `{`, `}`,
  `~`, `^`, backslash) so the export doesn't produce invalid LaTeX if the draft happens to
  contain any of those characters.

### Multimodal content (tables, equations, and figures)

Ingestion (`scholarmind/ingestion/multimodal.py`, wired into `parser.py`/`chunker.py`) also
extracts tables, equations, and figures from each PDF, alongside its body text — using
**PyMuPDF** (already a dependency) for layout-aware table and image detection, plus a
regex-based heuristic for equations. Each becomes its own retrievable, citable chunk tagged
with a `chunk_type` (`"text"` / `"table"` / `"equation"` / `"figure"`):

- **Tables** — PyMuPDF's built-in table detector (`page.find_tables()`) converts each detected
  table to Markdown, with a caption pulled from the nearest "Table N: ..." line on the same
  page. Stored and embedded as `Table (page P): <caption>\n\n<markdown>`, so it's searchable
  by topic and citable like any text chunk — `[N]` in an answer can point straight at a table,
  and the **Sources** panel renders it as an actual table, not just a page reference.
- **Equations** — a conservative heuristic, not true equation OCR/LaTeX recognition: a line is
  flagged as an equation only if it both ends with a parenthesized number (`"... (3)"`, the
  common numbering convention) and contains a math symbol (`=`, `≤`, `∑`, Greek letters, etc.).
  The equation text plus a sentence of surrounding context becomes its own chunk. This
  deliberately favors missing an equation over flagging ordinary numbered prose — expect some
  real equations (especially unnumbered ones) to be missed.
- **Figures** — every embedded raster image is saved to disk (under a per-paper folder next to
  your Qdrant store) and paired with a caption found the same way as tables ("Figure N: ...").
  The chunk's embedded/searchable text is the caption; the image file path is carried in
  metadata (`image_path`) for rendering and Figure Q&A. A page with several figures may
  occasionally attach the wrong caption to an image — this is a heuristic nearest-caption
  match, not true layout-proximity detection.
- **Robustness** — every extraction step (opening the PDF with PyMuPDF, detecting tables on
  one page, saving one image, PyMuPDF being unavailable at all) is individually wrapped so a
  failure only means *fewer* tables/figures/equations extracted, never a failed ingestion. A
  paper with no extractable tables, equations, or figures — or a corrupt/unusual PDF that
  PyMuPDF can't open at all — still ingests with its body text exactly as before.

**Figure Q&A** (`scholarmind/agents/figures.py`) lets you ask a question about one specific
figure, from the **Figures & tables** tab. This is explicitly gated by config, not automatic
intent detection — you pick the figure yourself:

- If `VISION_MODEL` is set in `.env` **and** the figure's image file still exists on disk, the
  question, the caption, and the actual image are sent to that model (via the same
  `LLM_API_KEY`/`LLM_BASE_URL`, using an OpenAI-compatible multimodal request — an OpenRouter
  Gemini vision model, e.g. `google/gemini-2.0-flash-001`, is a reasonable choice; it does not
  need to be a "flash-lite"/text-only variant, since those can't accept image input).
- Otherwise — `VISION_MODEL` unset, the image file missing, or the vision call raising any
  error — it falls back to a **caption-only** answer using your regular `LLM_MODEL`, with a
  prompt that explicitly tells the model it only has the caption, not the image, so it doesn't
  guess at visual content it can't see.

Set `VISION_MODEL=` (empty) to always use caption-only Figure Q&A.

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
  to the configured LLM provider. Ingestion, embedding, and retrieval are fully local. Reference
  metadata resolution queries Crossref, then OpenAlex, then Semantic Scholar (all public APIs) by
  paper title — once per paper; a successful result is saved into your local library so later use
  needs no network call, including fully offline.
- **Why does a reference show "Unknown Author (n.d.)"?** The first time a reference is shown, it's
  resolved live against Crossref/OpenAlex/Semantic Scholar by title (see [Reference management and
  export](#reference-management-and-export)) — this only happens when the paper's title genuinely
  isn't indexed anywhere yet (a very new preprint, an obscure workshop paper) *and* the PDF itself
  had no usable author/year metadata either. An unresolved lookup is deliberately never saved, so
  it's retried on every render and can resolve itself later without you doing anything.
- **How do I clean up a paper that shows up twice in the library?** Run `scholarmind dedupe` to
  see duplicate groups (same title, different ID — usually a re-downloaded or regenerated PDF),
  then `scholarmind dedupe --apply` to remove the extra copy. See
  [`dedupe`](#dedupe--find-and-remove-duplicate-papers) above.
- **Can I use OpenAI / a local model instead of OpenRouter?** Yes — set `LLM_BASE_URL`,
  `LLM_MODEL`, and `LLM_API_KEY` to any OpenAI-compatible endpoint.
- **How do I add another citation style or file format?** See `CONTRIBUTING.md` — the citation
  formatter uses a pluggable registry, and the ingestion loader is the place for new formats.
- **Where's the architecture overview?** [`ARCHITECTURE.md`](ARCHITECTURE.md).
