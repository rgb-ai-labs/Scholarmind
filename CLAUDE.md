# ScholarMind — Architecture Notes

Context for future sessions working on this codebase.

## 4-Layer Architecture

1. **Interaction layer** (`scholarmind/cli.py`, `scholarmind/api/`) — Typer CLI and FastAPI
   HTTP API. Thin: parses input, calls the orchestrator, formats output. No business logic.
2. **Orchestration layer** (`scholarmind/orchestrator/`) — a LangGraph graph with a supervisor
   node that routes a request to the right agent(s) and assembles their outputs into a
   final response.
3. **Agent layer** (`scholarmind/agents/`) — eight specialized agents, each a LangGraph node
   or subgraph with a narrow responsibility (below).
4. **Knowledge & retrieval layer** (`scholarmind/ingestion/`, `scholarmind/retrieval/`,
   `scholarmind/citations/`) — LlamaIndex-based document processing and hybrid search over a
   local, embedded Qdrant store (`QdrantClient(path=...)`, no server, no Docker).

Each layer only calls downward (interaction → orchestration → agent → knowledge/retrieval).
Agents never talk to Qdrant directly — they go through the retrieval layer.

## The 8 Agents

1. **Supervisor** — routes incoming requests to the correct agent(s); merges multi-agent output.
2. **Discovery** — finds candidate papers/sources for a research question (external search).
3. **Ingestion** — drives the parse → chunk → tag → embed → store pipeline for new documents.
4. **Q&A** — answers direct questions using retrieval + generation.
5. **Summarization** — produces summaries of papers or groups of papers.
6. **Gap-analysis** — identifies open questions / under-explored areas across the corpus.
7. **Citation** — resolves, formats, and verifies citations against source documents.
8. **Methodology** — extracts and compares methodology sections across papers.
9. **Writing** — drafts prose (e.g. literature review sections) grounded in retrieved sources.

(Eight specialized agents beyond the supervisor: discovery, ingestion, Q&A, summarization,
gap-analysis, citation, methodology, writing.)

## RAG Pipeline

**Ingestion (write path):** parse → chunk → tag → embed → store
- *Parse*: extract text from source documents (PDF, HTML, etc.) via LlamaIndex readers.
- *Chunk*: split parsed text into retrieval-sized units, preserving section/page metadata.
- *Tag*: attach metadata (source, authors, section, page) used for filtering and citation.
- *Embed*: encode chunks with a local `sentence-transformers` model.
- *Store*: upsert vectors + metadata into the embedded Qdrant collection.

**Query (read path):** hybrid-retrieve → rerank → generate-with-citations
- *Hybrid-retrieve*: combine dense vector search with keyword/BM25-style filtering.
- *Rerank*: reorder candidates with a local cross-encoder reranker for precision.
- *Generate-with-citations*: the LLM answers using only the reranked context, and every
  claim must be traceable to a retrieved chunk's metadata.

## Golden Rule: Never Cite From Memory

Agents must **never** produce a citation from the LLM's parametric memory. Every citation
emitted by the citation agent (or embedded in a Q&A/writing/summarization answer) must be
traced back to a chunk actually returned by the retrieval layer for that request, and
verified against the chunk's stored metadata before being surfaced to the user. If retrieval
returns nothing relevant, the correct behavior is to say so — not to fabricate a source.

## Build Phase Order

1. **Scaffold** (this phase) — project structure, config, CLI/API stubs, no logic.
2. **Ingestion pipeline** — parse/chunk/tag/embed/store working end-to-end for a single
   local document, with tests.
3. **Retrieval pipeline** — hybrid-retrieve + rerank working against the ingested index.
4. **Q&A agent + citation agent** — the first full generate-with-citations path.
5. **Orchestrator + supervisor** — LangGraph wiring so CLI/API route through the supervisor
   instead of calling agents directly.
6. **Remaining agents** — discovery, summarization, gap-analysis, methodology, writing.
7. **API hardening** — FastAPI endpoints, error handling, streaming responses.
8. **Polish** — docs, examples, packaging for external contributors.

Do not skip ahead to later phases before the current phase's tests pass.
