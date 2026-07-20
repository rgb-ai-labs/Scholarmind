from scholarmind.agents.base import AgentResult, grounded_generate
from scholarmind.agents.llm_client import LLMClient
from scholarmind.citations.verify import build_source_block
from scholarmind.config import Settings, get_settings
from scholarmind.retrieval.papers import get_paper_chunks

SYSTEM_PROMPT = (
    "You are a research summarization assistant. Summarize ONLY the information "
    "present in the numbered sources provided below. Never add facts, context, or "
    "conclusions from outside knowledge, even if you believe them to be true. If the "
    "sources disagree or are incomplete on some point, reflect that uncertainty rather "
    "than resolving it yourself. Stay concise, use plain language, and remain strictly "
    "faithful to what the sources actually say."
)

_TASK = "Write a concise, faithful summary of what the sources report about the topic."

# Used when summarizing one specific paper (paper_id set): the sources passed to the LLM
# are every chunk of that paper in reading order, not a topic-driven top-k search.
_PAPER_SYSTEM_PROMPT = (
    "You are a research summarization assistant. The numbered sources below are chunks "
    "from a single paper, given in reading order. Using ONLY those sources, write a "
    "structured summary with these headings: Overview, Methods, Findings, Limitations. "
    "Cite every claim with the matching [N] marker. Never add facts from outside the "
    "sources. If a heading has nothing to report, say so briefly rather than inventing "
    "content."
)

_PAPER_TASK = "Write the structured summary now."

_MAP_SYSTEM_PROMPT = (
    "You are taking notes on one excerpt from a longer paper, to help a later step write "
    "a full summary of the whole paper. List the key facts, methods, and findings "
    "mentioned in this excerpt in plain prose. Stay strictly faithful to the excerpt — do "
    "not add outside knowledge."
)

# Chunks per map-step batch when a paper is too long to summarize in one LLM call.
_MAP_BATCH_SIZE = 12


def summarize(
    query: str,
    llm_client: "LLMClient",
    settings: "Settings | None" = None,
    paper_id: str | None = None,
) -> AgentResult:
    if paper_id is not None:
        return summarize_paper(paper_id, llm_client, settings)
    return grounded_generate(query, SYSTEM_PROMPT, _TASK, llm_client, settings)


def summarize_paper(
    paper_id: str, llm_client: "LLMClient", settings: "Settings | None" = None
) -> AgentResult:
    settings = settings or get_settings()

    chunks = get_paper_chunks(paper_id, settings)
    if not chunks:
        return AgentResult(text="", sources=[], sources_found=0)

    source_block = build_source_block(chunks)

    if len(chunks) <= _MAP_BATCH_SIZE:
        text = llm_client.complete(
            _PAPER_SYSTEM_PROMPT, f"{_PAPER_TASK}\n\nSources:\n{source_block}"
        )
        return AgentResult(text=text, sources=chunks, sources_found=len(chunks))

    # Map: summarize each batch of chunks (in reading order) into plain-prose notes.
    notes = []
    for start in range(0, len(chunks), _MAP_BATCH_SIZE):
        batch = chunks[start : start + _MAP_BATCH_SIZE]
        batch_block = build_source_block(batch)
        note = llm_client.complete(_MAP_SYSTEM_PROMPT, f"Excerpt:\n{batch_block}")
        notes.append(note)

    # Reduce: synthesize the notes into one structured summary, but ground and cite
    # against the FULL, original source list so citation numbers stay valid.
    combined_notes = "\n\n".join(notes)
    reduce_prompt = (
        f"{_PAPER_TASK}\n\n"
        "Section notes gathered from this paper (for your reference only — ground every "
        f"claim in the numbered Sources below, not the notes):\n{combined_notes}"
        f"\n\nSources:\n{source_block}"
    )
    text = llm_client.complete(_PAPER_SYSTEM_PROMPT, reduce_prompt)
    return AgentResult(text=text, sources=chunks, sources_found=len(chunks))
