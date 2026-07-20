from dataclasses import dataclass, field

from scholarmind.agents.llm_client import LLMClient
from scholarmind.citations.verify import build_source_block
from scholarmind.config import Settings, get_settings
from scholarmind.discovery.models import Candidate
from scholarmind.discovery.service import search_external
from scholarmind.retrieval.dense import DenseResult
from scholarmind.retrieval.search import search

SYSTEM_PROMPT = (
    "You are a novelty-assessment assistant, NOT a plagiarism detector. You are given a "
    "draft passage or contribution statement, and a numbered list of the most similar "
    "content found in the user's own research library (and sometimes external literature). "
    "Your job is to give a plain, advisory assessment of how the passage relates to that "
    "retrieved content — not a verdict on originality, and not an accusation of copying. "
    "Structure your response in three short parts: (1) what the passage overlaps with, "
    "naming the specific numbered source(s); (2) what seems to be the passage's distinct or "
    "novel angle relative to those sources, if any; (3) one or two concrete suggestions for "
    "what to check, clarify, or differentiate further. Ground every specific overlap claim in "
    "the numbered sources — do not claim an overlap with a source that doesn't support it, "
    "and do not speculate about sources you were not given. If nothing closely related was "
    "retrieved, say so plainly and do not invent overlaps."
)

_TASK = (
    "Assess how the passage below relates to the numbered sources found in the user's "
    "library. This is an advisory novelty/overlap check, not a plagiarism verdict."
)


@dataclass
class NoveltyCheckResult:
    library_overlaps: list[DenseResult]
    external_overlaps: list[Candidate] = field(default_factory=list)
    external_search_errors: list[str] = field(default_factory=list)
    assessment: str = ""
    sources_found: int = 0


def check_novelty(
    passage: str,
    llm_client: "LLMClient",
    settings: "Settings | None" = None,
    include_external_search: bool = False,
) -> NoveltyCheckResult:
    settings = settings or get_settings()

    library_overlaps = search(passage, settings)

    external_overlaps: list[Candidate] = []
    external_errors: list[str] = []
    if include_external_search:
        # Reuses the same arXiv/Semantic Scholar/OpenAlex discovery pipeline built for
        # literature search, as an approximation of "optionally a web search" — checking
        # external literature databases, not the general web, since that's the search
        # capability this engine actually has.
        query = passage.strip()[:300]
        discovery_result = search_external(query, settings)
        external_overlaps = discovery_result.candidates
        external_errors = discovery_result.errors

    if not library_overlaps and not external_overlaps:
        return NoveltyCheckResult(
            library_overlaps=[],
            external_overlaps=[],
            external_search_errors=external_errors,
            assessment="",
            sources_found=0,
        )

    source_block = build_source_block(library_overlaps) if library_overlaps else ""
    if external_overlaps:
        external_block = "\n\n".join(
            f"[external] {c.title or 'Untitled'} ({', '.join(c.authors) or 'unknown authors'}"
            f"{f', {c.year}' if c.year else ''}) — {c.source}: {c.abstract or 'no abstract'}"
            for c in external_overlaps
        )
        source_block = f"{source_block}\n\n{external_block}".strip()

    user_prompt = f"{_TASK}\n\nSources:\n{source_block}\n\nPassage:\n{passage}"
    assessment = llm_client.complete(SYSTEM_PROMPT, user_prompt)

    return NoveltyCheckResult(
        library_overlaps=library_overlaps,
        external_overlaps=external_overlaps,
        external_search_errors=external_errors,
        assessment=assessment,
        sources_found=len(library_overlaps),
    )
