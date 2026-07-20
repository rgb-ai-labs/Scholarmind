import re

from scholarmind.agents.base import AgentResult, grounded_generate
from scholarmind.agents.llm_client import LLMClient
from scholarmind.citations.verify import extract_citation_markers
from scholarmind.config import Settings

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Default section type (used when the caller doesn't ask for one of the named section types
# below) — kept byte-for-byte identical to the original prompt so existing callers (the
# orchestrator's "write "/"draft " chat routing, and any code calling draft_section with just
# a query) see unchanged behavior.
SYSTEM_PROMPT = (
    "You are an academic writing assistant drafting literature-review prose. Use ONLY the "
    "numbered sources provided below; never introduce facts, findings, or claims that are not "
    "explicitly present in those sources. Every sentence that makes a substantive claim MUST "
    "end with an inline citation marker in the form [N], where N matches the number of the "
    "source it is drawn from. Do not cite a source number that was not provided. Synthesize "
    "the sources into a coherent, well-structured paragraph rather than a list, but do not add "
    "transitions, background, or conclusions that rely on information outside the sources. If "
    "the sources are insufficient to cover an aspect of the topic, omit that aspect rather than "
    "filling the gap from memory."
)

_TASK = (
    "Draft a well-structured literature-review paragraph on the topic below, synthesizing the "
    "numbered sources into coherent academic prose. Cite the numbered sources inline with [N] "
    "markers for every claim, using only the information contained in the sources."
)

_RELATED_WORK_SYSTEM_PROMPT = (
    "You are an academic writing assistant drafting a Related Work section. Use ONLY the "
    "numbered sources provided below; never introduce facts, findings, or claims that are not "
    "explicitly present in those sources. Every sentence that makes a substantive claim MUST "
    "end with an inline citation marker in the form [N]. Group the sources by theme or "
    "approach rather than listing them in order, and where sources take different or "
    "conflicting positions, say so explicitly rather than flattening the disagreement. Do not "
    "cite a source number that was not provided. If the sources are insufficient to cover an "
    "aspect of the topic, omit that aspect rather than filling the gap from memory."
)
_RELATED_WORK_TASK = (
    "Draft a Related Work section on the topic below, organizing the numbered sources by "
    "theme or approach. Cite the numbered sources inline with [N] markers for every claim, "
    "using only the information contained in the sources."
)

_INTRODUCTION_SYSTEM_PROMPT = (
    "You are an academic writing assistant drafting the background/motivation portion of an "
    "Introduction section. Use ONLY the numbered sources provided below; never introduce "
    "facts, findings, or claims that are not explicitly present in those sources. Every "
    "sentence that makes a substantive claim MUST end with an inline citation marker in the "
    "form [N]. Frame the sources as establishing the context, prior approaches, and open "
    "questions for the topic. Do not propose or describe the user's own unstated contribution "
    "— you only have access to the retrieved literature, not the user's own results. Do not "
    "cite a source number that was not provided."
)
_INTRODUCTION_TASK = (
    "Draft an Introduction-style background paragraph on the topic below: what the numbered "
    "sources establish, what approaches they take, and what open questions or gaps they leave. "
    "Cite the numbered sources inline with [N] markers for every claim."
)

_DISCUSSION_SYSTEM_PROMPT = (
    "You are an academic writing assistant drafting a Discussion section that interprets and "
    "contrasts prior work. Use ONLY the numbered sources provided below; never introduce "
    "facts, findings, or claims that are not explicitly present in those sources. Every "
    "sentence that makes a substantive claim MUST end with an inline citation marker in the "
    "form [N]. Explicitly note points of agreement, disagreement, or unresolved tension among "
    "the sources rather than treating them as a uniform consensus. Do not cite a source number "
    "that was not provided."
)
_DISCUSSION_TASK = (
    "Draft a Discussion-style synthesis of the numbered sources on the topic below, "
    "contrasting their positions and interpretations. Cite the numbered sources inline with "
    "[N] markers for every claim."
)

_ABSTRACT_SYSTEM_PROMPT = (
    "You are an academic writing assistant drafting a concise, abstract-style summary of what "
    "the retrieved literature says about a topic. Use ONLY the numbered sources provided "
    "below; never introduce facts, findings, or claims that are not explicitly present in "
    "those sources, and never invent results as if they belonged to an unpublished paper of "
    "the user's own — this summarizes the retrieved literature, not the user's own unstated "
    "work. Every sentence that makes a substantive claim MUST end with an inline citation "
    "marker in the form [N]. Keep it to one concise paragraph. Do not cite a source number "
    "that was not provided."
)
_ABSTRACT_TASK = (
    "Draft a concise, single-paragraph, abstract-style summary (roughly 150-250 words) of "
    "what the numbered sources report about the topic below. Cite the numbered sources inline "
    "with [N] markers for every claim."
)

_SECTION_PROMPTS: dict[str, tuple[str, str]] = {
    "related_work": (_RELATED_WORK_SYSTEM_PROMPT, _RELATED_WORK_TASK),
    "introduction": (_INTRODUCTION_SYSTEM_PROMPT, _INTRODUCTION_TASK),
    "discussion": (_DISCUSSION_SYSTEM_PROMPT, _DISCUSSION_TASK),
    "abstract": (_ABSTRACT_SYSTEM_PROMPT, _ABSTRACT_TASK),
}

SECTION_TYPES = sorted(_SECTION_PROMPTS)


def _prompts_for(section_type: str | None, voice_notes: str | None) -> tuple[str, str]:
    if section_type is None:
        system_prompt, task = SYSTEM_PROMPT, _TASK
    else:
        key = section_type.strip().lower().replace(" ", "_")
        if key not in _SECTION_PROMPTS:
            raise ValueError(
                f"Unknown section type: {section_type!r}. Choose one of: {SECTION_TYPES}"
            )
        system_prompt, task = _SECTION_PROMPTS[key]

    if voice_notes and voice_notes.strip():
        task = (
            f"{task}\n\nStyle/voice notes to follow (without violating the citation rules "
            f"above — never drop a citation to satisfy a style note): {voice_notes.strip()}"
        )
    return system_prompt, task


def _strip_uncited_sentences(text: str) -> str:
    # Enforces "never emit an uncited claim" as an actual guarantee rather than just a prompt
    # instruction: any sentence with no [N] marker at all is dropped from the returned draft.
    # (verify_citations/verify_claim_support, run separately by the caller, catch the other
    # failure mode — a citation that exists but is unsupported or points at a fabricated [N].)
    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    cited = [s for s in sentences if extract_citation_markers(s)]
    return " ".join(cited)


def draft_section(
    query: str,
    llm_client: "LLMClient",
    settings: "Settings | None" = None,
    section_type: str | None = None,
    paper_ids: list[str] | None = None,
    voice_notes: str | None = None,
) -> AgentResult:
    system_prompt, task = _prompts_for(section_type, voice_notes)
    result = grounded_generate(
        query, system_prompt, task, llm_client, settings, paper_ids=paper_ids
    )

    if not result.text:
        return result

    cleaned_text = _strip_uncited_sentences(result.text)
    return AgentResult(
        text=cleaned_text, sources=result.sources, sources_found=result.sources_found
    )
