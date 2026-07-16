from scholarmind.agents.base import AgentResult, grounded_generate
from scholarmind.agents.llm_client import LLMClient
from scholarmind.config import Settings

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


def draft_section(
    query: str, llm_client: "LLMClient", settings: "Settings | None" = None
) -> AgentResult:
    return grounded_generate(query, SYSTEM_PROMPT, _TASK, llm_client, settings)
