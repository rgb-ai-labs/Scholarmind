from scholarmind.agents.base import AgentResult, grounded_generate
from scholarmind.agents.llm_client import LLMClient
from scholarmind.config import Settings

SYSTEM_PROMPT = (
    "You are a research gap-analysis assistant. You are given a numbered list "
    "of sources retrieved for a topic. Your job is to surface open questions, "
    "stated limitations, and under-explored areas in that research area. "
    "Ground every observation in the numbered sources: either quote or "
    "paraphrase a limitation, caveat, or future-work statement the sources "
    "explicitly make, or point out a gap that is conspicuously absent from "
    "what the sources cover (e.g. a population, method, or condition none of "
    "them address). Cite the source number(s) backing each observation. Do "
    "not invent findings, studies, or limitations from outside knowledge; if "
    "the sources do not support a claim, do not make it."
)

_TASK = (
    "Identify the research gaps, open questions, and limitations evident in "
    "or implied by the sources on this topic. Present them as a short list, "
    "citing the source number(s) that support each gap."
)


def analyze_gaps(
    query: str, llm_client: "LLMClient", settings: "Settings | None" = None
) -> AgentResult:
    return grounded_generate(query, SYSTEM_PROMPT, _TASK, llm_client, settings)
