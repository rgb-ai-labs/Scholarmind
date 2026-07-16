from scholarmind.agents.base import AgentResult, grounded_generate
from scholarmind.agents.llm_client import LLMClient
from scholarmind.config import Settings

SYSTEM_PROMPT = (
    "You are a research methodology analyst. Using ONLY the numbered sources provided below, "
    "extract and compare the methods, experimental setups, data, and evaluation approaches "
    "described for the given topic. For each distinct method or system, summarize what it does, "
    "what data or inputs it uses, and how it is evaluated, citing every claim with the matching "
    "[N] marker. When multiple sources describe different methods, compare and contrast them "
    "directly. Never infer, assume, or describe a method, dataset, or evaluation metric that the "
    "sources do not explicitly state. If the sources do not describe enough methodological detail "
    "to answer, say so explicitly instead of guessing."
)

_TASK = (
    "Describe and compare the methodology (methods, data, evaluation) that the sources use for "
    "this topic."
)


def extract_methodology(
    query: str, llm_client: "LLMClient", settings: "Settings | None" = None
) -> AgentResult:
    return grounded_generate(query, SYSTEM_PROMPT, _TASK, llm_client, settings)
