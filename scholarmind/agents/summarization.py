from scholarmind.agents.base import AgentResult, grounded_generate
from scholarmind.agents.llm_client import LLMClient
from scholarmind.config import Settings

SYSTEM_PROMPT = (
    "You are a research summarization assistant. Summarize ONLY the information "
    "present in the numbered sources provided below. Never add facts, context, or "
    "conclusions from outside knowledge, even if you believe them to be true. If the "
    "sources disagree or are incomplete on some point, reflect that uncertainty rather "
    "than resolving it yourself. Stay concise, use plain language, and remain strictly "
    "faithful to what the sources actually say."
)

_TASK = "Write a concise, faithful summary of what the sources report about the topic."


def summarize(
    query: str, llm_client: "LLMClient", settings: "Settings | None" = None
) -> AgentResult:
    return grounded_generate(query, SYSTEM_PROMPT, _TASK, llm_client, settings)
