from dataclasses import dataclass

from scholarmind.agents.llm_client import LLMClient
from scholarmind.citations.verify import build_source_block
from scholarmind.config import Settings, get_settings
from scholarmind.retrieval.dense import DenseResult
from scholarmind.retrieval.search import search


@dataclass
class AgentResult:
    text: str
    sources: list[DenseResult]
    sources_found: int


def grounded_generate(
    query: str,
    system_prompt: str,
    task_instruction: str,
    llm_client: "LLMClient",
    settings: "Settings | None" = None,
) -> AgentResult:
    if settings is None:
        settings = get_settings()

    sources = search(query, settings)
    if not sources:
        return AgentResult(text="", sources=[], sources_found=0)

    source_block = build_source_block(sources)
    user_prompt = (
        f"{task_instruction}\n\nSources:\n{source_block}\n\nTopic: {query}"
    )
    text = llm_client.complete(system_prompt, user_prompt)
    return AgentResult(text=text, sources=sources, sources_found=len(sources))
