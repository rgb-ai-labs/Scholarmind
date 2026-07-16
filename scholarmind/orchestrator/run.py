from dataclasses import dataclass

from scholarmind.agents.llm_client import LLMClient, OpenRouterClient
from scholarmind.agents.qa import AnswerResult
from scholarmind.config import Settings, get_settings
from scholarmind.ingestion.pipeline import IngestResult
from scholarmind.orchestrator.graph import build_graph


@dataclass
class ChatResult:
    intent: str
    answer_result: AnswerResult | None
    ingest_result: IngestResult | None
    error: str | None


class _LazyOpenRouterClient:
    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._client: OpenRouterClient | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if self._client is None:
            self._client = OpenRouterClient(
                api_key=self._settings.llm_api_key,
                base_url=self._settings.llm_base_url,
                model=self._settings.llm_model,
                max_tokens=self._settings.llm_max_tokens,
            )
        return self._client.complete(system_prompt, user_prompt)


def run(
    request: str,
    llm_client: "LLMClient | None" = None,
    settings: "Settings | None" = None,
) -> ChatResult:
    settings = settings or get_settings()
    llm_client = llm_client or _LazyOpenRouterClient(settings)

    graph = build_graph(llm_client, settings)
    final_state = graph.invoke({"request": request})

    return ChatResult(
        intent=final_state.get("intent", ""),
        answer_result=final_state.get("answer_result"),
        ingest_result=final_state.get("ingest_result"),
        error=final_state.get("error"),
    )
