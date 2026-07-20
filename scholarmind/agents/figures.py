from pathlib import Path

from scholarmind.agents.base import AgentResult
from scholarmind.agents.llm_client import LLMClient
from scholarmind.config import Settings, get_settings
from scholarmind.retrieval.dense import DenseResult

SYSTEM_PROMPT = (
    "You are a research assistant answering a question about a specific figure from a "
    "paper, using the figure image itself plus its caption. Ground your answer ONLY in what "
    "is visible in the image and stated in the caption. If the figure doesn't contain enough "
    "information to answer, say so explicitly rather than guessing."
)

CAPTION_ONLY_SYSTEM_PROMPT = (
    "You are a research assistant answering a question about a specific figure from a "
    "paper, using ONLY its caption text — you do not have access to the image itself. "
    "Ground your answer only in the caption, and do not describe or speculate about what "
    "the figure might visually show. If the caption doesn't contain enough information to "
    "answer, say so explicitly."
)


def answer_about_figure(
    figure: "DenseResult",
    question: str,
    llm_client: "LLMClient",
    settings: "Settings | None" = None,
) -> AgentResult:
    settings = settings or get_settings()

    caption_block = f"Figure caption (page {figure.page_start}): {figure.text}"
    user_prompt = f"{question}\n\n{caption_block}"

    use_vision = (
        bool(settings.vision_model)
        and figure.image_path is not None
        and Path(figure.image_path).is_file()
        and hasattr(llm_client, "complete_with_image")
    )

    if use_vision:
        try:
            answer = llm_client.complete_with_image(
                SYSTEM_PROMPT, user_prompt, figure.image_path, settings.vision_model
            )
            return AgentResult(text=answer, sources=[figure], sources_found=1)
        except Exception:
            pass  # fall through to caption-only rather than failing the whole request

    answer = llm_client.complete(CAPTION_ONLY_SYSTEM_PROMPT, user_prompt)
    return AgentResult(text=answer, sources=[figure], sources_found=1)
