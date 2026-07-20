from dataclasses import dataclass
from typing import TYPE_CHECKING

from scholarmind.agents.llm_client import LLMClient
from scholarmind.citations.verify import VerifiedAnswer, build_source_block, verify_citations
from scholarmind.config import get_settings
from scholarmind.retrieval.search import search

if TYPE_CHECKING:
    from scholarmind.config import Settings


@dataclass
class AnswerResult:
    question: str
    answer: VerifiedAnswer | None
    sources_found: int


SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question using ONLY the numbered sources "
    "provided below. Cite every claim with the matching [N] marker. If the sources do not "
    "contain enough information to answer, say so explicitly instead of guessing."
)


def answer_question(
    question: str,
    llm_client: "LLMClient",
    settings: "Settings | None" = None,
    paper_id: str | None = None,
) -> AnswerResult:
    settings = settings or get_settings()

    sources = search(question, settings, paper_id=paper_id)

    if not sources:
        return AnswerResult(question=question, answer=None, sources_found=0)

    source_block = build_source_block(sources)
    user_prompt = f"Sources:\n{source_block}\n\nQuestion: {question}"

    raw_text = llm_client.complete(SYSTEM_PROMPT, user_prompt)
    verified = verify_citations(raw_text, sources)

    return AnswerResult(question=question, answer=verified, sources_found=len(sources))
