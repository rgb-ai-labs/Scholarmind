from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scholarmind.agents.llm_client import LLMClient
from scholarmind.citations.verify import VerifiedAnswer, build_source_block, verify_citations
from scholarmind.config import get_settings
from scholarmind.retrieval.search import search

if TYPE_CHECKING:
    from scholarmind.config import Settings
    from scholarmind.retrieval.dense import DenseResult


@dataclass
class AnswerResult:
    question: str
    answer: VerifiedAnswer | None
    sources_found: int


@dataclass
class StreamingAnswer:
    # The retrieved sources are captured up front so the caller can verify the answer against
    # them once the full text has streamed in (verification needs the whole answer, not deltas).
    sources: list["DenseResult"]
    tokens: Iterator[str]


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


def answer_question_streaming(
    question: str,
    llm_client: "LLMClient",
    settings: "Settings | None" = None,
    paper_id: str | None = None,
) -> "StreamingAnswer | None":
    # Streaming variant of answer_question for the web app: does the same retrieval and builds
    # the same prompt, but returns a token iterator to render live instead of the finished
    # answer. Returns None when retrieval found nothing (so the caller can say so without calling
    # the LLM). Once the caller has the full streamed text, it must call
    # finalize_streamed_answer() to verify citations — the two steps together are exactly
    # equivalent to answer_question().
    settings = settings or get_settings()

    sources = search(question, settings, paper_id=paper_id)
    if not sources:
        return None

    source_block = build_source_block(sources)
    user_prompt = f"Sources:\n{source_block}\n\nQuestion: {question}"

    return StreamingAnswer(sources=sources, tokens=llm_client.stream(SYSTEM_PROMPT, user_prompt))


def finalize_streamed_answer(
    question: str, full_text: str, sources: list["DenseResult"]
) -> AnswerResult:
    verified = verify_citations(full_text, sources)
    return AnswerResult(question=question, answer=verified, sources_found=len(sources))
