from typing import Annotated, TypedDict
import operator

from scholarmind.agents.qa import AnswerResult
from scholarmind.citations.service import FormattedAndVerifiedAnswer
from scholarmind.ingestion.pipeline import IngestResult


class GraphState(TypedDict, total=False):
    request: str
    intent: str
    ingest_path: str
    question: str
    ingest_result: IngestResult
    answer_result: AnswerResult
    formatted_answer: FormattedAndVerifiedAnswer
    formatting_error: str
    messages: Annotated[list[str], operator.add]
    error: str
