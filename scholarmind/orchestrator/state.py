from typing import Annotated, TypedDict
import operator

from scholarmind.agents.qa import AnswerResult
from scholarmind.ingestion.pipeline import IngestResult


class GraphState(TypedDict, total=False):
    request: str
    intent: str
    ingest_path: str
    question: str
    ingest_result: IngestResult
    answer_result: AnswerResult
    messages: Annotated[list[str], operator.add]
    error: str
