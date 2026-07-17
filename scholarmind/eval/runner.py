import json
from dataclasses import dataclass
from pathlib import Path

from scholarmind.agents.llm_client import LLMClient
from scholarmind.config import Settings
from scholarmind.eval.metrics import faithfulness, precision_at_k, recall_at_k
from scholarmind.orchestrator import run as orchestrator_run
from scholarmind.retrieval.search import search


@dataclass
class EvalCase:
    question: str
    expected_title: str


@dataclass
class CaseResult:
    question: str
    expected_title: str
    precision_at_k: float
    recall_at_k: float
    faithfulness: float


@dataclass
class Scorecard:
    k: int
    num_cases: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_faithfulness: float
    per_case: list[CaseResult]


def load_eval_set(path: "str | Path") -> list[EvalCase]:
    with open(path, encoding="utf-8") as f:
        raw_cases = json.load(f)
    return [
        EvalCase(question=case["question"], expected_title=case["expected_title"])
        for case in raw_cases
    ]


def run_eval(
    cases: list[EvalCase], llm_client: "LLMClient", settings: "Settings", k: int = 5
) -> Scorecard:
    per_case: list[CaseResult] = []

    for case in cases:
        retrieved = search(case.question, settings)
        titles = [r.title or "" for r in retrieved]

        precision = precision_at_k(titles, case.expected_title, k)
        recall = recall_at_k(titles, case.expected_title, k)

        result = orchestrator_run.run(case.question, llm_client=llm_client, settings=settings)

        if result.formatted_answer is not None:
            supported_flags = [
                v.supported for v in result.formatted_answer.verification_report.verifications
            ]
        else:
            supported_flags = []

        faith = faithfulness(supported_flags)

        per_case.append(
            CaseResult(
                question=case.question,
                expected_title=case.expected_title,
                precision_at_k=precision,
                recall_at_k=recall,
                faithfulness=faith,
            )
        )

    num_cases = len(per_case)
    if num_cases == 0:
        mean_precision = 0.0
        mean_recall = 0.0
        mean_faithfulness = 0.0
    else:
        mean_precision = sum(c.precision_at_k for c in per_case) / num_cases
        mean_recall = sum(c.recall_at_k for c in per_case) / num_cases
        mean_faithfulness = sum(c.faithfulness for c in per_case) / num_cases

    return Scorecard(
        k=k,
        num_cases=num_cases,
        mean_precision_at_k=mean_precision,
        mean_recall_at_k=mean_recall,
        mean_faithfulness=mean_faithfulness,
        per_case=per_case,
    )
