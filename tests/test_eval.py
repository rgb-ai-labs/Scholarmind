from pathlib import Path

from scholarmind.config import Settings
from scholarmind.eval.metrics import faithfulness, precision_at_k, recall_at_k
from scholarmind.eval.runner import EvalCase, run_eval
from scholarmind.ingestion.pipeline import run_ingestion

RAG_TITLE = "A Study of Retrieval-Augmented Generation for Scholarly Question Answering"
TIDAL_TITLE = "Tidal Energy Harvesting Efficiency in Coastal Turbine Arrays"

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"
FIXTURE_PATH_2 = Path(__file__).parent / "fixtures" / "sample_paper_2.pdf"


class RoutingFakeLLMClient:
    def __init__(self, answer_response: str, verification_response: str) -> None:
        self.answer_response = answer_response
        self.verification_response = verification_response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if "fact-checking" in system_prompt:
            return self.verification_response
        return self.answer_response


def test_precision_at_k_all_match():
    titles = [RAG_TITLE, RAG_TITLE, RAG_TITLE]
    assert precision_at_k(titles, RAG_TITLE, 3) == 1.0


def test_precision_at_k_partial_match():
    titles = [RAG_TITLE, TIDAL_TITLE, RAG_TITLE, TIDAL_TITLE]
    assert precision_at_k(titles, RAG_TITLE, 4) == 0.5


def test_precision_at_k_no_match():
    titles = [TIDAL_TITLE, TIDAL_TITLE]
    assert precision_at_k(titles, RAG_TITLE, 2) == 0.0


def test_precision_at_k_empty_list():
    assert precision_at_k([], RAG_TITLE, 5) == 0.0


def test_precision_at_k_k_larger_than_list():
    titles = [RAG_TITLE]
    assert precision_at_k(titles, RAG_TITLE, 5) == 1.0


def test_recall_at_k_present():
    titles = [TIDAL_TITLE, RAG_TITLE, TIDAL_TITLE]
    assert recall_at_k(titles, RAG_TITLE, 3) == 1.0


def test_recall_at_k_absent():
    titles = [TIDAL_TITLE, TIDAL_TITLE]
    assert recall_at_k(titles, RAG_TITLE, 2) == 0.0


def test_recall_at_k_present_but_outside_k():
    titles = [TIDAL_TITLE, TIDAL_TITLE, RAG_TITLE]
    assert recall_at_k(titles, RAG_TITLE, 2) == 0.0


def test_faithfulness_all_true():
    assert faithfulness([True, True, True]) == 1.0


def test_faithfulness_mixed():
    assert faithfulness([True, False, True, False]) == 0.5


def test_faithfulness_empty():
    assert faithfulness([]) == 1.0


def test_run_eval_end_to_end(tmp_path: Path):
    settings = Settings(
        qdrant_path=str(tmp_path / "q"),
        qdrant_collection="test_eval_chunks",
    )
    run_ingestion(FIXTURE_PATH, settings)
    run_ingestion(FIXTURE_PATH_2, settings)

    cases = [
        EvalCase(question="What is retrieval-augmented generation?", expected_title=RAG_TITLE),
        EvalCase(
            question="Why does grounding claims in retrieved chunks matter?",
            expected_title=RAG_TITLE,
        ),
        EvalCase(
            question="How do coastal turbines capture tidal energy?",
            expected_title=TIDAL_TITLE,
        ),
    ]

    fake_client = RoutingFakeLLMClient(
        answer_response="RAG grounds answers in sources [1].",
        verification_response="yes\nsupported",
    )

    scorecard = run_eval(cases, fake_client, settings, k=5)

    assert scorecard.num_cases == 3
    assert scorecard.k == 5
    assert len(scorecard.per_case) == 3

    assert isinstance(scorecard.mean_precision_at_k, float)
    assert isinstance(scorecard.mean_recall_at_k, float)
    assert isinstance(scorecard.mean_faithfulness, float)
    assert 0.0 <= scorecard.mean_precision_at_k <= 1.0
    assert 0.0 <= scorecard.mean_recall_at_k <= 1.0
    assert 0.0 <= scorecard.mean_faithfulness <= 1.0

    rag_case = scorecard.per_case[0]
    assert rag_case.expected_title == RAG_TITLE
    assert rag_case.recall_at_k == 1.0
