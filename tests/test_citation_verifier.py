import pytest

from scholarmind.citations.verifier import (
    ClaimVerification,
    extract_claim_for_citation,
    verify_claim_support,
)
from scholarmind.citations.verify import Citation
from scholarmind.config import get_settings

_has_llm_key = bool(get_settings().llm_api_key)


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        return self.response


def _make_citation(index: int, text: str = "The passage text.") -> Citation:
    return Citation(
        index=index,
        paper_id="paper-1",
        title="Some Title",
        authors=["Ada Lovelace"],
        year=2020,
        section="Results",
        page_start=1,
        page_end=2,
        text=text,
    )


def test_extract_claim_for_citation_returns_only_matching_sentences():
    text = (
        "RAG grounds answers in retrieved passages [1]. "
        "It reduces hallucination in generated text [2]. "
        "This is a general statement with no citation."
    )

    claim_1 = extract_claim_for_citation(text, 1)
    claim_2 = extract_claim_for_citation(text, 2)

    assert claim_1 == "RAG grounds answers in retrieved passages [1]."
    assert claim_2 == "It reduces hallucination in generated text [2]."


def test_extract_claim_for_citation_falls_back_to_whole_text_when_no_match():
    text = "RAG grounds answers in retrieved passages [1]."

    claim = extract_claim_for_citation(text, 99)

    assert claim == text


def test_verify_claim_support_supported_true():
    text = "RAG grounds answers in retrieved passages [1]."
    citation = _make_citation(1, text="RAG is a technique that grounds LLM answers in retrieved documents.")
    fake_client = FakeLLMClient("yes\nThe passage directly states this.")

    results = verify_claim_support(text, [citation], fake_client)

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, ClaimVerification)
    assert result.citation_index == 1
    assert result.claim == "RAG grounds answers in retrieved passages [1]."
    assert result.supported is True
    assert result.reason == "The passage directly states this."


def test_verify_claim_support_supported_false():
    text = "RAG grounds answers in retrieved passages [1]."
    citation = _make_citation(1, text="This passage is about an unrelated topic.")
    fake_client = FakeLLMClient("no\nThe passage does not mention this.")

    results = verify_claim_support(text, [citation], fake_client)

    assert results[0].supported is False
    assert results[0].reason == "The passage does not mention this."


def test_verify_claim_support_unparseable_response_is_fail_safe():
    text = "RAG grounds answers in retrieved passages [1]."
    citation = _make_citation(1)
    fake_client = FakeLLMClient("I'm not sure about this one.")

    results = verify_claim_support(text, [citation], fake_client)

    assert results[0].supported is False
    assert "could not be parsed" in results[0].reason


def test_verify_claim_support_empty_citations_never_calls_llm():
    fake_client = FakeLLMClient("yes\nreason")

    results = verify_claim_support("some text", [], fake_client)

    assert results == []
    assert fake_client.call_count == 0


@pytest.mark.skipif(not _has_llm_key, reason="LLM_API_KEY not configured")
def test_verify_claim_support_real_llm_flags_mismatched_claim():
    from scholarmind.agents.llm_client import OpenRouterClient

    settings = get_settings()
    client = OpenRouterClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    )

    text = "This passage proves the earth is flat [1]."
    citation = _make_citation(
        1,
        text=(
            "Retrieval-augmented generation (RAG) combines a retriever with a language "
            "model, grounding generated answers in retrieved documents to reduce "
            "hallucination in scholarly question answering."
        ),
    )

    results = verify_claim_support(text, [citation], client)

    assert len(results) == 1
    assert results[0].supported is False
