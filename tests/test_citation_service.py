from scholarmind.citations.service import format_and_verify
from scholarmind.citations.verify import Citation, VerifiedAnswer


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        return self.response


def _citation(index: int, text: str) -> Citation:
    return Citation(
        index=index,
        paper_id="paper-1",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=["Patrick Lewis"],
        year=2020,
        section="Introduction",
        page_start=1,
        page_end=1,
        text=text,
    )


def test_format_and_verify_builds_references_for_every_citation():
    answer = VerifiedAnswer(
        text="RAG grounds answers in retrieved passages [1].",
        citations=[_citation(1, "RAG grounds generation in retrieved passages.")],
        invalid_citation_markers=[],
    )
    client = FakeLLMClient("yes\nThe passage directly supports the claim.")

    result = format_and_verify(answer, client)

    assert len(result.references) == 1
    reference = result.references[0]
    assert reference.citation_index == 1
    assert reference.apa
    assert reference.bibtex.startswith("@article{")


def test_format_and_verify_flags_unsupported_claims():
    answer = VerifiedAnswer(
        text="This proves the earth is flat [1].",
        citations=[_citation(1, "RAG grounds generation in retrieved passages.")],
        invalid_citation_markers=[],
    )
    client = FakeLLMClient("no\nThe passage does not discuss the shape of the earth.")

    result = format_and_verify(answer, client)

    assert result.verification_report.unsupported_count == 1
    assert result.verification_report.verifications[0].supported is False


def test_format_and_verify_with_no_citations_returns_empty_report():
    answer = VerifiedAnswer(text="No sources were used.", citations=[], invalid_citation_markers=[])
    client = FakeLLMClient("yes\nreason")

    result = format_and_verify(answer, client)

    assert result.references == []
    assert result.verification_report.verifications == []
    assert result.verification_report.unsupported_count == 0
    assert client.call_count == 0
