from dataclasses import dataclass

from scholarmind.agents.llm_client import LLMClient
from scholarmind.citations.formatter import format_reference
from scholarmind.citations.metadata import normalize_metadata
from scholarmind.citations.verifier import ClaimVerification, verify_claim_support
from scholarmind.citations.verify import VerifiedAnswer


@dataclass
class Reference:
    citation_index: int
    apa: str
    bibtex: str


@dataclass
class VerificationReport:
    verifications: list[ClaimVerification]
    unsupported_count: int


@dataclass
class FormattedAndVerifiedAnswer:
    references: list[Reference]
    verification_report: VerificationReport


def format_and_verify(
    answer: "VerifiedAnswer", llm_client: "LLMClient"
) -> FormattedAndVerifiedAnswer:
    references = []
    for citation in answer.citations:
        metadata = normalize_metadata(citation.title, citation.authors, citation.year)
        references.append(
            Reference(
                citation_index=citation.index,
                apa=format_reference(metadata, "apa"),
                bibtex=format_reference(metadata, "bibtex"),
            )
        )

    verifications = verify_claim_support(answer.text, answer.citations, llm_client)
    unsupported_count = sum(1 for v in verifications if not v.supported)

    return FormattedAndVerifiedAnswer(
        references=references,
        verification_report=VerificationReport(
            verifications=verifications,
            unsupported_count=unsupported_count,
        ),
    )
