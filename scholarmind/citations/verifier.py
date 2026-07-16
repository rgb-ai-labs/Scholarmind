import re
from dataclasses import dataclass

from scholarmind.agents.llm_client import LLMClient
from scholarmind.citations.verify import Citation

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

SYSTEM_PROMPT = (
    "You are a fact-checking assistant. Given a passage and a claim, determine whether the "
    "passage supports the claim. Respond in exactly two lines: the first line is either "
    "'yes' or 'no' (lowercase, nothing else), the second line is a one-sentence reason."
)


@dataclass
class ClaimVerification:
    citation_index: int
    claim: str
    supported: bool
    reason: str


def extract_claim_for_citation(text: str, citation_index: int) -> str:
    marker = f"[{citation_index}]"
    sentences = _SENTENCE_SPLIT.split(text)
    matching = [sentence for sentence in sentences if marker in sentence]
    if not matching:
        return text
    return " ".join(matching)


def verify_claim_support(
    text: str, citations: list["Citation"], llm_client: "LLMClient"
) -> list["ClaimVerification"]:
    if not citations:
        return []

    results: list[ClaimVerification] = []
    for citation in citations:
        claim = extract_claim_for_citation(text, citation.index)
        user_prompt = (
            f"Passage:\n{citation.text}\n\nClaim:\n{claim}\n\nDoes the passage "
            f"support the claim?"
        )
        response = llm_client.complete(SYSTEM_PROMPT, user_prompt)

        lines = [line.strip() for line in response.splitlines() if line.strip()]
        verdict = lines[0].lower() if lines else ""

        if verdict.startswith("yes"):
            supported = True
            reason = lines[1] if len(lines) > 1 else ""
        elif verdict.startswith("no"):
            supported = False
            reason = lines[1] if len(lines) > 1 else ""
        else:
            supported = False
            reason = f"model response could not be parsed: {response[:200]}"

        results.append(
            ClaimVerification(
                citation_index=citation.index,
                claim=claim,
                supported=supported,
                reason=reason,
            )
        )

    return results
