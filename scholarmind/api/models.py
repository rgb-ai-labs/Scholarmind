from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class IngestRequest(BaseModel):
    path: str


class IngestResponse(BaseModel):
    papers_ingested: int
    chunks_created: int
    collection_name: str


class AskRequest(BaseModel):
    question: str


class CitationModel(BaseModel):
    index: int
    paper_id: str
    title: str | None
    authors: list[str]
    year: int | None
    section: str | None
    page_start: int
    page_end: int
    text: str


class ReferenceModel(BaseModel):
    citation_index: int
    apa: str
    bibtex: str


class ClaimVerificationModel(BaseModel):
    citation_index: int
    claim: str
    supported: bool
    reason: str


class VerificationReportModel(BaseModel):
    verifications: list[ClaimVerificationModel]
    unsupported_count: int


class AskResponse(BaseModel):
    intent: str
    answer: str | None
    sources_found: int
    sources: list[CitationModel]
    invalid_citation_markers: list[int]
    references: list[ReferenceModel]
    verification_report: VerificationReportModel | None
    formatting_error: str | None
    error: str | None
