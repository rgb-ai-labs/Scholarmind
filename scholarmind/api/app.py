import shutil
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from scholarmind.agents.llm_client import LLMClient
from scholarmind.api.models import (
    AskRequest,
    AskResponse,
    CitationModel,
    ClaimVerificationModel,
    HealthResponse,
    IngestResponse,
    ReferenceModel,
    VerificationReportModel,
)
from scholarmind.config import Settings, get_settings
from scholarmind.orchestrator import run as orchestrator_run
from scholarmind.orchestrator.run import ChatResult


def get_settings_dependency() -> Settings:
    return get_settings()


def get_llm_client_dependency() -> "LLMClient | None":
    return None


def _to_ask_response(result: ChatResult) -> AskResponse:
    answer_text = None
    sources: list[CitationModel] = []
    invalid_markers: list[int] = []
    if result.answer_result is not None and result.answer_result.answer is not None:
        verified = result.answer_result.answer
        answer_text = verified.text
        invalid_markers = list(verified.invalid_citation_markers)
        sources = [
            CitationModel(
                index=c.index,
                paper_id=c.paper_id,
                title=c.title,
                authors=c.authors,
                year=c.year,
                section=c.section,
                page_start=c.page_start,
                page_end=c.page_end,
                text=c.text,
            )
            for c in verified.citations
        ]

    references: list[ReferenceModel] = []
    verification_report = None
    if result.formatted_answer is not None:
        references = [
            ReferenceModel(citation_index=r.citation_index, apa=r.apa, bibtex=r.bibtex)
            for r in result.formatted_answer.references
        ]
        report = result.formatted_answer.verification_report
        verification_report = VerificationReportModel(
            verifications=[
                ClaimVerificationModel(
                    citation_index=v.citation_index,
                    claim=v.claim,
                    supported=v.supported,
                    reason=v.reason,
                )
                for v in report.verifications
            ],
            unsupported_count=report.unsupported_count,
        )

    sources_found = (
        result.answer_result.sources_found if result.answer_result is not None else 0
    )

    return AskResponse(
        intent=result.intent,
        answer=answer_text,
        sources_found=sources_found,
        sources=sources,
        invalid_citation_markers=invalid_markers,
        references=references,
        verification_report=verification_report,
        formatting_error=result.formatting_error,
        error=result.error,
    )


def create_app() -> FastAPI:
    api = FastAPI(title="ScholarMind API", version="0.1.0")

    @api.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @api.post("/ingest")
    def ingest(
        path: str | None = Form(default=None),
        file: UploadFile | None = File(default=None),
        settings: Settings = Depends(get_settings_dependency),
    ) -> IngestResponse:
        if file is not None and file.filename:
            tmp_dir = Path(tempfile.mkdtemp())
            target = tmp_dir / file.filename
            with target.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
            source = str(target)
        elif path:
            source = path
        else:
            raise HTTPException(status_code=400, detail="Provide a path or a file upload.")

        result = orchestrator_run.run(f"ingest {source}", settings=settings)
        if result.error is not None:
            raise HTTPException(status_code=400, detail=result.error)
        if result.ingest_result is None:
            raise HTTPException(status_code=500, detail="Ingestion produced no result.")

        return IngestResponse(
            papers_ingested=result.ingest_result.papers_ingested,
            chunks_created=result.ingest_result.chunks_created,
            collection_name=result.ingest_result.collection_name,
        )

    @api.post("/ask")
    def ask(
        request: AskRequest,
        settings: Settings = Depends(get_settings_dependency),
        llm_client: "LLMClient | None" = Depends(get_llm_client_dependency),
    ) -> AskResponse:
        result = orchestrator_run.run(
            request.question, llm_client=llm_client, settings=settings
        )
        return _to_ask_response(result)

    return api


app = create_app()
