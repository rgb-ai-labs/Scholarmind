import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from scholarmind.config import get_settings
from scholarmind.discovery.models import Candidate, DiscoverySourceError
from scholarmind.ingestion.pipeline import IngestResult, ingest_metadata_record, run_ingestion

if TYPE_CHECKING:
    from scholarmind.config import Settings

_USER_AGENT = "ScholarMind/0.1 (https://github.com/scholarmind; mailto:contact@example.com)"
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _downloads_dir_for(settings: "Settings") -> Path:
    # Mirrors scholarmind.webapp.library.papers_dir_for's one-line derivation, kept local so
    # this engine-layer module doesn't depend on the webapp layer.
    return Path(settings.qdrant_path).resolve().parent / "uploads"


def _safe_filename(candidate: Candidate) -> str:
    raw = f"{candidate.source}-{candidate.external_id or candidate.title or 'paper'}"
    return _UNSAFE_FILENAME_CHARS.sub("_", raw)[:150]


def _candidate_paper_id(candidate: Candidate) -> str:
    identity = candidate.doi or candidate.title or candidate.external_id or ""
    return hashlib.sha256(identity.strip().lower().encode()).hexdigest()


def _download_pdf(pdf_url: str, timeout: float = 30.0) -> bytes | None:
    try:
        response = httpx.get(
            pdf_url,
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    if not response.content.startswith(b"%PDF"):
        return None
    return response.content


def ingest_candidate(candidate: Candidate, settings: "Settings | None" = None) -> IngestResult:
    settings = settings or get_settings()

    if candidate.pdf_url:
        pdf_bytes = _download_pdf(candidate.pdf_url)
        if pdf_bytes is not None:
            downloads_dir = _downloads_dir_for(settings)
            downloads_dir.mkdir(parents=True, exist_ok=True)
            dest = downloads_dir / f"{_safe_filename(candidate)}.pdf"
            dest.write_bytes(pdf_bytes)
            try:
                return run_ingestion(dest, settings)
            except Exception as exc:
                raise DiscoverySourceError(
                    candidate.source, f"downloaded PDF but ingestion failed: {exc}"
                ) from exc

    abstract = candidate.abstract or "(no abstract available from this source)"
    return ingest_metadata_record(
        paper_id=_candidate_paper_id(candidate),
        title=candidate.title,
        authors=candidate.authors,
        year=candidate.year,
        venue=candidate.venue,
        text=abstract,
        source_filename=f"{candidate.source}:{candidate.external_id}",
        doi=candidate.doi,
        settings=settings,
    )
