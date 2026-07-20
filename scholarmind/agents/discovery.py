from scholarmind.agents.base import AgentResult
from scholarmind.agents.llm_client import LLMClient
from scholarmind.config import Settings, get_settings
from scholarmind.discovery.service import CitationGraphResult, DiscoveryResult
from scholarmind.discovery.service import get_citation_graph as _get_citation_graph
from scholarmind.discovery.service import search_external as _search_external
from scholarmind.retrieval.dense import DenseResult
from scholarmind.retrieval.search import search


def _format_line(paper: DenseResult) -> str:
    label = paper.title or paper.paper_id
    details = []
    if paper.authors:
        details.append(", ".join(paper.authors))
    if paper.year is not None:
        details.append(str(paper.year))
    if details:
        return f"- {label} ({', '.join(details)})"
    return f"- {label}"


def discover(
    query: str, llm_client: "LLMClient | None" = None, settings: "Settings | None" = None
) -> AgentResult:
    settings = settings or get_settings()

    sources = search(query, settings)
    if not sources:
        return AgentResult(text="", sources=[], sources_found=0)

    seen: set[str] = set()
    lines: list[str] = []
    for paper in sources:
        key = paper.title or paper.paper_id
        if key in seen:
            continue
        seen.add(key)
        lines.append(_format_line(paper))

    text = "\n".join(lines)
    return AgentResult(text=text, sources=sources, sources_found=len(sources))


# discover() above only browses the already-ingested library (no network, no LLM). The two
# functions below search real external literature databases (arXiv, Semantic Scholar,
# OpenAlex) for papers not yet in the library — a distinct capability, kept as separate
# entry points here so discover()'s existing orchestrator/chat behavior is unaffected. The
# actual HTTP clients and dedupe logic live in scholarmind.discovery; these just re-export
# them under the discovery agent for discoverability.


def discover_external(
    query: str, settings: "Settings | None" = None, limit_per_source: int = 10
) -> DiscoveryResult:
    return _search_external(query, settings, limit_per_source)


def discover_citation_graph(
    doi: str | None = None,
    s2_paper_id: str | None = None,
    title: str | None = None,
    settings: "Settings | None" = None,
    limit: int = 25,
) -> CitationGraphResult:
    return _get_citation_graph(doi, s2_paper_id, title, settings, limit)
