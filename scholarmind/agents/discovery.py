from scholarmind.agents.base import AgentResult
from scholarmind.agents.llm_client import LLMClient
from scholarmind.config import Settings, get_settings
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
