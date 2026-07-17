from pathlib import Path

from langgraph.graph import END, START, StateGraph

from scholarmind.agents.discovery import discover
from scholarmind.agents.gap_analysis import analyze_gaps
from scholarmind.agents.llm_client import LLMClient
from scholarmind.agents.methodology import extract_methodology
from scholarmind.agents.qa import answer_question
from scholarmind.agents.summarization import summarize
from scholarmind.agents.writing import draft_section
from scholarmind.citations.service import format_and_verify
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.orchestrator.state import GraphState

AGENT_REGISTRY = {
    "discover": discover,
    "summarize": summarize,
    "gaps": analyze_gaps,
    "methods": extract_methodology,
    "write": draft_section,
}

_AGENT_PREFIXES = [
    ("discover ", "discover"),
    ("summarize ", "summarize"),
    ("gaps ", "gaps"),
    ("methods ", "methods"),
    ("write ", "write"),
    ("draft ", "write"),
]


def classify_intent(request: str) -> tuple[str, str]:
    stripped = request.strip()
    lower = stripped.lower()

    if lower.startswith("ingest ") and stripped[len("ingest "):].strip():
        return "ingest", stripped[len("ingest "):].strip()
    for prefix, intent in _AGENT_PREFIXES:
        if lower.startswith(prefix) and stripped[len(prefix):].strip():
            return intent, stripped[len(prefix):].strip()
    # future intents: cite — add a prefix to _AGENT_PREFIXES + an entry to AGENT_REGISTRY
    if lower.endswith(".pdf"):
        return "ingest", stripped
    return "ask", stripped


def build_graph(llm_client: "LLMClient", settings: "Settings"):
    def supervisor(state: GraphState) -> dict:
        intent, payload = classify_intent(state["request"])
        if intent == "ingest":
            return {
                "intent": intent,
                "ingest_path": payload,
                "messages": [f"routed to {intent}"],
            }
        return {
            "intent": intent,
            "question": payload,
            "messages": [f"routed to {intent}"],
        }

    def ingest_node(state: GraphState) -> dict:
        try:
            result = run_ingestion(Path(state["ingest_path"]), settings)
        except Exception as exc:
            return {"error": str(exc), "messages": ["ingest failed"]}
        return {
            "ingest_result": result,
            "messages": [
                f"ingested {result.papers_ingested} paper(s), "
                f"{result.chunks_created} chunk(s)"
            ],
        }

    def qa_node(state: GraphState) -> dict:
        try:
            result = answer_question(state["question"], llm_client, settings)
        except Exception as exc:
            return {"error": str(exc), "messages": ["ask failed"]}
        return {
            "answer_result": result,
            "messages": [f"answered with {result.sources_found} source(s)"],
        }

    def agent_node(state: GraphState) -> dict:
        agent_fn = AGENT_REGISTRY[state["intent"]]
        try:
            result = agent_fn(state["question"], llm_client, settings)
        except Exception as exc:
            return {"error": str(exc), "messages": ["agent failed"]}
        return {
            "agent_result": result,
            "messages": [
                f"ran {state['intent']} agent, {result.sources_found} source(s)"
            ],
        }

    def final(state: GraphState) -> dict:
        answer_result = state.get("answer_result")
        if answer_result is None or answer_result.answer is None:
            return {"messages": ["done"]}
        try:
            formatted = format_and_verify(answer_result.answer, llm_client)
        except Exception as exc:
            return {
                "formatting_error": str(exc),
                "messages": ["citation formatting failed"],
            }
        return {
            "formatted_answer": formatted,
            "messages": [
                f"formatted {len(formatted.references)} reference(s), "
                f"{formatted.verification_report.unsupported_count} unsupported"
            ],
        }

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("ingest_node", ingest_node)
    graph.add_node("qa_node", qa_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("final", final)

    routing = {"ingest": "ingest_node", "ask": "qa_node"}
    for intent in AGENT_REGISTRY:
        routing[intent] = "agent_node"

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["intent"],
        routing,
    )
    graph.add_edge("ingest_node", "final")
    graph.add_edge("qa_node", "final")
    graph.add_edge("agent_node", "final")
    graph.add_edge("final", END)

    return graph.compile()
