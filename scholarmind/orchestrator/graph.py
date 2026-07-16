from pathlib import Path

from langgraph.graph import StateGraph, START, END

from scholarmind.agents.llm_client import LLMClient
from scholarmind.agents.qa import answer_question
from scholarmind.config import Settings
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.orchestrator.state import GraphState


def classify_intent(request: str) -> tuple[str, str]:
    stripped = request.strip()
    lower = stripped.lower()

    if lower.startswith("ingest ") and stripped[len("ingest "):].strip():
        return "ingest", stripped[len("ingest "):].strip()
    if lower.endswith(".pdf"):
        return "ingest", stripped
    return "ask", stripped
    # future intents: discover, summarize, gap_analysis, cite, methodology, write — add a branch here + a corresponding node


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
        result = answer_question(state["question"], llm_client, settings)
        return {
            "answer_result": result,
            "messages": [f"answered with {result.sources_found} source(s)"],
        }

    def final(state: GraphState) -> dict:
        return {"messages": ["done"]}

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("ingest_node", ingest_node)
    graph.add_node("qa_node", qa_node)
    graph.add_node("final", final)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["intent"],
        {"ingest": "ingest_node", "ask": "qa_node"},
    )
    # future intents: discover, summarize, gap_analysis, cite, methodology, write — add a branch here + a corresponding node
    graph.add_edge("ingest_node", "final")
    graph.add_edge("qa_node", "final")
    graph.add_edge("final", END)

    return graph.compile()
