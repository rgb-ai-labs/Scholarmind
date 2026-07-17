import streamlit as st

from scholarmind.agents.llm_client import OpenRouterClient
from scholarmind.agents.qa import AnswerResult, answer_question
from scholarmind.citations.service import FormattedAndVerifiedAnswer, format_and_verify
from scholarmind.config import Settings, get_settings
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.webapp.library import get_library_stats, papers_dir_for


@st.cache_resource
def _load_llm_client(
    api_key: str, base_url: str, model: str, max_tokens: int
) -> OpenRouterClient:
    return OpenRouterClient(api_key=api_key, base_url=base_url, model=model, max_tokens=max_tokens)


def _effective_settings() -> Settings:
    base = get_settings()
    override_key = st.session_state.get("llm_api_key_override", "")
    if override_key:
        return base.model_copy(update={"llm_api_key": override_key})
    return base


def _refresh_library_stats(settings: "Settings") -> None:
    st.session_state["library_stats"] = get_library_stats(settings)


def render_sidebar(settings: "Settings") -> None:
    with st.sidebar:
        st.header("Settings")
        st.text(f"LLM model: {settings.llm_model}")
        st.text(f"Embedding model: {settings.embedding_model}")
        st.text(f"Qdrant path: {settings.qdrant_path}")

        if not settings.llm_api_key:
            st.warning(
                "No LLM_API_KEY configured. Asking questions, verifying citations, and "
                "running domain agents all need a key (ingestion does not)."
            )
            st.caption("Set LLM_API_KEY in .env, or paste a key below for this session only.")

        key_input = st.text_input(
            "LLM API key (this session only)", type="password", value=""
        )
        if key_input:
            st.session_state["llm_api_key_override"] = key_input

        st.header("Library")
        if "library_stats" not in st.session_state:
            _refresh_library_stats(settings)
        papers, chunks = st.session_state["library_stats"]
        col1, col2 = st.columns(2)
        col1.metric("Papers indexed", papers)
        col2.metric("Chunks indexed", chunks)
        if st.button("Refresh"):
            _refresh_library_stats(settings)
            st.rerun()


def render_ingest_panel(settings: "Settings") -> None:
    st.subheader("Upload & ingest")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF papers", type=["pdf"], accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Upload a paper to get started.")
        return

    if not st.button("Ingest uploaded papers"):
        return

    papers_dir = papers_dir_for(settings)
    papers_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        dest = papers_dir / uploaded_file.name
        dest.write_bytes(uploaded_file.getbuffer())

        with st.spinner(
            f"Ingesting {uploaded_file.name}… (first run also downloads the "
            "embedding model, which can take a minute)"
        ):
            try:
                result = run_ingestion(dest, settings)
            except Exception as exc:
                st.error(f"{uploaded_file.name}: failed to ingest — {exc}")
                continue

        st.success(
            f"{uploaded_file.name}: {result.papers_ingested} paper(s), "
            f"{result.chunks_created} chunk(s) ingested into '{result.collection_name}'."
        )

    _refresh_library_stats(settings)


def render_sources(answer_result: "AnswerResult") -> None:
    verified = answer_result.answer
    if verified is None or not verified.citations:
        return

    with st.expander(f"Sources ({len(verified.citations)})"):
        for citation in verified.citations:
            authors = ", ".join(citation.authors) if citation.authors else "Unknown author"
            year = citation.year if citation.year is not None else "n.d."
            section = f", {citation.section}" if citation.section else ""
            st.markdown(
                f"**[{citation.index}]** {citation.title or 'Untitled'} — {authors} "
                f"({year}){section}, pp. {citation.page_start}-{citation.page_end}"
            )

    if verified.invalid_citation_markers:
        markers = ", ".join(f"[{m}]" for m in verified.invalid_citation_markers)
        st.warning(
            f"The model referenced source(s) {markers} that don't exist in the retrieved "
            "sources — they were not included above."
        )


def render_verification(formatted: "FormattedAndVerifiedAnswer") -> None:
    report = formatted.verification_report
    if not report.verifications:
        return

    supported = len(report.verifications) - report.unsupported_count
    with st.expander(
        f"Verification ({supported}/{len(report.verifications)} claims supported)",
        expanded=report.unsupported_count > 0,
    ):
        for verification in report.verifications:
            if verification.supported:
                st.success(f"[{verification.citation_index}] {verification.claim}")
            else:
                st.warning(
                    f"[{verification.citation_index}] {verification.claim}\n\n"
                    f"*Not supported by the cited source: {verification.reason}*"
                )

    if formatted.references:
        with st.expander("References (APA / BibTeX)"):
            for reference in formatted.references:
                st.markdown(f"**[{reference.citation_index}]** {reference.apa}")
                st.code(reference.bibtex, language="bibtex")


def render_answer(
    answer_result: "AnswerResult | None", formatted: "FormattedAndVerifiedAnswer | None"
) -> None:
    if answer_result is None:
        st.error("Something went wrong generating this answer.")
        return

    if answer_result.answer is None:
        st.info(f"No relevant sources found for: {answer_result.question}")
        return

    st.markdown(answer_result.answer.text)
    render_sources(answer_result)
    if formatted is not None:
        render_verification(formatted)


def _answer_and_verify(
    question: str, settings: "Settings"
) -> tuple["AnswerResult | None", "FormattedAndVerifiedAnswer | None"]:
    client = _load_llm_client(
        settings.llm_api_key, settings.llm_base_url, settings.llm_model, settings.llm_max_tokens
    )
    answer_result = answer_question(question, client, settings)

    formatted = None
    if answer_result.answer is not None:
        formatted = format_and_verify(answer_result.answer, client)

    return answer_result, formatted


def render_chat_panel(settings: "Settings") -> None:
    st.subheader("Ask")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer(message["answer_result"], message["formatted"])
            else:
                st.write(message["content"])

    question = st.chat_input("Ask a question about your ingested papers…")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if not settings.llm_api_key:
            st.info(
                "Add an LLM API key in the sidebar to ask questions — retrieval and "
                "ingestion work without one, but generating an answer needs a key."
            )
            answer_result, formatted = None, None
        else:
            with st.spinner(
                "Thinking… (the first question also loads the reranker model, which "
                "can take a minute)"
            ):
                try:
                    answer_result, formatted = _answer_and_verify(question, settings)
                except Exception as exc:
                    st.error(f"Couldn't answer that question: {exc}")
                    answer_result, formatted = None, None
                else:
                    render_answer(answer_result, formatted)

    st.session_state["messages"].append(
        {"role": "assistant", "answer_result": answer_result, "formatted": formatted}
    )


def main() -> None:
    st.set_page_config(page_title="ScholarMind", layout="wide")
    st.title("ScholarMind")
    st.caption("A local, citation-verified research assistant for your PDF library.")

    settings = _effective_settings()
    render_sidebar(settings)
    render_ingest_panel(settings)
    render_chat_panel(settings)


if __name__ == "__main__":
    main()
