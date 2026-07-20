import io
import zipfile
from pathlib import Path

import streamlit as st

from scholarmind.agents.discovery import discover
from scholarmind.agents.figures import answer_about_figure
from scholarmind.agents.gap_analysis import analyze_gaps
from scholarmind.agents.llm_client import OpenRouterClient
from scholarmind.agents.methodology import extract_methodology
from scholarmind.agents.novelty import NoveltyCheckResult, check_novelty
from scholarmind.agents.qa import (
    AnswerResult,
    answer_question_streaming,
    finalize_streamed_answer,
)
from scholarmind.agents.summarization import summarize
from scholarmind.agents.writing import SECTION_TYPES, draft_section
from scholarmind.citations.export import export_bibtex, paper_to_metadata
from scholarmind.citations.formatter import format_reference
from scholarmind.citations.latex import build_latex_bundle
from scholarmind.citations.service import FormattedAndVerifiedAnswer, format_and_verify
from scholarmind.citations.verify import Citation, VerifiedAnswer, verify_citations
from scholarmind.citations.zotero import ZoteroError, push_references
from scholarmind.config import Settings, get_settings
from scholarmind.discovery.ingest import ingest_candidate
from scholarmind.discovery.models import Candidate, DiscoverySourceError
from scholarmind.discovery.service import get_citation_graph, search_external
from scholarmind.ingestion.pipeline import run_ingestion
from scholarmind.retrieval.dense import DenseResult
from scholarmind.retrieval.papers import (
    PaperSummary,
    delete_papers,
    get_paper_chunks,
    list_papers,
)
from scholarmind.webapp.library import get_library_stats, papers_dir_for


@st.cache_resource
def _load_llm_client(
    api_key: str, base_url: str, model: str, max_tokens: int
) -> OpenRouterClient:
    return OpenRouterClient(api_key=api_key, base_url=base_url, model=model, max_tokens=max_tokens)


def _effective_settings() -> Settings:
    base = get_settings()
    overrides = {}
    llm_key = st.session_state.get("llm_api_key_override", "")
    if llm_key:
        overrides["llm_api_key"] = llm_key
    zotero_key = st.session_state.get("zotero_api_key_override", "")
    if zotero_key:
        overrides["zotero_api_key"] = zotero_key
    zotero_library_id = st.session_state.get("zotero_library_id_override", "")
    if zotero_library_id:
        overrides["zotero_library_id"] = zotero_library_id
    zotero_library_type = st.session_state.get("zotero_library_type_override", "")
    if zotero_library_type:
        overrides["zotero_library_type"] = zotero_library_type
    if overrides:
        return base.model_copy(update=overrides)
    return base


def _refresh_library_stats(settings: "Settings") -> None:
    st.session_state["library_stats"] = get_library_stats(settings)


def _format_llm_error(exc: Exception, settings: "Settings") -> str:
    # openai-sdk exceptions expose .status_code on the HTTP-error subclasses; fall back
    # to just the model name when the exception is something else (e.g. a network error).
    status_code = getattr(exc, "status_code", None)
    message = str(exc)
    if settings.llm_api_key and settings.llm_api_key in message:
        message = message.replace(settings.llm_api_key, "***")
    status_part = f"status {status_code}, " if status_code is not None else ""
    return f"LLM request failed ({status_part}model: {settings.llm_model}): {message}"


def render_sidebar(settings: "Settings") -> None:
    with st.sidebar:
        # Brand — with top navigation the app name isn't shown anywhere else, so it
        # lives here, persistent across every page.
        st.title(":material/school: ScholarMind")
        st.caption("Local, citation-verified research assistant")
        st.divider()

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

        st.header("Zotero")
        if not (settings.zotero_api_key and settings.zotero_library_id):
            st.caption(
                "Optional — needed only to push references to Zotero. Set "
                "ZOTERO_API_KEY/ZOTERO_LIBRARY_ID in .env, or fill in below for this "
                "session only (never written to .env)."
            )
        zotero_key_input = st.text_input(
            "Zotero API key (this session only)", type="password", value=""
        )
        if zotero_key_input:
            st.session_state["zotero_api_key_override"] = zotero_key_input
        zotero_id_input = st.text_input("Zotero library ID (this session only)", value="")
        if zotero_id_input:
            st.session_state["zotero_library_id_override"] = zotero_id_input
        zotero_type_input = st.selectbox(
            "Zotero library type",
            ["user", "group"],
            index=0 if settings.zotero_library_type != "group" else 1,
        )
        st.session_state["zotero_library_type_override"] = zotero_type_input


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
        for title in result.duplicate_title_warnings:
            st.warning(
                f"'{title}' already appears to be in your library under a different ID — "
                "this may be a duplicate. Run `scholarmind dedupe` from the command line to "
                "review and remove the extra copy."
            )

    _refresh_library_stats(settings)


def render_library_panel(settings: "Settings") -> None:
    papers = list_papers(settings)
    if not papers:
        return

    pending_id = st.session_state.get("pending_delete_paper_id")

    # Force the expander open while a delete is pending — a Streamlit expander collapses on every
    # rerun, which would otherwise hide the confirmation prompt (and a collapsed expander's
    # buttons aren't reliably clickable) right when the user needs to confirm or cancel.
    with st.expander(
        f"Your library ({len(papers)} paper(s)) — manage & delete",
        expanded=pending_id is not None,
    ):
        st.caption(
            "Delete removes a paper's chunks from the search index so it's no longer retrieved "
            "or cited. The original PDF stays in your uploads folder, so you can re-ingest it "
            "later. This can't be undone from here."
        )

        for paper in papers:
            label = f"{paper.label} ({paper.chunk_count} chunk(s))"
            if paper.is_metadata_only:
                label += " — metadata only"
            cols = st.columns([5, 1])
            cols[0].markdown(label)
            # A first click only arms the delete (records which paper); the actual removal
            # needs a second, explicit confirmation below — so no paper is ever one click
            # away from deletion.
            if cols[1].button("Delete", key=f"lib_delete_{paper.paper_id}"):
                st.session_state["pending_delete_paper_id"] = paper.paper_id
                st.rerun()

        if pending_id is None:
            return

        target = next((p for p in papers if p.paper_id == pending_id), None)
        if target is None:
            # The armed paper is already gone (e.g. deleted from the CLI in parallel) — clear it.
            st.session_state.pop("pending_delete_paper_id", None)
            return

        st.divider()
        st.warning(
            f"Delete **{target.label}** ({target.chunk_count} chunk(s))? This removes it from "
            "the search index and can't be undone (the PDF file itself is kept)."
        )
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button("Confirm delete", key="lib_delete_confirm", type="primary"):
            removed = delete_papers([target.paper_id], settings)
            st.session_state.pop("pending_delete_paper_id", None)
            _refresh_library_stats(settings)
            # A toast survives the rerun below; an st.success() here would be wiped by it.
            st.toast(f"Deleted '{target.label}' ({removed} chunk(s) removed).", icon="🗑️")
            st.rerun()
        if cancel_col.button("Cancel", key="lib_delete_cancel"):
            st.session_state.pop("pending_delete_paper_id", None)
            st.rerun()


def _render_citation_content(citation: "Citation") -> None:
    # Tables/equations/figures are retrieved and cited exactly like text chunks, but render
    # differently in the Sources panel so their content is actually usable, not just a page ref.
    if citation.chunk_type == "table":
        markdown_part = citation.text.split("\n\n", 1)
        st.markdown(markdown_part[1] if len(markdown_part) > 1 else citation.text)
    elif citation.chunk_type == "equation":
        st.code(citation.text, language="text")
    elif citation.chunk_type == "figure":
        if citation.image_path and Path(citation.image_path).is_file():
            st.image(citation.image_path, width=240)
        else:
            st.caption("(figure image not available on disk)")


def _render_citations(citations: list["Citation"], invalid_markers: list[int]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for citation in citations:
            authors = ", ".join(citation.authors) if citation.authors else "Unknown author"
            year = citation.year if citation.year is not None else "n.d."
            section = f", {citation.section}" if citation.section else ""
            st.markdown(
                f"**[{citation.index}]** {citation.title or 'Untitled'} — {authors} "
                f"({year}){section}, pp. {citation.page_start}-{citation.page_end}"
            )
            _render_citation_content(citation)

    if invalid_markers:
        markers = ", ".join(f"[{m}]" for m in invalid_markers)
        st.warning(
            f"The model referenced source(s) {markers} that don't exist in the retrieved "
            "sources — they were not included above."
        )


def render_sources(answer_result: "AnswerResult") -> None:
    verified = answer_result.answer
    if verified is None:
        return
    _render_citations(verified.citations, verified.invalid_citation_markers)


def _render_verification_badge(formatted: "FormattedAndVerifiedAnswer") -> None:
    # Citation-verification is ScholarMind's whole point, so its result gets a prominent,
    # colour-coded badge instead of being buried in a collapsed expander title. Every cited
    # claim was re-checked against its own source passage (see citations/verifier.py).
    report = formatted.verification_report
    if not report.verifications:
        return

    total = len(report.verifications)
    supported = total - report.unsupported_count
    if report.unsupported_count == 0:
        st.badge(
            f"All {total} claim(s) verified against their sources",
            icon=":material/verified:",
            color="green",
        )
    else:
        with st.container(horizontal=True):
            st.badge(
                f"{report.unsupported_count} of {total} claim(s) unsupported",
                icon=":material/warning:",
                color="red",
            )
            st.badge(f"{supported}/{total} verified", color="gray")


def render_verification(formatted: "FormattedAndVerifiedAnswer") -> None:
    report = formatted.verification_report
    if not report.verifications:
        return

    _render_verification_badge(formatted)

    supported = len(report.verifications) - report.unsupported_count
    with st.expander(
        f"Verification details ({supported}/{len(report.verifications)} claims supported)",
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


def _stream_answer(
    question: str, settings: "Settings", paper_id: str | None
) -> tuple["AnswerResult | None", "FormattedAndVerifiedAnswer | None"]:
    # Streams the answer token-by-token into the current chat bubble, then verifies. Must be
    # called inside a `with st.chat_message("assistant"):` block. The tokens are the answer's
    # live feedback; a spinner covers the (quieter) retrieval and verification phases around it.
    # Returns (answer_result, formatted) for the caller to persist in chat history — on a later
    # rerun those replay through render_answer(), which re-renders the identical finished text.
    client = _load_llm_client(
        settings.llm_api_key, settings.llm_base_url, settings.llm_model, settings.llm_max_tokens
    )
    try:
        with st.spinner(
            "Searching your library… (the first question also loads the retrieval models, "
            "which can take a minute)"
        ):
            streaming = answer_question_streaming(question, client, settings, paper_id=paper_id)

        if streaming is None:
            answer_result = AnswerResult(question=question, answer=None, sources_found=0)
            render_answer(answer_result, None)  # shows the "no relevant sources" message
            return answer_result, None

        full_text = st.write_stream(streaming.tokens)

        with st.spinner("Verifying each cited claim against its source…"):
            answer_result = finalize_streamed_answer(question, full_text, streaming.sources)
            formatted = (
                format_and_verify(answer_result.answer, client)
                if answer_result.answer is not None
                else None
            )
    except Exception as exc:
        st.error(_format_llm_error(exc, settings))
        return None, None

    # The answer text already streamed above — render only the sources, verification badge,
    # and references here (render_answer would re-print the text).
    render_sources(answer_result)
    if formatted is not None:
        render_verification(formatted)

    return answer_result, formatted


def _paper_picker(
    settings: "Settings",
    label: str,
    key: str,
    none_option_label: str,
    default_to_most_recent: bool = False,
) -> "PaperSummary | None":
    papers = list_papers(settings)
    options: list[PaperSummary | None] = [None, *papers]
    default_index = 1 if (default_to_most_recent and papers) else 0
    return st.selectbox(
        label,
        options,
        index=default_index,
        format_func=lambda p: none_option_label
        if p is None
        else f"{p.label} ({p.chunk_count} chunk(s))",
        key=key,
    )


def render_chat_panel(settings: "Settings") -> None:
    st.subheader("Ask")

    if not list_papers(settings):
        st.info(
            "Your library is empty. Open the **Library** page to upload and ingest a PDF, "
            "then come back here to ask questions grounded in it.",
            icon=":material/library_books:",
        )

    scope = _paper_picker(settings, "Scope", "ask_scope_choice", none_option_label="All papers")
    scope_paper_id = scope.paper_id if scope is not None else None

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
            answer_result, formatted = _stream_answer(question, settings, scope_paper_id)

    st.session_state["messages"].append(
        {"role": "assistant", "answer_result": answer_result, "formatted": formatted}
    )


# --- Domain agent panels -----------------------------------------------------
# summarize / analyze_gaps / extract_methodology / draft_section / discover all
# share the same AgentResult(text, sources, sources_found) shape. The orchestrator's
# graph only runs citation verification for the "ask" intent (see graph.py's final
# node), so panels here compose the same verify_citations + format_and_verify calls
# the graph already uses for "ask", to give agent output the same Sources/
# Verification treatment.


def _run_agent_with_verification(
    agent_fn, query: str, settings: "Settings", **agent_kwargs
) -> tuple[str | None, "VerifiedAnswer | None", "FormattedAndVerifiedAnswer | None", int]:
    client = _load_llm_client(
        settings.llm_api_key, settings.llm_base_url, settings.llm_model, settings.llm_max_tokens
    )
    result = agent_fn(query, client, settings, **agent_kwargs)

    if not result.text:
        return None, None, None, result.sources_found

    verified = verify_citations(result.text, result.sources)
    formatted = format_and_verify(verified, client) if verified.citations else None
    return result.text, verified, formatted, result.sources_found


def _render_agent_panel(
    key: str,
    title: str,
    description: str,
    input_label: str,
    input_placeholder: str,
    button_label: str,
    agent_fn,
    settings: "Settings",
    empty_hint: str,
) -> None:
    if title:
        st.subheader(title)
    if description:
        st.caption(description)

    query = st.text_area(input_label, placeholder=input_placeholder, key=f"{key}_input")
    run_clicked = st.button(button_label, key=f"{key}_run")

    if not run_clicked:
        return

    if not query.strip():
        st.info(empty_hint)
        return

    if not settings.llm_api_key:
        st.info(
            "Add an LLM API key in the sidebar to use this tool — it needs to generate "
            "text, not just retrieve passages."
        )
        return

    with st.spinner(
        "Working… (the first call also loads the reranker model, which can take a minute)"
    ):
        try:
            text, verified, formatted, sources_found = _run_agent_with_verification(
                agent_fn, query, settings
            )
        except Exception as exc:
            st.error(_format_llm_error(exc, settings))
            return

    if text is None:
        st.info(f"No relevant sources found in your library for: {query}")
        return

    st.markdown(text)
    if verified is not None:
        _render_citations(verified.citations, verified.invalid_citation_markers)
        # Remembered so the References & export panel can offer a LaTeX bundle for the most
        # recent draft without recomputing it.
        st.session_state[f"{key}_last_verified"] = verified
    if formatted is not None:
        render_verification(formatted)
    if sources_found == 0:
        st.caption("No sources were retrieved for this request.")


_ACROSS_LIBRARY = "Across whole library (by topic)"


def render_summarize_panel(settings: "Settings") -> None:
    st.subheader("Summarize")
    st.caption(
        "Pick a paper to summarize just that paper, in full — the engine gathers all of "
        "its chunks in reading order rather than doing a topic search. Or switch to "
        "whole-library mode to summarize by topic across everything you've ingested."
    )

    choice = _paper_picker(
        settings,
        "Paper",
        "summarize_paper_choice",
        none_option_label=_ACROSS_LIBRARY,
        default_to_most_recent=True,
    )

    if choice is None:
        _render_agent_panel(
            key="summarize",
            title="",
            description="",
            input_label="Topic to summarize",
            input_placeholder="e.g. reinforcement learning from human feedback",
            button_label="Summarize",
            agent_fn=summarize,
            settings=settings,
            empty_hint="Enter a topic to summarize.",
        )
        return

    st.caption(f"Summarizing **{choice.label}** ({choice.chunk_count} chunk(s)).")

    if not st.button("Summarize", key="summarize_paper_run"):
        return

    if not settings.llm_api_key:
        st.info(
            "Add an LLM API key in the sidebar to use this tool — it needs to generate "
            "text, not just retrieve passages."
        )
        return

    with st.spinner(
        "Reading the paper and summarizing… (long papers take multiple passes and can "
        "be slow, and the first call also loads the reranker model)"
    ):
        try:
            text, verified, formatted, sources_found = _run_agent_with_verification(
                summarize, "", settings, paper_id=choice.paper_id
            )
        except Exception as exc:
            st.error(_format_llm_error(exc, settings))
            return

    if text is None:
        st.info("No chunks found for this paper — it may need to be re-ingested.")
        return

    st.markdown(text)
    if verified is not None:
        _render_citations(verified.citations, verified.invalid_citation_markers)
    if formatted is not None:
        render_verification(formatted)
    if sources_found == 0:
        st.caption("No sources were retrieved for this request.")


def render_gaps_panel(settings: "Settings") -> None:
    _render_agent_panel(
        key="gaps",
        title="Gap analysis",
        description=(
            "Synthesizes across your whole ingested library to surface themes, contradictions, "
            "and open questions, citing sources."
        ),
        input_label="Research area to analyze",
        input_placeholder="e.g. reinforcement learning from human feedback",
        button_label="Analyze gaps",
        agent_fn=analyze_gaps,
        settings=settings,
        empty_hint="Enter a research area to analyze.",
    )


def render_methodology_panel(settings: "Settings") -> None:
    _render_agent_panel(
        key="methods",
        title="Methodology advisor",
        description="Ask a free-text question about study design, methods, or statistics.",
        input_label="Your methodology question",
        input_placeholder="e.g. What sample size justification do these papers use?",
        button_label="Get advice",
        agent_fn=extract_methodology,
        settings=settings,
        empty_hint="Enter a methodology question.",
    )


def render_writing_panel(settings: "Settings") -> None:
    st.subheader("Writing")
    st.caption(
        "Drafts a cited section grounded ONLY in retrieved chunks. Any sentence without a "
        "[N] citation marker is automatically dropped before you see it — this tool never "
        "returns an uncited claim. A citation that exists can still turn out to be "
        "unsupported by its source; that's what Verification below checks separately."
    )

    section_type = st.selectbox(
        "Section type",
        SECTION_TYPES,
        format_func=lambda s: s.replace("_", " ").title(),
        key="write_section_type",
    )

    papers = list_papers(settings)
    scope_papers = st.multiselect(
        "Scope (leave empty for whole library)",
        papers,
        format_func=lambda p: f"{p.label} ({p.chunk_count} chunk(s))",
        key="write_scope_papers",
    )

    topic = st.text_area(
        "Topic / focus",
        placeholder="e.g. prompt injection defenses in retrieval-augmented systems",
        key="write_topic_input",
    )
    voice_notes = st.text_input(
        "Style/voice notes (optional)",
        placeholder="e.g. keep it formal, under 200 words, present tense",
        key="write_voice_notes_input",
    )

    run_clicked = st.button("Generate draft", key="write_run")

    if run_clicked:
        if not topic.strip():
            st.info("Enter a topic or focus for the section.")
        elif not settings.llm_api_key:
            st.info("Add an LLM API key in the sidebar to use this tool.")
        else:
            paper_ids = [p.paper_id for p in scope_papers] or None
            with st.spinner(
                "Drafting… (the first call also loads the reranker model, which can take "
                "a minute)"
            ):
                try:
                    text, verified, formatted, sources_found = _run_agent_with_verification(
                        draft_section,
                        topic,
                        settings,
                        section_type=section_type,
                        paper_ids=paper_ids,
                        voice_notes=voice_notes.strip() or None,
                    )
                except Exception as exc:
                    st.error(_format_llm_error(exc, settings))
                    text, verified, formatted, sources_found = None, None, None, 0

            if sources_found == 0:
                st.info(f"No relevant sources found for: {topic}")
            elif not text:
                st.warning(
                    "The draft had no citable claims after removing uncited sentences — "
                    "try a different topic, widen the scope, or rephrase the request."
                )
            else:
                st.markdown(text)
                if verified is not None:
                    _render_citations(verified.citations, verified.invalid_citation_markers)
                    st.session_state["write_last_verified"] = verified
                if formatted is not None:
                    render_verification(formatted)

    render_novelty_check_panel(settings)


def render_novelty_check_panel(settings: "Settings") -> None:
    last_draft = st.session_state.get("write_last_verified")
    if last_draft is None or not last_draft.text.strip():
        return

    st.divider()
    st.markdown("**Novelty / overlap check**")
    st.caption(
        "Advisory only — NOT a plagiarism verdict. Retrieves the closest content to the "
        "most recent draft (from your library, and optionally external literature) and "
        "gives a plain-language read on what it overlaps with and what looks distinct."
    )

    include_external = st.checkbox(
        "Also check external literature (arXiv / Semantic Scholar / OpenAlex)",
        key="novelty_include_external",
    )

    if not st.button("Check novelty of this draft", key="novelty_check_run"):
        return

    if not settings.llm_api_key:
        st.info("Add an LLM API key in the sidebar to use this tool.")
        return

    with st.spinner("Checking for overlaps…"):
        try:
            client = _load_llm_client(
                settings.llm_api_key,
                settings.llm_base_url,
                settings.llm_model,
                settings.llm_max_tokens,
            )
            result: NoveltyCheckResult = check_novelty(
                last_draft.text, client, settings, include_external_search=include_external
            )
        except Exception as exc:
            st.error(_format_llm_error(exc, settings))
            return

    if not result.library_overlaps and not result.external_overlaps:
        st.info(
            "No closely related content found — this looks novel relative to what's "
            "retrievable right now."
        )
        return

    st.warning("Advisory only — not a plagiarism verdict.")
    st.markdown(result.assessment)

    if result.library_overlaps:
        with st.expander(f"Library overlaps ({len(result.library_overlaps)})"):
            for index, source in enumerate(result.library_overlaps, start=1):
                authors = ", ".join(source.authors) if source.authors else "Unknown author"
                year = source.year if source.year is not None else "n.d."
                st.markdown(f"**[{index}]** {source.title or 'Untitled'} — {authors} ({year})")
                excerpt = source.text[:300] + ("…" if len(source.text) > 300 else "")
                st.caption(excerpt)

    if result.external_overlaps:
        with st.expander(f"External literature overlaps ({len(result.external_overlaps)})"):
            for candidate in result.external_overlaps:
                badge = _source_badge(candidate.source)
                st.markdown(f"**{candidate.title or 'Untitled'}** ({badge})")
                if candidate.abstract:
                    truncated = candidate.abstract[:300]
                    suffix = "…" if len(candidate.abstract) > 300 else ""
                    st.caption(truncated + suffix)

    for error in result.external_search_errors:
        st.warning(f"One external source is unavailable: {error}")


def render_library_browse_panel(settings: "Settings") -> None:
    st.caption(
        "Browses papers already in your library by topical relevance — it does not search "
        "external sources or fetch new papers. To add papers, use Upload & ingest above, or "
        "the Literature search / Citation graph tabs to find and ingest new ones."
    )

    query = st.text_input("Topic to browse your library for", key="discover_library_input")
    run_clicked = st.button("Browse library", key="discover_library_run")

    if not run_clicked:
        return

    if not query.strip():
        st.info("Enter a topic to browse your ingested papers for.")
        return

    with st.spinner("Searching your library…"):
        try:
            client = _load_llm_client(
                settings.llm_api_key,
                settings.llm_base_url,
                settings.llm_model,
                settings.llm_max_tokens,
            )
            result = discover(query, client, settings)
        except Exception as exc:
            st.error(_format_llm_error(exc, settings))
            return

    if not result.text:
        st.info(f"No matching papers found in your library for: {query}")
        return

    st.markdown(result.text)


_SOURCE_LABELS = {
    "arxiv": "arXiv",
    "semantic_scholar": "Semantic Scholar",
    "openalex": "OpenAlex",
}


def _source_badge(source: str) -> str:
    return "+".join(_SOURCE_LABELS.get(part, part) for part in source.split("+"))


def _candidate_caption(candidate: "Candidate") -> str:
    parts = [_source_badge(candidate.source)]
    if candidate.authors:
        author_names = ", ".join(candidate.authors[:3])
        if len(candidate.authors) > 3:
            author_names += " et al."
        parts.append(author_names)
    if candidate.year is not None:
        parts.append(str(candidate.year))
    if candidate.venue:
        parts.append(candidate.venue)
    if candidate.pdf_url:
        parts.append("open PDF available")
    else:
        parts.append("no open PDF — would ingest as a metadata-only record")
    return " · ".join(parts)


def _render_candidate_list(
    candidates: list["Candidate"], key_prefix: str, settings: "Settings"
) -> None:
    if not candidates:
        st.info("No results.")
        return

    for i, candidate in enumerate(candidates):
        cols = st.columns([5, 1])
        with cols[0]:
            st.markdown(f"**{candidate.title or 'Untitled'}**")
            st.caption(_candidate_caption(candidate))
            if candidate.abstract:
                with st.expander("Abstract"):
                    st.write(candidate.abstract)
        with cols[1]:
            if candidate.already_ingested:
                st.caption("Already in library")
            elif st.button("Ingest", key=f"{key_prefix}_ingest_{i}"):
                with st.spinner(f"Ingesting {candidate.title or 'this paper'}…"):
                    try:
                        result = ingest_candidate(candidate, settings)
                    except DiscoverySourceError as exc:
                        st.error(f"Couldn't ingest — {exc}")
                    else:
                        st.success(f"Ingested {result.chunks_created} chunk(s).")
                        _refresh_library_stats(settings)
        st.divider()


def render_literature_search_panel(settings: "Settings") -> None:
    st.caption(
        "Searches arXiv, Semantic Scholar, and OpenAlex for papers on a topic — not just your "
        "library. Duplicate results across sources are merged, and papers you've already "
        "ingested are flagged. Ingest a result with an open-access PDF, or as a title+abstract "
        "record when no open PDF is available."
    )

    query = st.text_input("Topic to search external literature for", key="discover_search_input")
    run_clicked = st.button("Search", key="discover_search_run")

    if not run_clicked:
        return

    if not query.strip():
        st.info("Enter a topic to search external literature for.")
        return

    with st.spinner(
        "Searching arXiv, Semantic Scholar, and OpenAlex… this can take a few seconds"
    ):
        result = search_external(query, settings)

    for error in result.errors:
        st.warning(f"One source is unavailable right now: {error}")

    _render_candidate_list(result.candidates, "search", settings)


def render_citation_graph_panel(settings: "Settings") -> None:
    st.caption(
        "Given a paper, shows what it cites (references) and what cites it (citing papers), "
        "one hop out, via Semantic Scholar. Duplicate results are flagged if already ingested."
    )

    papers = list_papers(settings)
    library_choice = st.selectbox(
        "Library paper",
        [None, *papers],
        format_func=lambda p: "— pick a library paper, or use the fields below —"
        if p is None
        else f"{p.label} ({p.chunk_count} chunk(s))",
        key="citation_graph_paper_choice",
    )
    doi_input = st.text_input(
        "…or a DOI",
        value=(library_choice.doi if library_choice and library_choice.doi else ""),
        key="citation_graph_doi_input",
    )
    s2_id_input = st.text_input("…or a Semantic Scholar paper ID", key="citation_graph_s2_input")

    run_clicked = st.button("Fetch citation graph", key="citation_graph_run")
    if not run_clicked:
        return

    title = library_choice.label if library_choice else None
    if not (doi_input.strip() or s2_id_input.strip() or title):
        st.info("Pick a library paper, or enter a DOI or Semantic Scholar paper ID.")
        return

    with st.spinner("Fetching references and citations from Semantic Scholar…"):
        graph = get_citation_graph(
            doi=doi_input.strip() or None,
            s2_paper_id=s2_id_input.strip() or None,
            title=title,
            settings=settings,
        )

    for error in graph.errors:
        st.warning(error)

    if graph.s2_paper_id is None:
        return

    st.caption(f"Semantic Scholar paper ID: `{graph.s2_paper_id}`")

    ref_tab, citing_tab = st.tabs(
        [f"References ({len(graph.references)})", f"Cited by ({len(graph.citing)})"]
    )
    with ref_tab:
        _render_candidate_list(graph.references, "refs", settings)
    with citing_tab:
        _render_candidate_list(graph.citing, "citing", settings)


def render_discover_panel(settings: "Settings") -> None:
    st.subheader("Discover")
    tabs = st.tabs(["My library", "Literature search", "Citation graph"])
    with tabs[0]:
        render_library_browse_panel(settings)
    with tabs[1]:
        render_literature_search_panel(settings)
    with tabs[2]:
        render_citation_graph_panel(settings)


def render_citations_panel(settings: "Settings") -> None:
    st.subheader("Citation / verify")
    st.caption(
        "Format references (APA + BibTeX) and run claim verification against a draft you "
        "paste in, using the same citation engine as Ask."
    )

    draft_text = st.text_area(
        "Paste text with [N]-style citation markers",
        placeholder="Retrieval-augmented generation grounds claims in retrieved passages [1]...",
        key="citations_draft_input",
        height=200,
    )
    sources_text = st.text_area(
        "Paste matching source list (one per line: N | title | authors | year | section | pages)",
        placeholder="1 | Attention Is All You Need | Vaswani et al. | 2017 | Introduction | 1-2",
        key="citations_sources_input",
        height=150,
    )
    run_clicked = st.button("Verify & format", key="citations_run")

    if not run_clicked:
        return

    if not draft_text.strip() or not sources_text.strip():
        st.info("Paste both the draft text and its numbered source list.")
        return

    if not settings.llm_api_key:
        st.info("Add an LLM API key in the sidebar — claim verification needs one.")
        return

    sources: list[DenseResult] = []
    try:
        for line in sources_text.strip().splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 6:
                raise ValueError(f"expected 6 '|'-separated fields, got {len(parts)}: {line!r}")
            index_str, title, authors, year_str, section, pages = parts
            page_start_str, _, page_end_str = pages.partition("-")
            sources.append(
                DenseResult(
                    paper_id=f"pasted-{index_str}",
                    title=title,
                    authors=[a.strip() for a in authors.split(",") if a.strip()],
                    year=int(year_str) if year_str.strip().isdigit() else None,
                    venue=None,
                    section=section or None,
                    page_start=int(page_start_str) if page_start_str.strip().isdigit() else 0,
                    page_end=int(page_end_str) if page_end_str.strip().isdigit() else 0,
                    chunk_index=0,
                    text="",
                    score=0.0,
                )
            )
    except ValueError as exc:
        st.error(f"Couldn't parse the source list: {exc}")
        return

    with st.spinner("Verifying citations and formatting references…"):
        try:
            client = _load_llm_client(
                settings.llm_api_key,
                settings.llm_base_url,
                settings.llm_model,
                settings.llm_max_tokens,
            )
            verified = verify_citations(draft_text, sources)
            formatted = format_and_verify(verified, client) if verified.citations else None
        except Exception as exc:
            st.error(_format_llm_error(exc, settings))
            return

    _render_citations(verified.citations, verified.invalid_citation_markers)
    if formatted is not None:
        render_verification(formatted)
    elif not verified.citations:
        st.info("No [N] citation markers found in the pasted text.")


_CITATION_STYLES = ["apa", "mla", "chicago", "ieee", "vancouver"]


def _selected_or_all_papers(
    settings: "Settings",
) -> tuple[list["PaperSummary"], list["PaperSummary"]]:
    # Returns (all_papers, selected_papers) — selected defaults to all when nothing is checked.
    papers = list_papers(settings)
    if not papers:
        return papers, papers

    st.caption("Select papers (none selected = whole library):")
    chosen = []
    for paper in papers:
        label = f"{paper.label} ({paper.chunk_count} chunk(s))"
        if paper.is_metadata_only:
            label += " — metadata only"
        if st.checkbox(label, key=f"refexport_select_{paper.paper_id}"):
            chosen.append(paper)
    return papers, (chosen or papers)


def render_references_export_panel(settings: "Settings") -> None:
    st.subheader("References & export")
    st.caption(
        "Format and export references from your library in any supported style, push them "
        "to Zotero, or export a Writing-agent draft as a LaTeX/Overleaf bundle."
    )

    style = st.selectbox("Citation style", _CITATION_STYLES, key="refexport_style")

    all_papers, selected_papers = _selected_or_all_papers(settings)
    if not all_papers:
        st.info("Ingest a paper first — there's nothing in your library to export yet.")
        return

    st.markdown("**Preview**")
    with st.spinner("Resolving reference metadata (Crossref/OpenAlex/Semantic Scholar)…"):
        for paper in selected_papers:
            try:
                st.markdown(f"- {format_reference(paper_to_metadata(paper, settings), style)}")
            except ValueError as exc:
                st.error(str(exc))
                return

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**BibTeX export**")
        bib_content = export_bibtex(selected_papers, settings)
        st.download_button(
            "Export .bib",
            data=bib_content,
            file_name="scholarmind_references.bib",
            mime="text/x-bibtex",
            key="refexport_bib_download",
        )

    with col2:
        st.markdown("**Zotero**")
        if not (settings.zotero_api_key and settings.zotero_library_id):
            st.caption("Not configured — add a Zotero API key and library ID in the sidebar.")
        elif st.button("Push selected to Zotero", key="refexport_zotero_push"):
            with st.spinner("Pushing to Zotero…"):
                try:
                    result = push_references(
                        [paper_to_metadata(p, settings) for p in selected_papers],
                        api_key=settings.zotero_api_key,
                        library_id=settings.zotero_library_id,
                        library_type=settings.zotero_library_type,
                    )
                except ZoteroError as exc:
                    st.error(f"Couldn't push to Zotero — {exc}")
                else:
                    if result.failed:
                        st.warning(
                            f"Pushed {result.pushed} reference(s); {result.failed} failed: "
                            + "; ".join(result.errors)
                        )
                    else:
                        st.success(f"Pushed {result.pushed} reference(s) to Zotero.")

    st.divider()
    st.markdown("**LaTeX / Overleaf export**")
    st.caption(
        "Exports the most recent Writing-agent draft as a .tex file (with [N] markers "
        "replaced by \\cite{} commands) plus a matching .bib file, zipped together. Upload "
        "both files from the zip to Overleaf."
    )

    last_draft = st.session_state.get("write_last_verified")
    if last_draft is None or not last_draft.citations:
        st.info("Draft a section with citations in the Writing tab first.")
        return

    bundle = build_latex_bundle(last_draft.text, last_draft.citations)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("draft.tex", bundle.tex)
        archive.writestr("references.bib", bundle.bib)

    st.download_button(
        "Export draft as LaTeX bundle (.zip)",
        data=buffer.getvalue(),
        file_name="scholarmind_draft_latex.zip",
        mime="application/zip",
        key="refexport_latex_download",
    )


def render_figures_tables_panel(settings: "Settings") -> None:
    st.subheader("Figures & tables")
    st.caption(
        "Browse one paper's extracted tables, equations, and figures. Extraction is "
        "layout-heuristic (PyMuPDF table/image detection; a regex-based equation "
        "heuristic) — real but not guaranteed complete, and a page with several "
        "tables/figures may occasionally mismatch a caption to the wrong one."
    )

    papers = list_papers(settings)
    if not papers:
        st.info("Ingest a paper first.")
        return

    choice = st.selectbox(
        "Paper",
        papers,
        format_func=lambda p: f"{p.label} ({p.chunk_count} chunk(s))",
        key="figtab_paper_choice",
    )

    chunks = get_paper_chunks(choice.paper_id, settings)
    tables = [c for c in chunks if c.chunk_type == "table"]
    equations = [c for c in chunks if c.chunk_type == "equation"]
    figures = [c for c in chunks if c.chunk_type == "figure"]

    if not tables and not equations and not figures:
        st.info("No tables, equations, or figures were extracted for this paper.")
        return

    tabs = st.tabs(
        [f"Tables ({len(tables)})", f"Equations ({len(equations)})", f"Figures ({len(figures)})"]
    )

    with tabs[0]:
        if not tables:
            st.info("No tables extracted for this paper.")
        for table in tables:
            caption = f"Page {table.page_start}"
            if table.section:
                caption += f" — {table.section}"
            st.caption(caption)
            markdown_part = table.text.split("\n\n", 1)
            st.markdown(markdown_part[1] if len(markdown_part) > 1 else table.text)
            st.divider()

    with tabs[1]:
        if not equations:
            st.info("No equations detected for this paper.")
        for equation in equations:
            st.caption(f"Page {equation.page_start}")
            st.code(equation.text, language="text")
            st.divider()

    with tabs[2]:
        if not figures:
            st.info("No figures extracted for this paper.")
        if not settings.vision_model:
            st.caption(
                "No VISION_MODEL configured — figure questions below use the caption only, "
                "not the image itself."
            )
        for i, figure in enumerate(figures):
            caption_line = figure.section or f"Figure (page {figure.page_start})"
            st.markdown(f"**Page {figure.page_start}** — {caption_line}")
            if figure.image_path and Path(figure.image_path).is_file():
                st.image(figure.image_path, width=280)
            else:
                st.caption("(image not available on disk)")

            question_key = f"figtab_q_{choice.paper_id}_{i}"
            question = st.text_input("Ask about this figure", key=question_key)
            if st.button("Ask", key=f"figtab_ask_{choice.paper_id}_{i}"):
                if not question.strip():
                    st.info("Enter a question about this figure.")
                elif not settings.llm_api_key:
                    st.info("Add an LLM API key in the sidebar to use this tool.")
                else:
                    with st.spinner("Answering…"):
                        try:
                            client = _load_llm_client(
                                settings.llm_api_key,
                                settings.llm_base_url,
                                settings.llm_model,
                                settings.llm_max_tokens,
                            )
                            result = answer_about_figure(figure, question, client, settings)
                        except Exception as exc:
                            st.error(_format_llm_error(exc, settings))
                        else:
                            st.markdown(result.text)
            st.divider()


# --- Pages ------------------------------------------------------------------
# The app is a st.navigation multi-page app: five focused pages instead of one
# long scroll. Each page is a zero-arg callable (required by st.navigation) that
# re-derives settings itself, and composes the same render_*_panel functions the
# app has always used — the pages only regroup them by user intent.


def _ask_page() -> None:
    render_chat_panel(_effective_settings())


def _library_page() -> None:
    # Everything about the papers you already have: add them, manage/delete them,
    # and browse their extracted tables/equations/figures.
    settings = _effective_settings()
    render_ingest_panel(settings)
    render_library_panel(settings)
    st.divider()
    render_figures_tables_panel(settings)


def _analyze_page() -> None:
    # Read and understand your library: summaries, cross-paper gaps, methodology Q&A.
    settings = _effective_settings()
    summarize_tab, gaps_tab, methods_tab = st.tabs(
        ["Summarize", "Gap analysis", "Methodology"]
    )
    with summarize_tab:
        render_summarize_panel(settings)
    with gaps_tab:
        render_gaps_panel(settings)
    with methods_tab:
        render_methodology_panel(settings)


def _write_page() -> None:
    # Produce cited output: draft sections, format/verify references, export.
    settings = _effective_settings()
    writing_tab, references_tab, verify_tab = st.tabs(
        ["Writing", "References & export", "Citation / verify"]
    )
    with writing_tab:
        render_writing_panel(settings)
    with references_tab:
        render_references_export_panel(settings)
    with verify_tab:
        render_citations_panel(settings)


def _discover_page() -> None:
    # Find new papers (browse your library, external literature search, citation graph).
    render_discover_panel(_effective_settings())


def main() -> None:
    st.set_page_config(
        page_title="ScholarMind", page_icon=":material/school:", layout="wide"
    )

    settings = _effective_settings()
    render_sidebar(settings)

    page = st.navigation(
        [
            st.Page(_ask_page, title="Ask", icon=":material/forum:", default=True),
            st.Page(
                _library_page,
                title="Library",
                icon=":material/library_books:",
                url_path="library",
            ),
            st.Page(
                _analyze_page,
                title="Analyze",
                icon=":material/analytics:",
                url_path="analyze",
            ),
            st.Page(
                _write_page, title="Write", icon=":material/edit_note:", url_path="write"
            ),
            st.Page(
                _discover_page,
                title="Discover",
                icon=":material/travel_explore:",
                url_path="discover",
            ),
        ],
        position="top",
    )
    page.run()


if __name__ == "__main__":
    main()
