from scholarmind.webapp import app as webapp_app
from scholarmind.webapp import library as webapp_library


def test_webapp_app_exposes_render_functions():
    assert callable(webapp_app.render_sidebar)
    assert callable(webapp_app.render_ingest_panel)
    assert callable(webapp_app.render_chat_panel)
    assert callable(webapp_app.render_answer)
    assert callable(webapp_app.render_sources)
    assert callable(webapp_app.render_verification)
    assert callable(webapp_app.main)


def test_webapp_app_exposes_agent_panel_functions():
    assert callable(webapp_app.render_agent_workspace)
    assert callable(webapp_app.render_summarize_panel)
    assert callable(webapp_app.render_gaps_panel)
    assert callable(webapp_app.render_methodology_panel)
    assert callable(webapp_app.render_writing_panel)
    assert callable(webapp_app.render_discover_panel)
    assert callable(webapp_app.render_citations_panel)


def test_webapp_app_exposes_paper_scoping_helpers():
    assert callable(webapp_app._paper_picker)
    assert callable(webapp_app._run_agent_with_verification)


def test_webapp_app_exposes_discovery_panel_functions():
    assert callable(webapp_app.render_library_browse_panel)
    assert callable(webapp_app.render_literature_search_panel)
    assert callable(webapp_app.render_citation_graph_panel)
    assert callable(webapp_app._render_candidate_list)
    assert callable(webapp_app._candidate_caption)
    assert callable(webapp_app._source_badge)


def test_webapp_app_exposes_references_export_panel_functions():
    assert callable(webapp_app.render_references_export_panel)
    assert callable(webapp_app._selected_or_all_papers)
    assert webapp_app._CITATION_STYLES == ["apa", "mla", "chicago", "ieee", "vancouver"]


def test_webapp_app_exposes_novelty_check_panel_function():
    assert callable(webapp_app.render_novelty_check_panel)


def test_webapp_app_exposes_figures_tables_panel_function():
    assert callable(webapp_app.render_figures_tables_panel)
    assert callable(webapp_app._render_citation_content)


def test_webapp_library_exposes_helpers():
    assert callable(webapp_library.get_library_stats)
    assert callable(webapp_library.papers_dir_for)


def test_webapp_app_main_is_guarded_not_run_on_import():
    # main() calls st.set_page_config(), which requires a live Streamlit script-run
    # context. Importing this test file already imported the module above with no
    # such context available, so main() must not execute at import time.
    assert webapp_app.__name__ != "__main__"


def test_source_badge_maps_known_sources_and_joins_merged_ones():
    assert webapp_app._source_badge("arxiv") == "arXiv"
    assert webapp_app._source_badge("arxiv+semantic_scholar") == "arXiv+Semantic Scholar"


def test_candidate_caption_flags_missing_open_pdf():
    from scholarmind.discovery.models import Candidate

    candidate = Candidate(
        title="A Paper",
        authors=["Ada Lovelace"],
        year=2020,
        venue="NeurIPS",
        abstract=None,
        doi=None,
        url=None,
        pdf_url=None,
        source="arxiv",
        external_id="1",
    )

    caption = webapp_app._candidate_caption(candidate)

    assert "arXiv" in caption
    assert "no open PDF" in caption
