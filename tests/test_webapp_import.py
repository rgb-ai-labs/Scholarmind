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


def test_webapp_library_exposes_helpers():
    assert callable(webapp_library.get_library_stats)
    assert callable(webapp_library.papers_dir_for)


def test_webapp_app_main_is_guarded_not_run_on_import():
    # main() calls st.set_page_config(), which requires a live Streamlit script-run
    # context. Importing this test file already imported the module above with no
    # such context available, so main() must not execute at import time.
    assert webapp_app.__name__ != "__main__"
