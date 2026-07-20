from pathlib import Path

from scholarmind.ingestion.multimodal import extract_equations, extract_tables_and_figures

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper_multimodal.pdf"
TEXT_ONLY_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


def test_extract_tables_and_figures_finds_table_with_caption(tmp_path: Path):
    tables, figures = extract_tables_and_figures(FIXTURE_PATH, tmp_path / "images")

    assert len(tables) == 1
    table = tables[0]
    assert table.page == 1
    assert table.caption is not None
    assert "Table 1" in table.caption
    assert "Configuration" in table.markdown
    assert "Hybrid + reranker" in table.markdown


def test_extract_tables_and_figures_finds_figure_and_saves_image(tmp_path: Path):
    tables, figures = extract_tables_and_figures(FIXTURE_PATH, tmp_path / "images")

    assert len(figures) == 1
    figure = figures[0]
    assert figure.page == 1
    assert figure.caption is not None
    assert "Figure 1" in figure.caption
    assert Path(figure.image_path).is_file()
    assert Path(figure.image_path).stat().st_size > 0


def test_extract_tables_and_figures_on_text_only_pdf_returns_nothing_but_does_not_raise(
    tmp_path: Path,
):
    tables, figures = extract_tables_and_figures(TEXT_ONLY_FIXTURE_PATH, tmp_path / "images")

    assert tables == []
    assert figures == []


def test_extract_tables_and_figures_missing_file_does_not_raise(tmp_path: Path):
    tables, figures = extract_tables_and_figures(tmp_path / "does_not_exist.pdf", tmp_path / "images")

    assert tables == []
    assert figures == []


def test_extract_tables_and_figures_missing_pymupdf_falls_back_gracefully(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "pymupdf":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    tables, figures = extract_tables_and_figures(FIXTURE_PATH, tmp_path / "images")

    assert tables == []
    assert figures == []


def test_extract_equations_finds_numbered_equation_with_context():
    pages = [
        "Section 2\nThe final relevance score combines dense and sparse signals as follows.\n"
        "score(q, d) = a * dense(q, d) + b * sparse(q, d) (1)\n"
        "where a and b are tunable weights."
    ]

    equations = extract_equations(pages)

    assert len(equations) == 1
    equation = equations[0]
    assert equation.page == 1
    assert equation.text == "score(q, d) = a * dense(q, d) + b * sparse(q, d) (1)"
    assert "tunable weights" in equation.context
    assert "combines dense and sparse" in equation.context


def test_extract_equations_ignores_ordinary_numbered_prose():
    pages = ["This is the third point discussed in this paper (1)."]

    # No math symbol present, so this should NOT be flagged as an equation.
    assert extract_equations(pages) == []


def test_extract_equations_ignores_lines_without_trailing_number():
    pages = ["x = y + z, an equation with no trailing equation number."]

    assert extract_equations(pages) == []


def test_extract_equations_handles_multiple_pages():
    pages = [
        "no equations here",
        "E = mc^2 (2)",
    ]

    equations = extract_equations(pages)

    assert len(equations) == 1
    assert equations[0].page == 2
