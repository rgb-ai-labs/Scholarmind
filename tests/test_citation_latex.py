from scholarmind.citations.latex import build_latex_bundle
from scholarmind.citations.verify import Citation


def _citation(index: int, title: str, authors: list[str], year: int) -> Citation:
    return Citation(
        index=index,
        paper_id=f"paper-{index}",
        title=title,
        authors=authors,
        year=year,
        section=None,
        page_start=1,
        page_end=1,
        text="body text",
    )


def test_build_latex_bundle_replaces_single_marker_with_cite():
    citations = [_citation(1, "Attention Is All You Need", ["Ashish Vaswani"], 2017)]
    bundle = build_latex_bundle("Transformers changed NLP [1].", citations)

    assert "\\cite{vaswani2017}" in bundle.tex
    assert "[1]" not in bundle.tex
    assert "@article{vaswani2017," in bundle.bib


def test_build_latex_bundle_replaces_combined_marker_with_multiple_keys():
    citations = [
        _citation(1, "Paper One", ["Ada Lovelace"], 2020),
        _citation(2, "Paper Two", ["Grace Hopper"], 2021),
    ]
    bundle = build_latex_bundle("Both agree [1, 2].", citations)

    assert "\\cite{lovelace2020,hopper2021}" in bundle.tex
    assert bundle.bib.count("@article{") == 2


def test_build_latex_bundle_leaves_unresolvable_marker_untouched():
    citations = [_citation(1, "Paper One", ["Ada Lovelace"], 2020)]
    bundle = build_latex_bundle("See [1] and also [99].", citations)

    assert "\\cite{lovelace2020}" in bundle.tex
    assert "[99]" in bundle.tex


def test_build_latex_bundle_disambiguates_same_author_year_across_citations():
    citations = [
        _citation(1, "Paper One", ["Ada Lovelace"], 2020),
        _citation(2, "Paper Two", ["Ada Lovelace"], 2020),
    ]
    bundle = build_latex_bundle("First [1], second [2].", citations)

    assert "\\cite{lovelace2020}" in bundle.tex
    assert "\\cite{lovelace2020a}" in bundle.tex
    assert "@article{lovelace2020," in bundle.bib
    assert "@article{lovelace2020a," in bundle.bib


def test_build_latex_bundle_escapes_tex_special_characters_in_body():
    citations = [_citation(1, "Paper One", ["Ada Lovelace"], 2020)]
    bundle = build_latex_bundle("Cost was 50% & $10 for item_1 [1].", citations)

    assert r"50\%" in bundle.tex
    assert r"\&" in bundle.tex
    assert r"\$10" in bundle.tex
    assert r"item\_1" in bundle.tex
    assert "\\cite{lovelace2020}" in bundle.tex


def test_build_latex_bundle_wraps_valid_document_structure():
    bundle = build_latex_bundle("No citations here.", [])

    assert bundle.tex.startswith("\\documentclass{article}")
    assert "\\begin{document}" in bundle.tex
    assert "\\end{document}" in bundle.tex
    assert "\\bibliography{references}" in bundle.tex
    assert bundle.bib == ""
