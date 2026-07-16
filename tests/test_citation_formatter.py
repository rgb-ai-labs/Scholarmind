import pytest

from scholarmind.citations.formatter import (
    APAFormatter,
    BibTeXFormatter,
    _split_author_name,
    format_reference,
)
from scholarmind.citations.metadata import NormalizedMetadata


def test_split_author_name_two_tokens():
    assert _split_author_name("Ada Lovelace") == ("Ada", "Lovelace")


def test_split_author_name_single_token():
    assert _split_author_name("Cher") == ("", "Cher")


def test_split_author_name_empty():
    assert _split_author_name("") == ("", "")


def _full_metadata() -> NormalizedMetadata:
    return NormalizedMetadata(
        doi="10.1234/example",
        title="A Study of Retrieval-Augmented Generation",
        authors=["Ada Lovelace", "Grace Hopper"],
        year=2024,
        venue="Advanced Retrieval",
        source="crossref",
    )


def test_apa_formatter_full_metadata():
    result = APAFormatter().format(_full_metadata())
    assert "Lovelace, A." in result
    assert "& Hopper, G." in result
    assert "(2024)" in result
    assert "A Study of Retrieval-Augmented Generation" in result
    assert "Advanced Retrieval" in result
    assert "https://doi.org/10.1234/example" in result


def test_apa_formatter_no_authors():
    metadata = NormalizedMetadata(
        doi=None, title="Some Title", authors=[], year=2020, venue=None, source="unresolved"
    )
    result = APAFormatter().format(metadata)
    assert "Unknown Author" in result


def test_apa_formatter_no_year():
    metadata = NormalizedMetadata(
        doi=None,
        title="Some Title",
        authors=["Ada Lovelace"],
        year=None,
        venue=None,
        source="unresolved",
    )
    result = APAFormatter().format(metadata)
    assert "(n.d.)" in result


def test_apa_formatter_no_venue_no_doi_no_none_leak():
    metadata = NormalizedMetadata(
        doi=None,
        title="Some Title",
        authors=["Ada Lovelace"],
        year=2020,
        venue=None,
        source="unresolved",
    )
    result = APAFormatter().format(metadata)
    assert "None" not in result
    assert "  " not in result
    assert ".." not in result
    assert not result.endswith(" .")
    assert result.strip() == result


def test_apa_formatter_three_authors_joining():
    metadata = NormalizedMetadata(
        doi=None,
        title="Some Title",
        authors=["Ada Lovelace", "Grace Hopper", "Alan Turing"],
        year=2020,
        venue=None,
        source="unresolved",
    )
    result = APAFormatter().format(metadata)
    assert "Lovelace, A., Hopper, G., & Turing, A." in result


def test_bibtex_formatter_full_metadata():
    result = BibTeXFormatter().format(_full_metadata())
    assert result.startswith("@article{")
    assert "lovelace2024" in result
    assert "author={" in result
    assert "title={" in result
    assert "year={" in result
    assert "journal={" in result
    assert "doi={" in result
    assert result.endswith("}")


def test_bibtex_formatter_omits_missing_fields():
    metadata = NormalizedMetadata(
        doi=None,
        title="Some Title",
        authors=["Ada Lovelace"],
        year=None,
        venue=None,
        source="unresolved",
    )
    result = BibTeXFormatter().format(metadata)
    assert "year={" not in result
    assert "journal={" not in result
    assert "doi={" not in result


def test_bibtex_formatter_no_authors_key():
    metadata = NormalizedMetadata(
        doi=None, title="Some Title", authors=[], year=2020, venue=None, source="unresolved"
    )
    result = BibTeXFormatter().format(metadata)
    assert "anon2020" in result


def test_format_reference_case_insensitive():
    metadata = _full_metadata()
    result_lower = format_reference(metadata, "apa")
    result_upper = format_reference(metadata, "APA")
    assert result_lower == result_upper


def test_format_reference_unknown_style_raises():
    metadata = _full_metadata()
    with pytest.raises(ValueError):
        format_reference(metadata, "mla")
