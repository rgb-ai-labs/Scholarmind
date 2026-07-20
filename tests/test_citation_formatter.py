import pytest

from scholarmind.citations.formatter import (
    APAFormatter,
    BibTeXFormatter,
    ChicagoFormatter,
    IEEEFormatter,
    MLAFormatter,
    VancouverFormatter,
    _split_author_name,
    bibtex_key,
    format_reference,
    unique_bibtex_key,
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


def test_mla_formatter_full_metadata():
    result = MLAFormatter().format(_full_metadata())
    assert "Lovelace, Ada, and Grace Hopper." in result
    assert '"A Study of Retrieval-Augmented Generation."' in result
    assert "Advanced Retrieval," in result
    assert "2024." in result
    assert "https://doi.org/10.1234/example." in result


def test_mla_formatter_three_plus_authors_uses_et_al():
    metadata = NormalizedMetadata(
        doi=None,
        title="Some Title",
        authors=["Ada Lovelace", "Grace Hopper", "Alan Turing"],
        year=2020,
        venue=None,
        source="unresolved",
    )
    result = MLAFormatter().format(metadata)
    assert "Lovelace, Ada, et al." in result


def test_mla_formatter_no_authors_omits_leading_comma():
    metadata = NormalizedMetadata(
        doi=None, title="Some Title", authors=[], year=2020, venue=None, source="unresolved"
    )
    result = MLAFormatter().format(metadata)
    assert result.startswith('"Some Title."')


def test_chicago_formatter_full_metadata():
    result = ChicagoFormatter().format(_full_metadata())
    assert "Lovelace, Ada, and Grace Hopper. 2024." in result
    assert '"A Study of Retrieval-Augmented Generation."' in result
    assert "Advanced Retrieval." in result


def test_chicago_formatter_no_year_uses_nd():
    metadata = NormalizedMetadata(
        doi=None, title="Some Title", authors=["Ada Lovelace"], year=None, venue=None, source="x"
    )
    result = ChicagoFormatter().format(metadata)
    assert "n.d." in result


def test_ieee_formatter_full_metadata():
    result = IEEEFormatter().format(_full_metadata())
    assert "A. Lovelace, and G. Hopper," in result
    assert '"A Study of Retrieval-Augmented Generation,"' in result
    assert "doi: 10.1234/example." in result


def test_ieee_formatter_four_plus_authors_uses_et_al_without_and():
    metadata = NormalizedMetadata(
        doi=None,
        title="Some Title",
        authors=["Ada Lovelace", "Grace Hopper", "Alan Turing", "Katherine Johnson"],
        year=2020,
        venue=None,
        source="unresolved",
    )
    result = IEEEFormatter().format(metadata)
    assert result.startswith('A. Lovelace, G. Hopper, A. Turing, et al., "Some Title,"')
    assert "and" not in result


def test_vancouver_formatter_full_metadata():
    result = VancouverFormatter().format(_full_metadata())
    assert "Lovelace A, Hopper G." in result
    assert "A Study of Retrieval-Augmented Generation." in result
    assert '"' not in result  # Vancouver titles are not quoted


def test_vancouver_formatter_seven_plus_authors_uses_et_al():
    metadata = NormalizedMetadata(
        doi=None,
        title="Some Title",
        authors=[f"Author{i} Family{i}" for i in range(7)],
        year=2020,
        venue=None,
        source="unresolved",
    )
    result = VancouverFormatter().format(metadata)
    assert "et al" in result


def test_bibtex_key_derives_from_first_author_and_year():
    assert bibtex_key(_full_metadata()) == "lovelace2024"


def test_bibtex_key_no_authors_or_year():
    metadata = NormalizedMetadata(
        doi=None, title="Some Title", authors=[], year=None, venue=None, source="x"
    )
    assert bibtex_key(metadata) == "anonnd"


def test_unique_bibtex_key_disambiguates_collisions():
    metadata = _full_metadata()
    used: set[str] = set()

    first = unique_bibtex_key(metadata, used)
    used.add(first)
    second = unique_bibtex_key(metadata, used)
    used.add(second)
    third = unique_bibtex_key(metadata, used)

    assert first == "lovelace2024"
    assert second == "lovelace2024a"
    assert third == "lovelace2024b"


def test_format_reference_case_insensitive():
    metadata = _full_metadata()
    result_lower = format_reference(metadata, "apa")
    result_upper = format_reference(metadata, "APA")
    assert result_lower == result_upper


@pytest.mark.parametrize("style", ["apa", "mla", "chicago", "ieee", "vancouver", "bibtex"])
def test_format_reference_all_styles_registered(style):
    result = format_reference(_full_metadata(), style)
    assert isinstance(result, str)
    assert result.strip() != ""


def test_format_reference_unknown_style_raises():
    metadata = _full_metadata()
    with pytest.raises(ValueError):
        format_reference(metadata, "harvard")
