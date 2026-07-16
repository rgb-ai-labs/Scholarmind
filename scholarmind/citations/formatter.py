from typing import Protocol

from scholarmind.citations.metadata import NormalizedMetadata


class ReferenceFormatter(Protocol):
    def format(self, metadata: "NormalizedMetadata") -> str: ...


class APAFormatter:
    def format(self, metadata: "NormalizedMetadata") -> str:
        authors_part = _apa_authors(metadata.authors)
        year_part = f"({metadata.year})" if metadata.year is not None else "(n.d.)"
        title = metadata.title if metadata.title else "Untitled"
        if metadata.venue:
            title_part = f"{title}. {metadata.venue}."
        else:
            title_part = f"{title}."
        doi_part = f" https://doi.org/{metadata.doi}" if metadata.doi else ""
        return f"{authors_part} {year_part}. {title_part}{doi_part}"


class BibTeXFormatter:
    def format(self, metadata: "NormalizedMetadata") -> str:
        if metadata.authors:
            _, first_family = _split_author_name(metadata.authors[0])
        else:
            first_family = "anon"
        family_key = "".join(ch for ch in first_family.lower() if ch.isalnum())
        year_key = str(metadata.year) if metadata.year is not None else "nd"
        key = f"{family_key}{year_key}"

        fields: list[tuple[str, str]] = []
        if metadata.authors:
            author_value = " and ".join(
                f"{family}, {given}" if given else family
                for given, family in (_split_author_name(a) for a in metadata.authors)
            )
            fields.append(("author", author_value))
        fields.append(("title", metadata.title if metadata.title else "Untitled"))
        if metadata.year is not None:
            fields.append(("year", str(metadata.year)))
        if metadata.venue:
            fields.append(("journal", metadata.venue))
        if metadata.doi:
            fields.append(("doi", metadata.doi))

        body = "".join(f"  {name}={{{value}}},\n" for name, value in fields)
        return f"@article{{{key},\n{body}}}"


_FORMATTERS: dict[str, "ReferenceFormatter"] = {
    "apa": APAFormatter(),
    "bibtex": BibTeXFormatter(),
}


def format_reference(metadata: "NormalizedMetadata", style: str) -> str:
    formatter = _FORMATTERS.get(style.lower())
    if formatter is None:
        raise ValueError(f"Unknown citation style: {style!r}")
    return formatter.format(metadata)


def _split_author_name(author: str) -> tuple[str, str]:
    tokens = author.split()
    if not tokens:
        return "", ""
    if len(tokens) == 1:
        return "", tokens[0]
    return " ".join(tokens[:-1]), tokens[-1]


def _apa_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown Author"
    formatted = []
    for author in authors:
        given, family = _split_author_name(author)
        if given:
            formatted.append(f"{family}, {given[0].upper()}.")
        else:
            formatted.append(family)
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} & {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


# TODO: Zotero sync hook -- export formatted references via Zotero's Web API
