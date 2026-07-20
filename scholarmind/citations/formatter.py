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


class MLAFormatter:
    def format(self, metadata: "NormalizedMetadata") -> str:
        authors_part = _mla_authors(metadata.authors)
        title = metadata.title if metadata.title else "Untitled"
        parts = [f'{authors_part}"{title}."' if authors_part else f'"{title}."']
        if metadata.venue:
            parts.append(f"{metadata.venue},")
        if metadata.year is not None:
            parts.append(f"{metadata.year}.")
        if metadata.doi:
            parts.append(f"https://doi.org/{metadata.doi}.")
        return " ".join(parts)


class ChicagoFormatter:
    def format(self, metadata: "NormalizedMetadata") -> str:
        authors_part = _chicago_authors(metadata.authors)
        year_part = str(metadata.year) if metadata.year is not None else "n.d."
        title = metadata.title if metadata.title else "Untitled"
        if authors_part:
            lead = f'{authors_part} {year_part}. "{title}."'
        else:
            lead = f'{year_part}. "{title}."'
        parts = [lead]
        if metadata.venue:
            parts.append(f"{metadata.venue}.")
        if metadata.doi:
            parts.append(f"https://doi.org/{metadata.doi}.")
        return " ".join(parts)


class IEEEFormatter:
    def format(self, metadata: "NormalizedMetadata") -> str:
        authors_part = _ieee_authors(metadata.authors)
        title = metadata.title if metadata.title else "Untitled"
        parts = [f'{authors_part}"{title},"' if authors_part else f'"{title},"']
        if metadata.venue:
            parts.append(f"{metadata.venue},")
        if metadata.year is not None:
            parts.append(f"{metadata.year}.")
        else:
            parts.append("n.d.")
        if metadata.doi:
            parts.append(f"doi: {metadata.doi}.")
        return " ".join(parts)


class VancouverFormatter:
    def format(self, metadata: "NormalizedMetadata") -> str:
        authors_part = _vancouver_authors(metadata.authors)
        title = metadata.title if metadata.title else "Untitled"
        parts = [f"{authors_part}{title}." if authors_part else f"{title}."]
        if metadata.venue:
            parts.append(f"{metadata.venue}.")
        if metadata.year is not None:
            parts.append(f"{metadata.year}.")
        if metadata.doi:
            parts.append(f"doi:{metadata.doi}.")
        return " ".join(parts)


def bibtex_key(metadata: "NormalizedMetadata") -> str:
    if metadata.authors:
        _, first_family = _split_author_name(metadata.authors[0])
    else:
        first_family = "anon"
    family_key = "".join(ch for ch in first_family.lower() if ch.isalnum())
    year_key = str(metadata.year) if metadata.year is not None else "nd"
    return f"{family_key}{year_key}"


def unique_bibtex_key(metadata: "NormalizedMetadata", used_keys: set[str]) -> str:
    # Disambiguates keys that would otherwise collide (e.g. two papers by the same
    # first author in the same year) when formatting several references into one file.
    base = bibtex_key(metadata)
    if base not in used_keys:
        return base
    suffix = "a"
    while f"{base}{suffix}" in used_keys:
        suffix = chr(ord(suffix) + 1)
    return f"{base}{suffix}"


class BibTeXFormatter:
    def format(self, metadata: "NormalizedMetadata", key: str | None = None) -> str:
        key = key or bibtex_key(metadata)

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
    "mla": MLAFormatter(),
    "chicago": ChicagoFormatter(),
    "ieee": IEEEFormatter(),
    "vancouver": VancouverFormatter(),
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


def _mla_authors(authors: list[str]) -> str:
    # MLA lists at most two authors by name; three or more collapse to "et al."
    if not authors:
        return ""
    if len(authors) == 1:
        given, family = _split_author_name(authors[0])
        return f"{family}, {given}. " if given else f"{family}. "
    if len(authors) == 2:
        given0, family0 = _split_author_name(authors[0])
        first = f"{family0}, {given0}" if given0 else family0
        return f"{first}, and {authors[1]}. "
    given0, family0 = _split_author_name(authors[0])
    first = f"{family0}, {given0}" if given0 else family0
    return f"{first}, et al. "


def _chicago_authors(authors: list[str]) -> str:
    if not authors:
        return ""
    if len(authors) == 1:
        given, family = _split_author_name(authors[0])
        return f"{family}, {given}." if given else f"{family}."
    if len(authors) <= 3:
        given0, family0 = _split_author_name(authors[0])
        first = f"{family0}, {given0}" if given0 else family0
        if len(authors) == 2:
            rest = "and " + authors[1]
        else:
            rest = ", ".join(authors[1:-1]) + ", and " + authors[-1]
        return f"{first}, {rest}."
    given0, family0 = _split_author_name(authors[0])
    first = f"{family0}, {given0}" if given0 else family0
    return f"{first}, et al."


def _ieee_authors(authors: list[str]) -> str:
    if not authors:
        return ""
    formatted = []
    for author in authors[:3]:
        given, family = _split_author_name(author)
        formatted.append(f"{given[0].upper()}. {family}" if given else family)
    if len(authors) > 3:
        result = ", ".join(formatted) + ", et al."
    elif len(formatted) > 1:
        result = ", ".join(formatted[:-1]) + ", and " + formatted[-1]
    else:
        result = formatted[0]
    return f"{result}, "


def _vancouver_authors(authors: list[str]) -> str:
    if not authors:
        return ""
    formatted = []
    for author in authors[:6]:
        given, family = _split_author_name(author)
        initials = "".join(part[0].upper() for part in given.split()) if given else ""
        formatted.append(f"{family} {initials}".strip())
    result = ", ".join(formatted)
    if len(authors) > 6:
        result += ", et al"
    return f"{result}. "
