import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scholarmind.citations.formatter import BibTeXFormatter, unique_bibtex_key
from scholarmind.citations.metadata import NormalizedMetadata
from scholarmind.citations.verify import CITATION_MARKER_PATTERN

if TYPE_CHECKING:
    from scholarmind.citations.verify import Citation

_TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


@dataclass
class LatexBundle:
    tex: str
    bib: str


def _escape_tex(text: str) -> str:
    # Character-by-character against the ORIGINAL text (not a mutating accumulator), so a
    # backslash introduced by escaping one character is never itself re-escaped.
    return "".join(_TEX_ESCAPES.get(ch, ch) for ch in text)


def _replace_markers_with_cite(text: str, keys_by_index: dict[int, str]) -> str:
    def _replace(match: re.Match) -> str:
        indices = [int(part.strip()) for part in match.group(1).split(",")]
        keys = [keys_by_index[i] for i in indices if i in keys_by_index]
        if not keys:
            return match.group(0)  # leave unresolvable markers as-is rather than dropping them
        return "\\cite{" + ",".join(keys) + "}"

    return CITATION_MARKER_PATTERN.sub(_replace, text)


def _wrap_tex_document(body: str) -> str:
    return (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{cite}\n"
        "\\title{ScholarMind Draft}\n"
        "\\begin{document}\n"
        "\\maketitle\n\n"
        f"{body}\n\n"
        "\\bibliographystyle{plain}\n"
        "\\bibliography{references}\n"
        "\\end{document}\n"
    )


def build_latex_bundle(text: str, citations: list["Citation"]) -> LatexBundle:
    # [N] markers become \cite{key} referencing this same draft's own citation list, and the
    # matching .bib entries are built with the SAME keys via unique_bibtex_key, the same
    # collision-safe key generator export_bibtex() uses for whole-library exports.
    formatter = BibTeXFormatter()
    used_keys: set[str] = set()
    keys_by_index: dict[int, str] = {}
    entries = []

    for citation in citations:
        metadata = NormalizedMetadata(
            doi=None,
            title=citation.title,
            authors=citation.authors,
            year=citation.year,
            venue=None,
            source="draft",
        )
        key = unique_bibtex_key(metadata, used_keys)
        used_keys.add(key)
        keys_by_index[citation.index] = key
        entries.append(formatter.format(metadata, key=key))

    # Escape the draft body first, then substitute [N] markers — none of the characters in a
    # marker ([, ], digits, comma) are TeX-special, so escaping first and substituting after
    # never mangles the \cite{} commands this step inserts.
    escaped_body = _escape_tex(text)
    body_with_cites = _replace_markers_with_cite(escaped_body, keys_by_index)

    return LatexBundle(tex=_wrap_tex_document(body_with_cites), bib="\n\n".join(entries))
