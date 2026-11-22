"""Normalisation of text extracted from CVs and job descriptions.

The goal is a stable, plain-text representation for later skill extraction and
embedding: Unicode is canonicalised, hyphenation across line breaks is repaired,
and decorative Markdown syntax can be stripped while keeping the wording.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalise_text"]

# Byte-order mark and common zero-width characters that survive copy-paste.
_ZERO_WIDTH = dict.fromkeys(map(ord, "\ufeff\u200b\u200c\u200d\u2060"), None)

_RE_HYPHEN_BREAK = re.compile(r"(?<=[A-Za-z])-\n(?=[a-z])")
_RE_BLANK_RUNS = re.compile(r"\n{3,}")
_RE_FENCE = re.compile(r"^```[^\n]*\n(.*?)```", re.DOTALL | re.MULTILINE)
_RE_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_RE_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_RE_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*")
_RE_BOLD_UNDERSCORE = re.compile(r"__(.+?)__")
_RE_ITALIC = re.compile(r"(?<!\w)\*([^*]+)\*(?!\w)")
_RE_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_LIST = re.compile(r"^(\s*)(?:[-*+]|\d+\.)\s+", re.MULTILINE)


def normalise_text(text: str, *, strip_markdown: bool = False) -> str:
    """Return a cleaned copy of extracted document text.

    Args:
        text: Raw text taken from a file or an extraction library.
        strip_markdown: If True, drop common Markdown markers (headings, emphasis,
            links, images, code fences and list bullets) while keeping the wording.

    Returns:
        Text with Unicode NFKC normalisation, Unix line endings, repaired
        hyphenation across line breaks, trailing spaces removed, and runs of
        blank lines collapsed. Leading and trailing whitespace is stripped.
        An all-whitespace input becomes an empty string.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZERO_WIDTH)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _RE_HYPHEN_BREAK.sub("", text)
    if strip_markdown:
        text = _strip_markdown(text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _RE_BLANK_RUNS.sub("\n\n", text)
    return text.strip()


def _strip_markdown(text: str) -> str:
    """Remove common Markdown syntax, keeping the underlying wording."""
    text = _RE_FENCE.sub(lambda match: match.group(1).rstrip() + "\n", text)
    text = _RE_IMAGE.sub(r"\1", text)
    text = _RE_LINK.sub(r"\1", text)
    text = _RE_HEADING.sub("", text)
    text = _RE_INLINE_CODE.sub(r"\1", text)
    text = _RE_BOLD.sub(r"\1", text)
    text = _RE_BOLD_UNDERSCORE.sub(r"\1", text)
    text = _RE_ITALIC.sub(r"\1", text)
    return _RE_LIST.sub(r"\1", text)
