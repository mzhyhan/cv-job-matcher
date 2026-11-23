"""Data models and errors for loaded CVs and job descriptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = [
    "DocumentKind",
    "EmptyDocumentError",
    "LoadError",
    "LoadedDocument",
    "Skill",
    "SkillHit",
    "SourceFormat",
    "UnsupportedFormatError",
]


class DocumentKind(str, Enum):
    """What role a document plays when matching a CV to a vacancy.

    Attributes:
        CV: A curriculum vitae or resume.
        JOB_DESCRIPTION: A job advertisement or role specification.
    """

    CV = "cv"
    JOB_DESCRIPTION = "job_description"


class SourceFormat(str, Enum):
    """File format that a document was loaded from.

    Attributes:
        PDF: Portable Document Format (``.pdf``).
        DOCX: Office Open XML Word document (``.docx``).
        TXT: Plain text (``.txt``).
        MD: Markdown (``.md`` or ``.markdown``).
    """

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class LoadError(Exception):
    """Raised when a document cannot be read or parsed."""


class UnsupportedFormatError(LoadError):
    """Raised when the file type is not one of PDF, DOCX, TXT or Markdown."""


class EmptyDocumentError(LoadError):
    """Raised when a document contains no extractable text."""


@dataclass(frozen=True)
class LoadedDocument:
    """Text extracted from a CV or job description.

    Attributes:
        text: Normalised (or raw, if requested) document text.
        kind: Whether this document is a CV or a job description.
        source_format: Format the text was extracted from.
        source_name: Original file name, or a caller-supplied label for bytes.
        source_path: Resolved path when the document was loaded from disk.
        page_count: Number of pages in a PDF; ``None`` for non-paginated formats.
        character_count: Length of :attr:`text` in characters.
        metadata: Optional extra fields reserved for format-specific details.
    """

    text: str
    kind: DocumentKind
    source_format: SourceFormat
    source_name: str
    source_path: Path | None = None
    page_count: int | None = None
    character_count: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fill ``character_count`` when the caller does not set it."""
        if not self.character_count:
            object.__setattr__(self, "character_count", len(self.text))


@dataclass(frozen=True)
class Skill:
    """A skill from the curated taxonomy.

    Attributes:
        name: Canonical skill name, for example ``"PostgreSQL"``.
        category: Broad group such as ``"language"``, ``"ml"`` or ``"cloud"``.
        aliases: Alternative spellings that map to the canonical name.
        ambiguous: ``True`` for names that are also common words or letters
            (``"R"``, ``"Go"``); these are matched case-sensitively and with
            stricter boundaries.
    """

    name: str
    category: str
    aliases: tuple[str, ...] = ()
    ambiguous: bool = False


@dataclass(frozen=True)
class SkillHit:
    """A skill found in a piece of text.

    Attributes:
        name: Canonical skill name.
        category: Taxonomy category of the skill.
        confidence: Match confidence in ``(0, 1]``; ``1.0`` for an exact
            canonical name, slightly lower for aliases, ambiguous names and
            fuzzy matches.
        matched_text: The surface form as written in the text.
        alias: The taxonomy name or alias that matched.
        method: How the hit was found: ``"exact"``, ``"alias"``,
            ``"normalised"`` (punctuation or spacing variant) or ``"fuzzy"``.
    """

    name: str
    category: str
    confidence: float
    matched_text: str
    alias: str
    method: str
