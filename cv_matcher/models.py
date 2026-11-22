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
