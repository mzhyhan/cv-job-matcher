"""Data models and errors for loaded CVs and job descriptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = [
    "DocumentKind",
    "EmbeddingsUnavailableError",
    "EmptyDocumentError",
    "LoadError",
    "LoadedDocument",
    "MatchResult",
    "ScoreWeights",
    "Skill",
    "SkillHit",
    "SkillMatch",
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


class EmbeddingsUnavailableError(RuntimeError):
    """Raised when embeddings were requested but cannot be used.

    This happens when ``sentence-transformers`` is not installed (install the
    ``embeddings`` extra) or when the model cannot be loaded.
    """


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


@dataclass(frozen=True)
class SkillMatch:
    """A job skill that is also present on the CV.

    Attributes:
        name: Canonical skill name.
        category: Taxonomy category of the skill.
        cv_text: Surface form found in the CV.
        job_text: Surface form found in the job description.
        confidence: The lower of the CV and job match confidences.
    """

    name: str
    category: str
    cv_text: str
    job_text: str
    confidence: float


@dataclass(frozen=True)
class ScoreWeights:
    """Relative weights of the skill and text-similarity scores.

    The weights only need to be non-negative with a positive sum; they are
    normalised so that they add up to one when scores are combined.

    Attributes:
        skill: Weight of the skill-overlap score.
        text: Weight of the text-similarity score.

    Raises:
        ValueError: If a weight is negative or both weights are zero.
    """

    skill: float = 0.6
    text: float = 0.4

    def __post_init__(self) -> None:
        """Validate the weights."""
        if self.skill < 0 or self.text < 0:
            raise ValueError("Weights must not be negative.")
        if self.skill + self.text <= 0:
            raise ValueError("At least one weight must be positive.")

    def normalised(self) -> ScoreWeights:
        """Return weights scaled so that they sum to one."""
        total = self.skill + self.text
        return ScoreWeights(self.skill / total, self.text / total)


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching a CV against a job description.

    Attributes:
        overall_score: Weighted combination of the two scores, in ``[0, 1]``.
        skill_score: Fraction of job skills found on the CV, in ``[0, 1]``.
        text_score: Text similarity of the two documents, in ``[0, 1]``.
        weights: Effective (normalised) weights used for ``overall_score``.
            When the job lists no extractable skills the skill weight is zero.
        backend: Text-similarity backend used: ``"tfidf"`` or ``"embeddings"``.
        matched_skills: Job skills that also appear on the CV.
        missing_skills: Job skills that are absent from the CV.
        extra_skills: CV skills that the job description does not mention.
    """

    overall_score: float
    skill_score: float
    text_score: float
    weights: ScoreWeights
    backend: str
    matched_skills: tuple[SkillMatch, ...] = ()
    missing_skills: tuple[SkillHit, ...] = ()
    extra_skills: tuple[SkillHit, ...] = ()
