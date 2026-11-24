"""cv-job-matcher: match a CV against a job description.

Public API (document loading):
    - :func:`load_document` and :func:`load_from_bytes` - extract text from a
      CV or job description in PDF, DOCX, TXT or Markdown.
    - :func:`normalise_text` - clean extracted text for later matching.
    - :class:`LoadedDocument`, :class:`DocumentKind`, :class:`SourceFormat`.
    - :class:`LoadError`, :class:`UnsupportedFormatError`,
      :class:`EmptyDocumentError`.

Public API (skills):
    - :func:`extract_skills` - find taxonomy skills in text, with fuzzy matching.
    - :func:`load_taxonomy`, :class:`Skill`, :class:`SkillHit`.

Public API (scoring):
    - :func:`match_cv` - score a CV against a job description, returning a
      :class:`MatchResult` with skill overlap, text similarity and the overall
      score. TF-IDF is the default fallback; embeddings need the optional
      ``embeddings`` extra.
    - :func:`tfidf_similarity`, :class:`SentenceTransformerEmbedder`,
      :class:`ScoreWeights`, :class:`SkillMatch`.

Public API (gap analysis):
    - :func:`analyse_gap` - produce a summary and improvement suggestions as an
      :class:`AnalysisResult`. An offline :class:`RuleBasedAnalyser` is the
      fallback; :class:`OpenAICompatibleAnalyser` uses any OpenAI-compatible
      endpoint when an API key is set.
    - :class:`GapAnalyser`, :data:`Provider`, :class:`Suggestion`,
      :class:`AnalysisError`.

Example:
    >>> import cv_matcher
    >>> cv_matcher.__version__
    '0.1.0'
    >>> from cv_matcher import DocumentKind, load_document  # doctest: +SKIP
    >>> cv = load_document("cv.pdf", kind=DocumentKind.CV)  # doctest: +SKIP
"""

import logging

from cv_matcher.analysis import (
    GapAnalyser,
    OpenAICompatibleAnalyser,
    Provider,
    RuleBasedAnalyser,
    analyse_gap,
)
from cv_matcher.loaders import (
    SUPPORTED_EXTENSIONS,
    detect_format,
    load_document,
    load_from_bytes,
)
from cv_matcher.models import (
    AnalysisError,
    AnalysisResult,
    DocumentKind,
    EmbeddingsUnavailableError,
    EmptyDocumentError,
    LoadedDocument,
    LoadError,
    MatchResult,
    ScoreWeights,
    Skill,
    SkillHit,
    SkillMatch,
    SourceFormat,
    Suggestion,
    UnsupportedFormatError,
)
from cv_matcher.scoring import DEFAULT_WEIGHTS, Backend, match_cv
from cv_matcher.similarity import (
    DEFAULT_MODEL_NAME,
    Embedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
    tfidf_similarity,
)
from cv_matcher.skills import DEFAULT_FUZZY_THRESHOLD, extract_skills, load_taxonomy
from cv_matcher.text import normalise_text

__all__ = [
    "DEFAULT_FUZZY_THRESHOLD",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_WEIGHTS",
    "SUPPORTED_EXTENSIONS",
    "AnalysisError",
    "AnalysisResult",
    "Backend",
    "DocumentKind",
    "Embedder",
    "EmbeddingsUnavailableError",
    "EmptyDocumentError",
    "GapAnalyser",
    "LoadError",
    "LoadedDocument",
    "MatchResult",
    "OpenAICompatibleAnalyser",
    "Provider",
    "RuleBasedAnalyser",
    "ScoreWeights",
    "SentenceTransformerEmbedder",
    "Skill",
    "SkillHit",
    "SkillMatch",
    "SourceFormat",
    "Suggestion",
    "UnsupportedFormatError",
    "__version__",
    "analyse_gap",
    "cosine_similarity",
    "detect_format",
    "extract_skills",
    "load_document",
    "load_from_bytes",
    "load_taxonomy",
    "match_cv",
    "normalise_text",
    "tfidf_similarity",
]
__version__ = "0.1.0"

# Library best practice: let the application decide how (and whether) to log.
logging.getLogger(__name__).addHandler(logging.NullHandler())
