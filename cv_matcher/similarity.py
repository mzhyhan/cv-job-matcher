"""Text similarity backends: pure-Python TF-IDF and optional embeddings.

The TF-IDF backend needs no dependencies and no model download, so it is the
default fallback (and what the test suite uses). The embeddings backend uses
the optional ``sentence-transformers`` package (``pip install
"cv-job-matcher[embeddings]"``).
"""

from __future__ import annotations

import importlib
import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

from cv_matcher.models import EmbeddingsUnavailableError

__all__ = [
    "DEFAULT_MODEL_NAME",
    "Embedder",
    "SentenceTransformerEmbedder",
    "cosine_similarity",
    "embedding_similarity",
    "load_local_embedder",
    "tfidf_similarity",
]

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
"""Default sentence-transformers model, small enough for CPU use."""

_RE_WORD = re.compile(r"[a-z0-9][a-z0-9+#]*")
_STOPWORDS = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "will",
        "with",
        "you",
        "your",
        "our",
        "their",
        "they",
        "them",
        "he",
        "she",
        "not",
        "but",
        "if",
        "into",
        "over",
        "than",
        "then",
        "so",
        "such",
        "these",
        "those",
        "who",
        "which",
        "what",
        "when",
        "where",
        "can",
        "may",
        "should",
        "would",
        "could",
        "also",
    )
)


class Embedder(Protocol):
    """Anything that can turn texts into fixed-length vectors."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding vector per input text."""
        ...


def _tokens(text: str) -> list[str]:
    return [word for word in _RE_WORD.findall(text.lower()) if word not in _STOPWORDS]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors, clipped to ``[0, 1]``.

    Args:
        left: First vector.
        right: Second vector of the same length.

    Returns:
        The similarity, or ``0.0`` if either vector has zero length or norm.

    Raises:
        ValueError: If the vectors differ in length.
    """
    if len(left) != len(right):
        raise ValueError("Vectors must have the same length.")
    norm = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    if norm == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(left, right, strict=True))
    return min(1.0, max(0.0, dot / norm))


def tfidf_similarity(left: str, right: str) -> float:
    """Return the TF-IDF cosine similarity of two texts.

    Terms are lower-cased words without common English stop words. Term
    frequency is sub-linear (``1 + ln(tf)``) and the inverse document
    frequency is smoothed over the two texts.

    Args:
        left: First text.
        right: Second text.

    Returns:
        A score in ``[0, 1]``: ``1.0`` for texts with identical term counts
        and ``0.0`` for texts with no terms in common or when either text has
        no usable terms.
    """
    counts = (Counter(_tokens(left)), Counter(_tokens(right)))
    if not counts[0] or not counts[1]:
        return 0.0
    vocabulary = sorted(set(counts[0]) | set(counts[1]))
    vectors: list[list[float]] = [[], []]
    for term in vocabulary:
        document_frequency = sum(1 for count in counts if term in count)
        idf = math.log((1 + 2) / (1 + document_frequency)) + 1.0
        for vector, count in zip(vectors, counts, strict=True):
            vector.append((1.0 + math.log(count[term])) * idf if term in count else 0.0)
    return cosine_similarity(vectors[0], vectors[1])


def embedding_similarity(left: str, right: str, embedder: Embedder) -> float:
    """Return the cosine similarity of two texts under an embedder.

    Args:
        left: First text.
        right: Second text.
        embedder: Object providing :meth:`Embedder.encode`.

    Returns:
        A score in ``[0, 1]``; ``0.0`` if either text is blank.

    Raises:
        ValueError: If the embedder does not return two vectors of equal length.
    """
    if not left.strip() or not right.strip():
        return 0.0
    vectors = embedder.encode([left, right])
    if len(vectors) != 2:
        raise ValueError("Embedder must return one vector per text.")
    return cosine_similarity(vectors[0], vectors[1])


def _import_sentence_transformers() -> Any:
    """Import ``sentence_transformers`` or raise a helpful error."""
    try:
        return importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise EmbeddingsUnavailableError(
            "The 'embeddings' backend needs sentence-transformers. "
            'Install it with: pip install "cv-job-matcher[embeddings]"'
        ) from exc


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model.

    The model is loaded when the object is created.

    Args:
        model_name: Name or path of the model.
        local_only: When ``True`` never touch the network; the model must
            already be in the local Hugging Face cache (or be a local path).

    Raises:
        EmbeddingsUnavailableError: If the library is missing or the model
            cannot be loaded.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, *, local_only: bool = False) -> None:
        """Load the model."""
        module = _import_sentence_transformers()
        try:
            self._model = module.SentenceTransformer(model_name, local_files_only=local_only)
        except Exception as exc:
            raise EmbeddingsUnavailableError(
                f"Could not load sentence-transformers model {model_name!r}: {exc}"
            ) from exc
        self.model_name = model_name

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding vector per text."""
        vectors = self._model.encode(list(texts))
        return [[float(value) for value in vector] for vector in vectors]


def load_local_embedder(model_name: str = DEFAULT_MODEL_NAME) -> Embedder | None:
    """Return an embedder only if no download is needed.

    Args:
        model_name: Name or path of the sentence-transformers model.

    Returns:
        A ready embedder, or ``None`` when the library is not installed or the
        model is not available locally. Never downloads anything.
    """
    try:
        return SentenceTransformerEmbedder(model_name, local_only=True)
    except EmbeddingsUnavailableError as exc:
        logger.debug("Embeddings not available locally, using TF-IDF: %s", exc)
        return None
