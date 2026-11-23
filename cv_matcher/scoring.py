"""Overall CV-to-job match scoring.

The overall score is a weighted combination of two components:

* **Skill score** - the fraction of skills in the job description that also
  appear on the CV.
* **Text score** - cosine similarity of the two documents, using embeddings
  when available and TF-IDF otherwise.
"""

from __future__ import annotations

import logging
from typing import Literal, get_args

from cv_matcher.models import (
    LoadedDocument,
    MatchResult,
    ScoreWeights,
    SkillHit,
    SkillMatch,
)
from cv_matcher.similarity import (
    DEFAULT_MODEL_NAME,
    Embedder,
    SentenceTransformerEmbedder,
    embedding_similarity,
    load_local_embedder,
    tfidf_similarity,
)
from cv_matcher.skills import extract_skills

__all__ = ["DEFAULT_WEIGHTS", "Backend", "match_cv"]

logger = logging.getLogger(__name__)

Backend = Literal["auto", "tfidf", "embeddings"]
"""Names accepted for the text-similarity backend."""

DEFAULT_WEIGHTS = ScoreWeights(skill=0.6, text=0.4)
"""Default weighting: 60% skill overlap, 40% text similarity."""


def _text_of(document: LoadedDocument | str) -> str:
    return document.text if isinstance(document, LoadedDocument) else document


def _resolve_embedder(backend: str, embedder: Embedder | None, model_name: str) -> Embedder | None:
    """Return the embedder to use, or ``None`` for TF-IDF."""
    if backend == "tfidf":
        return None
    if embedder is not None:
        return embedder
    if backend == "embeddings":
        return SentenceTransformerEmbedder(model_name)
    return load_local_embedder(model_name)


def match_cv(
    cv: LoadedDocument | str,
    job: LoadedDocument | str,
    *,
    backend: Backend = "auto",
    weights: ScoreWeights | None = None,
    embedder: Embedder | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> MatchResult:
    """Score how well a CV matches a job description.

    The skill score is the fraction of job skills found on the CV; it is
    ``1.0`` when the job lists no extractable skills, in which case the skill
    weight is dropped and the overall score equals the text score. Blank
    inputs never raise: a blank document has no skills and a text score of
    ``0.0``.

    Args:
        cv: The CV, as a loaded document or raw text.
        job: The job description, as a loaded document or raw text.
        backend: ``"tfidf"`` always uses TF-IDF. ``"embeddings"`` requires
            sentence-transformers (or an injected ``embedder``) and may
            download the model on first use. ``"auto"`` uses embeddings only
            if the library is installed and the model is already available
            locally, otherwise TF-IDF; it never downloads a model.
        weights: Relative weights of the skill and text scores; defaults to
            :data:`DEFAULT_WEIGHTS` (0.6 / 0.4). They are normalised to sum to one.
        embedder: Pre-built embedder to use for the ``"embeddings"`` and
            ``"auto"`` backends, for example a stub in tests.
        model_name: sentence-transformers model for the embeddings backends.

    Returns:
        The score breakdown with matched, missing and extra skills.

    Raises:
        ValueError: If ``backend`` is not a known backend name.
        EmbeddingsUnavailableError: If ``backend="embeddings"`` is requested
            but sentence-transformers is missing or the model cannot be loaded.
    """
    if backend not in get_args(Backend):
        raise ValueError(f"Unknown backend {backend!r}; expected one of {get_args(Backend)}.")
    cv_text, job_text = _text_of(cv), _text_of(job)

    active = _resolve_embedder(backend, embedder, model_name)
    if active is None:
        text_score, used = tfidf_similarity(cv_text, job_text), "tfidf"
    else:
        text_score, used = embedding_similarity(cv_text, job_text, active), "embeddings"

    cv_hits = {hit.name: hit for hit in extract_skills(cv_text)}
    job_hits = {hit.name: hit for hit in extract_skills(job_text)}
    matched = tuple(
        SkillMatch(
            name=name,
            category=job_hit.category,
            cv_text=cv_hits[name].matched_text,
            job_text=job_hit.matched_text,
            confidence=min(job_hit.confidence, cv_hits[name].confidence),
        )
        for name, job_hit in job_hits.items()
        if name in cv_hits
    )
    missing: tuple[SkillHit, ...] = tuple(h for n, h in job_hits.items() if n not in cv_hits)
    extra: tuple[SkillHit, ...] = tuple(h for n, h in cv_hits.items() if n not in job_hits)

    if job_hits:
        skill_score = len(matched) / len(job_hits)
        effective = (weights or DEFAULT_WEIGHTS).normalised()
    else:
        skill_score = 1.0
        effective = ScoreWeights(skill=0.0, text=1.0)
    overall = effective.skill * skill_score + effective.text * text_score
    logger.debug(
        "Match: skill=%.3f text=%.3f overall=%.3f backend=%s",
        skill_score,
        text_score,
        overall,
        used,
    )
    return MatchResult(
        overall_score=min(1.0, max(0.0, overall)),
        skill_score=skill_score,
        text_score=text_score,
        weights=effective,
        backend=used,
        matched_skills=matched,
        missing_skills=missing,
        extra_skills=extra,
    )
