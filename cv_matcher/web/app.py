"""FastAPI application exposing CV/job matching over HTTP plus a small upload page.

Run locally with::

    uvicorn cv_matcher.web.app:app --reload

then open http://localhost:8000 for the web UI or http://localhost:8000/docs for the
interactive OpenAPI documentation.

Endpoints:
    - ``GET /`` - upload page for a CV and a job description.
    - ``POST /match`` - score a CV against a job description and suggest improvements.
    - ``GET /health`` - liveness probe.
"""

from __future__ import annotations

import logging
from importlib import resources
from typing import Annotated, Any, cast, get_args

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from cv_matcher import __version__
from cv_matcher.analysis import Provider, analyse_gap
from cv_matcher.loaders import load_from_bytes
from cv_matcher.models import (
    AnalysisError,
    DocumentKind,
    EmbeddingsUnavailableError,
    EmptyDocumentError,
    LoadedDocument,
    LoadError,
    SkillHit,
    SkillMatch,
    Suggestion,
    UnsupportedFormatError,
)
from cv_matcher.scoring import Backend, match_cv

__all__ = ["MAX_UPLOAD_BYTES", "app"]

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
"""Largest accepted upload per file (10 MB)."""

app = FastAPI(
    title="CV Job Matcher",
    description="Match a CV against a job description: score the fit and suggest improvements.",
    version=__version__,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    """Serve the single-page upload UI.

    Returns:
        The HTML of ``static/index.html``.
    """
    return resources.files("cv_matcher.web").joinpath("static/index.html").read_text("utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the service is running.

    Returns:
        The service status and package version.
    """
    return {"status": "ok", "version": __version__}


def _skill_match(item: SkillMatch) -> dict[str, Any]:
    return {
        "name": item.name,
        "category": item.category,
        "confidence": item.confidence,
        "cv_text": item.cv_text,
        "job_text": item.job_text,
    }


def _skill_hit(item: SkillHit) -> dict[str, Any]:
    return {
        "name": item.name,
        "category": item.category,
        "confidence": item.confidence,
        "matched_text": item.matched_text,
    }


def _suggestion(item: Suggestion) -> dict[str, Any]:
    return {
        "title": item.title,
        "detail": item.detail,
        "skill": item.skill,
        "category": item.category,
    }


async def _load(upload: UploadFile, kind: DocumentKind) -> LoadedDocument:
    """Read and parse one upload, mapping failures to HTTP errors.

    Args:
        upload: The uploaded file.
        kind: Whether the file is a CV or a job description.

    Returns:
        The loaded document.

    Raises:
        HTTPException: 413 if too large, 415 if the format is unsupported, and
            422 if the document is empty or cannot be read.
    """
    name = upload.filename or ""
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"{kind.value} file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
        )
    try:
        return load_from_bytes(data, kind, source_name=name or "upload")
    except UnsupportedFormatError as exc:
        raise HTTPException(415, str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(422, str(exc)) from exc
    except LoadError as exc:
        logger.warning("Could not load %s: %s", name, exc)
        raise HTTPException(422, str(exc)) from exc


@app.post("/match")
async def match(
    cv: Annotated[UploadFile, File(description="The CV (PDF, DOCX, TXT or Markdown)")],
    job: Annotated[
        UploadFile, File(description="The job description (PDF, DOCX, TXT or Markdown)")
    ],
    backend: Annotated[
        str, Form(description="Similarity backend: auto, tfidf or embeddings")
    ] = "auto",
    provider: Annotated[
        str, Form(description="Suggestion provider: auto, rules or openai")
    ] = "auto",
) -> dict[str, Any]:
    """Score a CV against a job description and suggest improvements.

    Args:
        cv: The uploaded CV.
        job: The uploaded job description.
        backend: Similarity backend.
        provider: Gap-analysis provider; ``auto`` stays offline without an API key.

    Returns:
        Scores in ``[0, 1]``, skill lists, a summary and suggestions.

    Raises:
        HTTPException: On oversized, unsupported or unreadable input, or invalid options.
    """
    if backend not in get_args(Backend):
        raise HTTPException(
            422, f"Unknown backend {backend!r}; expected one of {get_args(Backend)}"
        )
    if provider not in get_args(Provider):
        raise HTTPException(
            422, f"Unknown provider {provider!r}; expected one of {get_args(Provider)}"
        )
    cv_doc = await _load(cv, DocumentKind.CV)
    job_doc = await _load(job, DocumentKind.JOB_DESCRIPTION)
    try:
        result = match_cv(cv_doc, job_doc, backend=cast(Backend, backend))
        analysis = analyse_gap(cv_doc, job_doc, match=result, provider=cast(Provider, provider))
    except (EmbeddingsUnavailableError, AnalysisError) as exc:
        raise HTTPException(422, str(exc)) from exc
    logger.info("Matched %s against %s: %.2f", cv.filename, job.filename, result.overall_score)
    return {
        "overall_score": result.overall_score,
        "skill_score": result.skill_score,
        "text_score": result.text_score,
        "backend": result.backend,
        "provider": analysis.provider,
        "summary": analysis.summary,
        "matched_skills": [_skill_match(s) for s in result.matched_skills],
        "missing_skills": [_skill_hit(s) for s in result.missing_skills],
        "extra_skills": [_skill_hit(s) for s in result.extra_skills],
        "suggestions": [_suggestion(s) for s in analysis.suggestions],
        "cv_filename": cv.filename,
        "job_filename": job.filename,
    }
