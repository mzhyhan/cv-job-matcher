"""Tests for the FastAPI web API (offline; no live server)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cv_matcher import __version__
from cv_matcher.web.app import app

from .conftest import SAMPLE_JOB_TEXT

client = TestClient(app)
OFFLINE = {"backend": "tfidf", "provider": "rules"}


def _post(
    cv: tuple[str, bytes], job: tuple[str, bytes], data: dict[str, str] | None = None
) -> httpx.Response:
    return client.post(
        "/match",
        files={"cv": cv, "job": job},
        data=OFFLINE if data is None else data,
    )


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_match_text_and_markdown(txt_cv_path: Path, md_job_path: Path) -> None:
    response = client.post(
        "/match",
        files={
            "cv": (txt_cv_path.name, txt_cv_path.read_bytes()),
            "job": (md_job_path.name, md_job_path.read_bytes()),
        },
        data=OFFLINE,
    )
    assert response.status_code == 200
    body = response.json()
    for key in ("overall_score", "skill_score", "text_score"):
        assert 0.0 <= body[key] <= 1.0
    assert body["backend"] == "tfidf"
    assert body["provider"] == "rules"
    assert body["summary"]
    assert "Python" in {s["name"] for s in body["matched_skills"]}
    assert isinstance(body["missing_skills"], list)
    assert isinstance(body["extra_skills"], list)
    assert body["suggestions"]
    assert set(body["suggestions"][0]) == {"title", "detail", "skill", "category"}
    assert body["cv_filename"] == "cv.txt"
    assert body["job_filename"] == "job.md"


def test_match_pdf_and_docx(pdf_cv_path: Path, docx_cv_path: Path) -> None:
    job = ("job.txt", SAMPLE_JOB_TEXT.encode())
    for path in (pdf_cv_path, docx_cv_path):
        response = _post((path.name, path.read_bytes()), job)
        assert response.status_code == 200
        assert response.json()["cv_filename"] == path.name


def test_auto_defaults_stay_offline(
    txt_cv_path: Path, md_job_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post(
        "/match",
        files={
            "cv": (txt_cv_path.name, txt_cv_path.read_bytes()),
            "job": (md_job_path.name, md_job_path.read_bytes()),
        },
    )
    assert response.status_code == 200
    assert response.json()["provider"] == "rules"


def test_rejects_unsupported_type() -> None:
    response = _post(("cv.exe", b"MZ"), ("job.txt", b"Python"))
    assert response.status_code == 415


def test_rejects_missing_suffix() -> None:
    response = _post(("cv", b"Python"), ("job.txt", b"Python"))
    assert response.status_code == 415


def test_rejects_oversized_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cv_matcher.web.app.MAX_UPLOAD_BYTES", 10)
    response = _post(("cv.txt", b"x" * 100), ("job.txt", b"Python"))
    assert response.status_code == 413


def test_rejects_empty_document() -> None:
    response = _post(("cv.txt", b"   \n"), ("job.txt", b"Python"))
    assert response.status_code == 422


def test_rejects_corrupt_pdf() -> None:
    response = _post(("cv.pdf", b"not a pdf"), ("job.txt", b"Python"))
    assert response.status_code == 422


def test_rejects_invalid_options() -> None:
    cv, job = ("cv.txt", b"Python"), ("job.txt", b"Python")
    assert _post(cv, job, {"backend": "nope"}).status_code == 422
    assert _post(cv, job, {"provider": "nope"}).status_code == 422


def test_embeddings_unavailable_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    from cv_matcher.models import EmbeddingsUnavailableError

    def _boom(*_a: object, **_k: object) -> None:
        raise EmbeddingsUnavailableError("not installed")

    monkeypatch.setattr("cv_matcher.web.app.match_cv", _boom)
    response = _post(("cv.txt", b"Python"), ("job.txt", b"Python"))
    assert response.status_code == 422


def test_analysis_error_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    from cv_matcher.models import AnalysisError

    def _boom(*_a: object, **_k: object) -> None:
        raise AnalysisError("provider failed")

    monkeypatch.setattr("cv_matcher.web.app.analyse_gap", _boom)
    response = _post(("cv.txt", b"Python"), ("job.txt", b"Python"))
    assert response.status_code == 422


def test_missing_file_is_422() -> None:
    response = client.post("/match", files={"cv": ("cv.txt", b"Python")}, data=OFFLINE)
    assert response.status_code == 422
