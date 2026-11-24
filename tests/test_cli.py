from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from cv_matcher import EmbeddingsUnavailableError, __version__
from cv_matcher.cli import build_parser, main

CV_TEXT = "Python developer. Built APIs with Docker and PostgreSQL."
JOB_TEXT = "We need Python, Docker, Kubernetes and PostgreSQL experience."


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)

    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    cv = tmp_path / "cv.txt"
    job = tmp_path / "job.md"
    cv.write_text(CV_TEXT, encoding="utf-8")
    job.write_text(JOB_TEXT, encoding="utf-8")
    return cv, job


def test_plain_output(files: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    cv, job = files
    assert main([str(cv), str(job), "--backend", "tfidf"]) == 0
    out = capsys.readouterr().out
    assert "Overall score:" in out and "%" in out
    assert "(tfidf)" in out
    assert "Matched skills: " in out and "Docker" in out
    assert "Missing skills: Kubernetes" in out
    assert "Suggestions (provider: rules):" in out
    assert "1. Add evidence of Kubernetes" in out


def test_json_output(files: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    cv, job = files
    assert main([str(cv), str(job), "--json", "--backend", "tfidf"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload) == [
        "overall_score",
        "skill_score",
        "text_score",
        "backend",
        "matched_skills",
        "missing_skills",
        "extra_skills",
        "provider",
        "summary",
        "suggestions",
    ]
    assert payload["backend"] == "tfidf"
    assert payload["provider"] == "rules"
    assert "Kubernetes" in payload["missing_skills"]
    assert "Python" in payload["matched_skills"]
    assert payload["suggestions"][0]["skill"] == "Kubernetes"
    assert 0 <= payload["overall_score"] <= 1


def test_output_file(files: tuple[Path, Path], tmp_path: Path) -> None:
    cv, job = files
    out = tmp_path / "report.txt"
    assert main([str(cv), str(job), "--backend", "tfidf", "-o", str(out), "-v"]) == 0
    assert "Overall score:" in out.read_text(encoding="utf-8")


def test_missing_file(
    files: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cv, _ = files
    assert main([str(cv), str(tmp_path / "nope.md")]) == 2
    assert "error: file not found" in capsys.readouterr().err
    assert main([str(tmp_path / "nope.txt"), str(cv)]) == 2


def test_file_vanishes_during_load(
    files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def gone(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError(2, "No such file", "x.txt")

    monkeypatch.setattr("cv_matcher.cli.load_document", gone)
    assert main([str(files[0]), str(files[1])]) == 2
    assert "file not found: x.txt" in capsys.readouterr().err


def test_load_error(
    tmp_path: Path, files: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "cv.xyz"
    bad.write_text("x", encoding="utf-8")
    assert main([str(bad), str(files[1])]) == 1
    assert capsys.readouterr().err.startswith("error:")


def test_openai_without_key_is_error(
    files: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    cv, job = files
    assert main([str(cv), str(job), "--provider", "openai", "--backend", "tfidf"]) == 1
    assert "API key" in capsys.readouterr().err


def test_embeddings_unavailable_is_error(
    files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise EmbeddingsUnavailableError("sentence-transformers is not installed")

    monkeypatch.setattr("cv_matcher.scoring.SentenceTransformerEmbedder", unavailable)
    assert main([str(files[0]), str(files[1]), "--backend", "embeddings"]) == 1
    assert capsys.readouterr().err.startswith("error:")


def test_rules_provider_ignores_api_key(
    files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cv, job = files
    assert main([str(cv), str(job), "--provider", "rules", "--backend", "tfidf", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["provider"] == "rules"


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"cv-match {__version__}"


def test_missing_args() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_parser_defaults() -> None:
    args = build_parser().parse_args(["a", "b"])
    assert (args.backend, args.provider, args.json) == ("auto", "auto", False)
