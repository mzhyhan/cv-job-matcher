from __future__ import annotations

import socket
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from cv_matcher import (
    DEFAULT_WEIGHTS,
    DocumentKind,
    EmbeddingsUnavailableError,
    LoadedDocument,
    MatchResult,
    ScoreWeights,
    SourceFormat,
    match_cv,
)
from tests.conftest import SAMPLE_CV_TEXT, SAMPLE_JOB_TEXT

STRONG_CV = (
    "Senior Python engineer. Built NLP and machine learning services with PyTorch, "
    "Hugging Face and FastAPI. Deployed on AWS using Docker and Kubernetes with "
    "CI/CD pipelines. Strong SQL and PostgreSQL. Git, pytest and Redis."
)
JOB = (
    "We are hiring a Python engineer to build machine learning and NLP services. "
    "Requirements: Python, PyTorch, FastAPI, Docker, Kubernetes, AWS, PostgreSQL."
)
WEAK_CV = (
    "Experienced retail manager. Led teams, managed stock, handled customer "
    "complaints and trained new staff. Proficient in Excel and Java."
)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    yield


class BagEmbedder:
    """Deterministic stub: hashed bag-of-words vectors, no model needed."""

    def __init__(self) -> None:
        self.calls: list[Sequence[str]] = []

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls.append(texts)
        vectors = []
        for text in texts:
            vector = [0.0] * 16
            for word in text.lower().split():
                vector[sum(map(ord, word)) % 16] += 1.0
            vectors.append(vector)
        return vectors


class TestScoreWeights:
    def test_defaults(self) -> None:
        assert (DEFAULT_WEIGHTS.skill, DEFAULT_WEIGHTS.text) == (0.6, 0.4)

    def test_normalised(self) -> None:
        weights = ScoreWeights(skill=3.0, text=1.0).normalised()
        assert (weights.skill, weights.text) == (0.75, 0.25)

    @pytest.mark.parametrize(("skill", "text"), [(-0.1, 1.0), (1.0, -1.0), (0.0, 0.0)])
    def test_invalid(self, skill: float, text: float) -> None:
        with pytest.raises(ValueError, match="eight"):
            ScoreWeights(skill=skill, text=text)


class TestMatchCv:
    def test_strong_match_beats_weak_match(self) -> None:
        strong = match_cv(STRONG_CV, JOB, backend="tfidf")
        weak = match_cv(WEAK_CV, JOB, backend="tfidf")
        assert strong.overall_score > 0.7
        assert weak.overall_score < 0.2
        assert strong.skill_score > weak.skill_score
        assert strong.text_score > weak.text_score

    def test_result_shape(self) -> None:
        result = match_cv(STRONG_CV, JOB, backend="tfidf")
        assert isinstance(result, MatchResult)
        assert result.backend == "tfidf"
        for score in (result.overall_score, result.skill_score, result.text_score):
            assert 0.0 <= score <= 1.0
        assert result.weights == ScoreWeights(0.6, 0.4).normalised()

    def test_matched_missing_and_extra_skills(self) -> None:
        job = "Needs Python, Docker and Terraform."
        cv = "I use python, docker and Redis."
        result = match_cv(cv, job, backend="tfidf")
        assert [m.name for m in result.matched_skills] == ["Python", "Docker"]
        assert [h.name for h in result.missing_skills] == ["Terraform"]
        assert [h.name for h in result.extra_skills] == ["Redis"]
        assert result.skill_score == pytest.approx(2 / 3)
        first = result.matched_skills[0]
        assert (first.cv_text, first.job_text, first.category) == ("python", "Python", "language")
        assert first.confidence == 1.0

    def test_matching_uses_canonical_names_across_aliases(self) -> None:
        result = match_cv("Postgres and k8s", "PostgreSQL and Kubernetes", backend="tfidf")
        assert result.skill_score == 1.0
        assert not result.missing_skills
        assert result.matched_skills[0].confidence < 1.0

    def test_default_weights_combination(self) -> None:
        result = match_cv("Python only", "Python and Docker", backend="tfidf")
        expected = 0.6 * result.skill_score + 0.4 * result.text_score
        assert result.skill_score == 0.5
        assert result.overall_score == pytest.approx(expected)

    def test_custom_weights_are_normalised(self) -> None:
        base = match_cv("Python only", "Python and Docker", backend="tfidf")
        text_only = match_cv(
            "Python only",
            "Python and Docker",
            backend="tfidf",
            weights=ScoreWeights(skill=0.0, text=5.0),
        )
        skill_only = match_cv(
            "Python only",
            "Python and Docker",
            backend="tfidf",
            weights=ScoreWeights(skill=2.0, text=0.0),
        )
        assert text_only.overall_score == pytest.approx(base.text_score)
        assert skill_only.overall_score == pytest.approx(base.skill_score)
        assert text_only.weights == ScoreWeights(0.0, 1.0)

    def test_job_without_skills_uses_text_similarity(self) -> None:
        job = "Friendly shop assistant needed for weekend shifts."
        cv = "Friendly shop assistant with weekend availability, plus Python."
        result = match_cv(cv, job, backend="tfidf")
        assert result.skill_score == 1.0
        assert result.weights == ScoreWeights(0.0, 1.0)
        assert result.overall_score == pytest.approx(result.text_score)
        assert result.text_score > 0
        assert [h.name for h in result.extra_skills] == ["Python"]

    def test_identical_texts_score_close_to_one(self) -> None:
        result = match_cv(JOB, JOB, backend="tfidf")
        assert result.text_score == pytest.approx(1.0)
        assert result.overall_score == pytest.approx(1.0)
        assert not result.missing_skills
        assert not result.extra_skills

    def test_accepts_loaded_documents(self) -> None:
        cv = LoadedDocument(SAMPLE_CV_TEXT, DocumentKind.CV, SourceFormat.TXT, "cv.txt")
        job = LoadedDocument(
            SAMPLE_JOB_TEXT, DocumentKind.JOB_DESCRIPTION, SourceFormat.TXT, "job.txt"
        )
        from_documents = match_cv(cv, job, backend="tfidf")
        from_text = match_cv(SAMPLE_CV_TEXT, SAMPLE_JOB_TEXT, backend="tfidf")
        assert from_documents == from_text
        assert [m.name for m in from_documents.matched_skills] == ["Python"]
        assert [h.name for h in from_documents.missing_skills] == ["Natural Language Processing"]

    def test_accepts_files_from_loader(self, txt_cv_path: Path, md_job_path: Path) -> None:
        from cv_matcher import load_document

        cv = load_document(txt_cv_path, kind=DocumentKind.CV)
        job = load_document(md_job_path, kind=DocumentKind.JOB_DESCRIPTION)
        result = match_cv(cv, job, backend="tfidf")
        assert {m.name for m in result.matched_skills} == {"Python"}
        assert {h.name for h in result.missing_skills} == {
            "Natural Language Processing",
            "FastAPI",
            "Embeddings",
        }

    def test_result_is_immutable(self) -> None:
        result = match_cv("Python", "Python", backend="tfidf")
        with pytest.raises(AttributeError):
            result.overall_score = 0.0  # type: ignore[misc]


class TestEmptyInputs:
    def test_empty_cv(self) -> None:
        result = match_cv("", JOB, backend="tfidf")
        assert (result.overall_score, result.skill_score, result.text_score) == (0.0, 0.0, 0.0)
        assert not result.matched_skills
        assert result.missing_skills

    def test_empty_job(self) -> None:
        result = match_cv(STRONG_CV, "  ", backend="tfidf")
        assert result.text_score == 0.0
        assert result.overall_score == 0.0
        assert result.skill_score == 1.0
        assert result.extra_skills

    def test_both_empty(self) -> None:
        result = match_cv("", "", backend="tfidf")
        assert result.overall_score == 0.0
        assert not result.matched_skills
        assert not result.missing_skills
        assert not result.extra_skills

    def test_empty_inputs_with_embeddings_backend(self) -> None:
        result = match_cv("", "", backend="embeddings", embedder=BagEmbedder())
        assert result.overall_score == 0.0
        assert result.backend == "embeddings"


class TestBackends:
    def test_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            match_cv("a", "b", backend="magic")  # type: ignore[arg-type]

    def test_tfidf_ignores_embedder(self) -> None:
        embedder = BagEmbedder()
        result = match_cv(STRONG_CV, JOB, backend="tfidf", embedder=embedder)
        assert result.backend == "tfidf"
        assert embedder.calls == []

    def test_embeddings_backend_with_injected_embedder(self) -> None:
        embedder = BagEmbedder()
        result = match_cv(STRONG_CV, JOB, backend="embeddings", embedder=embedder)
        assert result.backend == "embeddings"
        assert embedder.calls == [[STRONG_CV, JOB]]
        assert 0.0 < result.text_score <= 1.0
        # Skill scoring does not depend on the text backend.
        assert result.skill_score == match_cv(STRONG_CV, JOB, backend="tfidf").skill_score

    def test_embeddings_text_score_feeds_overall(self) -> None:
        result = match_cv(
            "Python", "Python and Docker", backend="embeddings", embedder=BagEmbedder()
        )
        assert result.overall_score == pytest.approx(0.6 * 0.5 + 0.4 * result.text_score)

    def test_embeddings_backend_without_library_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(name: str) -> Any:
            raise ImportError(name)

        monkeypatch.setattr("cv_matcher.similarity.importlib.import_module", fail)
        with pytest.raises(EmbeddingsUnavailableError, match="embeddings"):
            match_cv("a", "b", backend="embeddings")

    def test_embeddings_backend_loads_model_by_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[tuple[str, dict[str, Any]]] = []

        class Model:
            def __init__(self, name: str, **kwargs: Any) -> None:
                seen.append((name, kwargs))

            def encode(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

        class Module:
            SentenceTransformer = Model

        monkeypatch.setattr("cv_matcher.similarity.importlib.import_module", lambda n: Module)
        result = match_cv("a b", "a b", backend="embeddings", model_name="custom")
        assert result.backend == "embeddings"
        assert result.text_score == pytest.approx(1.0)
        assert seen == [("custom", {"local_files_only": False})]

    def test_auto_falls_back_to_tfidf_without_library(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(name: str) -> Any:
            raise ImportError(name)

        monkeypatch.setattr("cv_matcher.similarity.importlib.import_module", fail)
        result = match_cv(STRONG_CV, JOB)
        assert result.backend == "tfidf"
        assert result == match_cv(STRONG_CV, JOB, backend="tfidf")

    def test_auto_falls_back_when_model_is_not_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[dict[str, Any]] = []

        class Model:
            def __init__(self, name: str, **kwargs: Any) -> None:
                seen.append(kwargs)
                raise OSError("model not found locally")

        class Module:
            SentenceTransformer = Model

        monkeypatch.setattr("cv_matcher.similarity.importlib.import_module", lambda n: Module)
        assert match_cv(STRONG_CV, JOB, backend="auto").backend == "tfidf"
        assert seen == [{"local_files_only": True}]  # never asks to download

    def test_auto_uses_embeddings_when_model_is_local(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Model:
            def __init__(self, name: str, **kwargs: Any) -> None:
                assert kwargs == {"local_files_only": True}

            def encode(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 1.0] for _ in texts]

        class Module:
            SentenceTransformer = Model

        monkeypatch.setattr("cv_matcher.similarity.importlib.import_module", lambda n: Module)
        result = match_cv(STRONG_CV, JOB)
        assert result.backend == "embeddings"
        assert result.text_score == pytest.approx(1.0)

    def test_auto_uses_injected_embedder(self) -> None:
        result = match_cv(STRONG_CV, JOB, embedder=BagEmbedder())
        assert result.backend == "embeddings"

    def test_auto_is_default_and_offline_here(self) -> None:
        # sentence-transformers is not a dependency of the test environment, and
        # the network is blocked: the default call must still succeed.
        assert match_cv(STRONG_CV, JOB).backend in {"tfidf", "embeddings"}
