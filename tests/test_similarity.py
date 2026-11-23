from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from cv_matcher import (
    EmbeddingsUnavailableError,
    SentenceTransformerEmbedder,
    cosine_similarity,
    tfidf_similarity,
)
from cv_matcher.similarity import embedding_similarity, load_local_embedder


class FakeModel:
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


def install_fake_library(monkeypatch: pytest.MonkeyPatch, model: Any = FakeModel) -> None:
    module = SimpleNamespace(SentenceTransformer=model)
    monkeypatch.setattr("cv_matcher.similarity.importlib.import_module", lambda name: module)


def install_missing_library(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr("cv_matcher.similarity.importlib.import_module", fail)


class TestCosine:
    def test_identical_vectors(self) -> None:
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_negative_similarity_is_clipped(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == 0.0

    def test_zero_vector(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert cosine_similarity([], []) == 0.0

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity([1.0], [1.0, 2.0])


class TestTfidf:
    def test_identical_texts_score_one(self) -> None:
        text = "Python engineer building NLP pipelines with FastAPI"
        assert tfidf_similarity(text, text) == pytest.approx(1.0)

    def test_is_symmetric_and_bounded(self) -> None:
        a, b = "python nlp engineer", "java backend developer with python"
        assert tfidf_similarity(a, b) == pytest.approx(tfidf_similarity(b, a))
        assert 0.0 < tfidf_similarity(a, b) < 1.0

    def test_disjoint_texts_score_zero(self) -> None:
        assert tfidf_similarity("python docker", "painting sculpture") == 0.0

    def test_more_overlap_scores_higher(self) -> None:
        job = "python machine learning engineer with docker and aws"
        close = "python machine learning engineer using docker"
        partial = "python scripting for office tasks"
        unrelated = "retail assistant stocking shelves"
        scores = [tfidf_similarity(t, job) for t in (close, partial, unrelated)]
        assert scores[0] > scores[1] > scores[2]

    def test_is_case_and_punctuation_insensitive(self) -> None:
        assert tfidf_similarity("Python, Docker!", "python docker") == pytest.approx(1.0)

    def test_stop_words_are_ignored(self) -> None:
        assert tfidf_similarity("the and of", "the and of") == 0.0
        assert tfidf_similarity("the python", "a python") == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("a", "b"), [("", ""), ("python", ""), ("   ", "python"), ("!!!", "?")]
    )
    def test_empty_inputs_score_zero(self, a: str, b: str) -> None:
        assert tfidf_similarity(a, b) == 0.0

    def test_repeated_terms_do_not_dominate(self) -> None:
        job = "python docker kubernetes"
        spam = "python " * 50 + "docker"
        balanced = "python docker"
        assert tfidf_similarity(balanced, job) > tfidf_similarity(spam, job)


class TestEmbeddingSimilarity:
    def test_uses_embedder_vectors(self) -> None:
        class Same:
            def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
                return [[1.0, 2.0] for _ in texts]

        assert embedding_similarity("a", "b", Same()) == pytest.approx(1.0)

    def test_blank_text_scores_zero_without_encoding(self) -> None:
        class Boom:
            def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
                raise AssertionError("must not be called")

        assert embedding_similarity("", "x", Boom()) == 0.0
        assert embedding_similarity("x", "  ", Boom()) == 0.0

    def test_wrong_vector_count(self) -> None:
        class One:
            def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
                return [[1.0]]

        with pytest.raises(ValueError, match="one vector per text"):
            embedding_similarity("a", "b", One())


class TestSentenceTransformerEmbedder:
    def test_encodes_with_loaded_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_library(monkeypatch)
        embedder = SentenceTransformerEmbedder("tiny", local_only=True)
        assert embedder.model_name == "tiny"
        assert embedder.encode(["ab", "abc"]) == [[2.0, 1.0], [3.0, 1.0]]

    def test_passes_local_only_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created: list[FakeModel] = []

        def factory(name: str, **kwargs: Any) -> FakeModel:
            created.append(FakeModel(name, **kwargs))
            return created[-1]

        install_fake_library(monkeypatch, factory)
        SentenceTransformerEmbedder("tiny", local_only=True)
        SentenceTransformerEmbedder("tiny")
        assert [m.kwargs for m in created] == [
            {"local_files_only": True},
            {"local_files_only": False},
        ]

    def test_missing_library(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_missing_library(monkeypatch)
        with pytest.raises(EmbeddingsUnavailableError, match=r"cv-job-matcher\[embeddings\]"):
            SentenceTransformerEmbedder()

    def test_model_load_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(name: str, **kwargs: Any) -> FakeModel:
            raise OSError("not cached")

        install_fake_library(monkeypatch, broken)
        with pytest.raises(EmbeddingsUnavailableError, match="not cached"):
            SentenceTransformerEmbedder("tiny", local_only=True)


class TestLoadLocalEmbedder:
    def test_returns_embedder_when_model_is_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_library(monkeypatch)
        assert load_local_embedder("tiny") is not None

    def test_returns_none_without_library(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_missing_library(monkeypatch)
        assert load_local_embedder() is None

    def test_returns_none_when_model_not_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(name: str, **kwargs: Any) -> FakeModel:
            raise OSError("not cached")

        install_fake_library(monkeypatch, broken)
        assert load_local_embedder() is None
