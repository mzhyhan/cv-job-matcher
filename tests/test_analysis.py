from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

import pytest

from cv_matcher import (
    AnalysisError,
    AnalysisResult,
    DocumentKind,
    GapAnalyser,
    LoadedDocument,
    MatchResult,
    OpenAICompatibleAnalyser,
    RuleBasedAnalyser,
    SourceFormat,
    Suggestion,
    analyse_gap,
    match_cv,
)
from cv_matcher import analysis as analysis_module

from .conftest import SAMPLE_CV_TEXT, SAMPLE_JOB_TEXT

CV = "Python developer. Built APIs with Docker and PostgreSQL."
JOB = "We need Python, Docker, Kubernetes, Terraform and PostgreSQL experience."


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)


def _match(cv: str = CV, job: str = JOB) -> MatchResult:
    return match_cv(cv, job, backend="tfidf")


def _payload(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


GOOD = {
    "summary": "Good fit.",
    "suggestions": [
        {
            "title": "Add Kubernetes",
            "detail": "Show it.",
            "skill": "Kubernetes",
            "category": "missing_skill",
        },
        {"title": "Reword", "detail": "Tidy.", "skill": None, "category": "bogus"},
    ],
}


class Recorder:
    def __init__(self, response: bytes | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, Mapping[str, str], bytes, float]] = []

    def __call__(self, url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> bytes:
        self.calls.append((url, headers, body, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_rules_missing_skills() -> None:
    match = _match()
    result = RuleBasedAnalyser().analyse(CV, JOB, match)
    assert result.provider == "rules"
    assert result.match is match
    missing = [s for s in result.suggestions if s.category == "missing_skill"]
    assert [s.skill for s in missing] == [h.name for h in match.missing_skills]
    assert "Kubernetes" in {s.skill for s in missing}
    assert any(s.category == "emphasis" for s in result.suggestions)
    assert "partial" in result.summary or "weak" in result.summary


def test_rules_is_deterministic() -> None:
    match = _match()
    assert RuleBasedAnalyser().analyse(CV, JOB, match) == RuleBasedAnalyser().analyse(
        CV, JOB, match
    )


def test_rules_caps_missing_suggestions() -> None:
    job = (
        "Python Java Go Rust Kotlin Swift Scala Ruby PHP Docker Kubernetes Terraform "
        "PostgreSQL MySQL MongoDB Redis"
    )
    result = analyse_gap("cooking", job, provider="rules", backend="tfidf")
    assert len(result.match.missing_skills) > 10
    assert len([s for s in result.suggestions if s.category == "missing_skill"]) == 10


def test_rules_full_overlap_still_suggests() -> None:
    result = analyse_gap(JOB, JOB, provider="rules", backend="tfidf")
    assert not result.match.missing_skills
    assert "strong" in result.summary
    assert result.suggestions
    assert all(s.category == "emphasis" for s in result.suggestions)


def test_rules_no_job_skills() -> None:
    result = analyse_gap(CV, "We value kindness and curiosity.", provider="rules", backend="tfidf")
    assert not result.match.matched_skills and not result.match.missing_skills
    assert [s.category for s in result.suggestions] == ["wording"]


def test_rules_never_empty_for_blank_inputs() -> None:
    result = analyse_gap("", "", provider="rules", backend="tfidf")
    assert result.suggestions


def test_rules_matched_only_branch_has_two_suggestions() -> None:
    result = analyse_gap("Python", "Python", provider="rules", backend="tfidf")
    assert [s.title for s in result.suggestions] == [
        "Lead with your matching skills",
        "Keep skill evidence prominent",
    ]


def test_protocol_conformance() -> None:
    analysers: list[GapAnalyser] = [
        RuleBasedAnalyser(),
        OpenAICompatibleAnalyser("k", transport=Recorder(b"")),
    ]
    assert [a.name for a in analysers] == ["rules", "openai"]


def test_auto_uses_rules_without_key() -> None:
    result = analyse_gap(CV, JOB, backend="tfidf")
    assert result.provider == "rules"


def test_injected_analyser_is_used() -> None:
    class Fake:
        name = "fake"

        def analyse(self, cv: str, job: str, match: MatchResult) -> AnalysisResult:
            return AnalysisResult("fake", "s", (Suggestion("t", "d"),), match)

    result = analyse_gap(CV, JOB, analyser=Fake(), backend="tfidf")
    assert result.provider == "fake"


def test_precomputed_match_is_reused() -> None:
    match = _match()
    assert analyse_gap(CV, JOB, match=match, provider="rules").match is match


def test_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        analyse_gap(CV, JOB, provider="nope")  # type: ignore[arg-type]


def test_accepts_loaded_documents() -> None:
    cv = LoadedDocument(SAMPLE_CV_TEXT, DocumentKind.CV, SourceFormat.TXT, "cv.txt")
    job = LoadedDocument(SAMPLE_JOB_TEXT, DocumentKind.JOB_DESCRIPTION, SourceFormat.TXT, "j.txt")
    assert analyse_gap(cv, job, backend="tfidf").suggestions
    assert analyse_gap(SAMPLE_CV_TEXT, SAMPLE_JOB_TEXT, backend="tfidf").suggestions


def test_openai_success_and_request_shape() -> None:
    transport = Recorder(_payload(json.dumps(GOOD)))
    analyser = OpenAICompatibleAnalyser(
        "sk-secret", base_url="https://example.test/v1/", model="m1", timeout=5, transport=transport
    )
    match = _match()
    result = analyser.analyse(CV, JOB, match)
    url, headers, body, timeout = transport.calls[0]
    assert url == "https://example.test/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-secret"
    assert headers["User-Agent"] == "cv-job-matcher/0.1.0"
    assert timeout == 5
    sent = json.loads(body)
    assert sent["model"] == "m1"
    assert CV in sent["messages"][1]["content"]
    assert result.provider == "openai"
    assert result.match is match
    assert result.summary == "Good fit."
    assert result.suggestions[0] == Suggestion(
        "Add Kubernetes", "Show it.", "Kubernetes", "missing_skill"
    )
    assert result.suggestions[1].skill is None
    assert result.suggestions[1].category == "wording"


def test_openai_does_not_log_key(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    analyser = OpenAICompatibleAnalyser("sk-secret", transport=Recorder(_payload(json.dumps(GOOD))))
    analyser.analyse(CV, JOB, _match())
    assert "sk-secret" not in caplog.text


def test_openai_markdown_fenced_json() -> None:
    fenced = "```json\n" + json.dumps(GOOD) + "\n```"
    analyser = OpenAICompatibleAnalyser("k", transport=Recorder(_payload(fenced)))
    assert analyser.analyse(CV, JOB, _match()).summary == "Good fit."


def test_openai_missing_key() -> None:
    with pytest.raises(AnalysisError, match="API key"):
        OpenAICompatibleAnalyser("")
    with pytest.raises(AnalysisError, match="API key"):
        OpenAICompatibleAnalyser("   ")


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b"")),  # type: ignore[arg-type]
        urllib.error.URLError("down"),
        TimeoutError("slow"),
    ],
)
def test_openai_transport_errors(error: Exception) -> None:
    analyser = OpenAICompatibleAnalyser("k", transport=Recorder(error))
    with pytest.raises(AnalysisError):
        analyser.analyse(CV, JOB, _match())


def test_openai_http_error_message() -> None:
    err = urllib.error.HTTPError("u", 429, "Too Many", {}, io.BytesIO(b""))  # type: ignore[arg-type]
    analyser = OpenAICompatibleAnalyser("k", transport=Recorder(err))
    with pytest.raises(AnalysisError, match="429"):
        analyser.analyse(CV, JOB, _match())


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        json.dumps({"choices": []}).encode(),
        json.dumps({"choices": [{"message": {"content": 5}}]}).encode(),
        _payload("not json at all"),
        _payload("[1, 2]"),
        _payload(json.dumps({"summary": 1, "suggestions": []})),
        _payload(json.dumps({"summary": "s"})),
        _payload(json.dumps({"summary": "s", "suggestions": ["x"]})),
        _payload(json.dumps({"summary": "s", "suggestions": [{"title": "t"}]})),
    ],
)
def test_openai_bad_payloads(raw: bytes) -> None:
    analyser = OpenAICompatibleAnalyser("k", transport=Recorder(raw))
    with pytest.raises(AnalysisError):
        analyser.analyse(CV, JOB, _match())


def test_openai_empty_match_lists_in_prompt() -> None:
    transport = Recorder(_payload(json.dumps(GOOD)))
    OpenAICompatibleAnalyser("k", transport=transport).analyse("", "", _match("", ""))
    assert "Matched skills: none" in json.loads(transport.calls[0][2])["messages"][1]["content"]


def test_default_transport_uses_urlopen(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"ok"

    def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = analysis_module._urllib_transport("https://x.test/a", {"A": "b"}, b"{}", 3.0)
    assert out == b"ok"
    assert seen == {"url": "https://x.test/a", "method": "POST", "timeout": 3.0}


def test_analyse_gap_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "envkey")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    transport = Recorder(_payload(json.dumps(GOOD)))
    original = analysis_module._urllib_transport
    monkeypatch.setattr(analysis_module, "_urllib_transport", transport)
    try:
        result = analyse_gap(CV, JOB, backend="tfidf")
    finally:
        monkeypatch.setattr(analysis_module, "_urllib_transport", original)
    assert result.provider == "openai"
    url, headers, body, _ = transport.calls[0]
    assert url == "https://env.test/v1/chat/completions"
    assert headers["Authorization"] == "Bearer envkey"
    assert json.loads(body)["model"] == "env-model"


def test_analyse_gap_explicit_args_beat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    transport = Recorder(_payload(json.dumps(GOOD)))
    monkeypatch.setattr(analysis_module, "_urllib_transport", transport)
    analyse_gap(
        CV,
        JOB,
        provider="openai",
        api_key="k",
        base_url="https://a.test/v1",
        model="mine",
        backend="tfidf",
    )
    assert transport.calls[0][0] == "https://a.test/v1/chat/completions"
    assert json.loads(transport.calls[0][2])["model"] == "mine"


def test_analyse_gap_openai_without_key_raises() -> None:
    with pytest.raises(AnalysisError):
        analyse_gap(CV, JOB, provider="openai", backend="tfidf")


def test_rules_ignores_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert analyse_gap(CV, JOB, provider="rules", backend="tfidf").provider == "rules"
