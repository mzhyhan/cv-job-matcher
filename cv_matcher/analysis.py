"""Gap analysis and CV improvement suggestions.

Two analysers implement the :class:`GapAnalyser` protocol: a deterministic
:class:`RuleBasedAnalyser` that works offline, and an
:class:`OpenAICompatibleAnalyser` that asks any OpenAI-compatible chat
completions endpoint. :func:`analyse_gap` picks one and scores the documents
if no :class:`~cv_matcher.models.MatchResult` is supplied.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol, get_args

from cv_matcher.models import (
    AnalysisError,
    AnalysisResult,
    LoadedDocument,
    MatchResult,
    ScoreWeights,
    Suggestion,
)
from cv_matcher.scoring import Backend, match_cv
from cv_matcher.similarity import DEFAULT_MODEL_NAME, Embedder

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_OPENAI_MODEL",
    "GapAnalyser",
    "OpenAICompatibleAnalyser",
    "Provider",
    "RuleBasedAnalyser",
    "Transport",
    "analyse_gap",
]

logger = logging.getLogger(__name__)

Provider = Literal["auto", "rules", "openai"]
"""Names accepted for the gap-analysis provider."""

Transport = Callable[[str, Mapping[str, str], bytes, float], bytes]
"""Callable ``(url, headers, body, timeout) -> response bytes`` used for HTTP."""

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
USER_AGENT = "cv-job-matcher/0.1.0"
MAX_SKILL_SUGGESTIONS = 10
_CATEGORIES = frozenset({"missing_skill", "emphasis", "wording"})
_RE_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)


class GapAnalyser(Protocol):
    """Anything that can turn a match into suggestions."""

    name: str

    def analyse(self, cv: str, job: str, match: MatchResult) -> AnalysisResult:
        """Analyse the gap between a CV and a job description.

        Args:
            cv: CV text.
            job: Job description text.
            match: The match already computed for these documents.

        Returns:
            The summary and suggestions.
        """
        ...


def _band(score: float) -> str:
    """Describe an overall score as strong, partial or weak."""
    if score >= 0.8:
        return "strong"
    if score >= 0.5:
        return "partial"
    return "weak"


class RuleBasedAnalyser:
    """Deterministic, offline analyser based on the match result."""

    name = "rules"

    def analyse(self, cv: str, job: str, match: MatchResult) -> AnalysisResult:
        """Build a summary and suggestions from the match.

        Args:
            cv: CV text (unused; the match already captures it).
            job: Job description text (unused).
            match: The match to analyse.

        Returns:
            An analysis with at least one suggestion.
        """
        matched, missing = match.matched_skills, match.missing_skills
        summary = (
            f"This is a {_band(match.overall_score)} match: overall {match.overall_score:.0%} "
            f"(skills {match.skill_score:.0%}, text {match.text_score:.0%}). "
            f"{len(matched)} job skill(s) matched and {len(missing)} missing."
        )
        suggestions = [
            Suggestion(
                title=f"Add evidence of {hit.name}",
                detail=(
                    f"The job description mentions {hit.name} but your CV does not. If you have "
                    f"used it, add a concrete example: what you built, and the outcome."
                ),
                skill=hit.name,
                category="missing_skill",
            )
            for hit in missing[:MAX_SKILL_SUGGESTIONS]
        ]
        if matched:
            names = ", ".join(m.name for m in matched)
            suggestions.append(
                Suggestion(
                    title="Lead with your matching skills",
                    detail=(
                        f"Put {names} near the top of your CV and quantify the impact "
                        f"(for example scale, speed or savings)."
                    ),
                    category="emphasis",
                )
            )
        if not matched and not missing:
            suggestions.append(
                Suggestion(
                    title="Echo key phrases from the job description",
                    detail=(
                        "No specific skills were detected in the job description. Reuse its key "
                        "phrases and terminology where they honestly describe your experience."
                    ),
                    category="wording",
                )
            )
        if not missing and matched:
            suggestions.append(
                Suggestion(
                    title="Keep skill evidence prominent",
                    detail="All required skills are covered; quantify outcomes to stand out.",
                    category="emphasis",
                )
            )
        return AnalysisResult(self.name, summary, tuple(suggestions), match)


def _urllib_transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> bytes:
    """Send a POST request with urllib and return the response body.

    Args:
        url: Target URL.
        headers: Request headers.
        body: Request body.
        timeout: Timeout in seconds.

    Returns:
        The raw response body.
    """
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data: bytes = response.read()
    return data


class OpenAICompatibleAnalyser:
    """Analyser backed by an OpenAI-compatible chat completions endpoint."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_OPENAI_MODEL,
        timeout: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
        """Configure the analyser.

        Args:
            api_key: Bearer token for the endpoint.
            base_url: API base URL, without the ``/chat/completions`` suffix.
            model: Model name to request.
            timeout: Request timeout in seconds.
            transport: Optional HTTP callable, mainly for tests.

        Raises:
            AnalysisError: If ``api_key`` is empty.
        """
        if not api_key or not api_key.strip():
            raise AnalysisError("An API key is required for the openai provider.")
        self._api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._transport: Transport = transport or _urllib_transport

    def _messages(self, cv: str, job: str, match: MatchResult) -> list[dict[str, str]]:
        """Build the chat messages for a request."""
        missing = ", ".join(h.name for h in match.missing_skills) or "none"
        matched = ", ".join(m.name for m in match.matched_skills) or "none"
        system = (
            "You are a careful CV reviewer. Reply with JSON only, in the form "
            '{"summary": str, "suggestions": [{"title": str, "detail": str, '
            '"skill": str|null, "category": "missing_skill"|"emphasis"|"wording"}]}. '
            "Most important suggestions first. Never invent experience."
        )
        user = (
            f"Overall match: {match.overall_score:.0%}. Matched skills: {matched}. "
            f"Missing skills: {missing}.\n\n--- CV ---\n{cv}\n\n--- JOB DESCRIPTION ---\n{job}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def analyse(self, cv: str, job: str, match: MatchResult) -> AnalysisResult:
        """Ask the model for a summary and suggestions.

        Args:
            cv: CV text.
            job: Job description text.
            match: The match to attach to the result.

        Returns:
            The parsed analysis.

        Raises:
            AnalysisError: On HTTP, JSON or schema failures.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        payload = {
            "model": self.model,
            "messages": self._messages(cv, job, match),
            "temperature": 0.2,
        }
        logger.debug("POST %s (model=%s)", url, self.model)
        try:
            raw = self._transport(url, headers, json.dumps(payload).encode("utf-8"), self.timeout)
        except urllib.error.HTTPError as exc:
            raise AnalysisError(f"HTTP error from {url}: {exc.code} {exc.reason}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise AnalysisError(f"Request to {url} failed: {exc}") from exc
        return self._parse(raw, match)

    def _parse(self, raw: bytes, match: MatchResult) -> AnalysisResult:
        """Parse a chat completions response into an analysis.

        Args:
            raw: Raw response body.
            match: The match to attach.

        Returns:
            The parsed analysis.

        Raises:
            AnalysisError: If the payload is not valid or has the wrong shape.
        """
        try:
            content = json.loads(raw)["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise AnalysisError("Unexpected response format from the API.") from exc
        if not isinstance(content, str):
            raise AnalysisError("Unexpected response format from the API.")
        text = content.strip()
        fence = _RE_FENCE.match(text)
        if fence:
            text = fence.group(1).strip()
        try:
            data: Any = json.loads(text)
        except ValueError as exc:
            raise AnalysisError("The model did not return valid JSON.") from exc
        if not isinstance(data, dict) or not isinstance(data.get("summary"), str):
            raise AnalysisError("The model response is missing a summary.")
        items = data.get("suggestions")
        if not isinstance(items, list):
            raise AnalysisError("The model response is missing a suggestions list.")
        suggestions: list[Suggestion] = []
        for item in items:
            if not isinstance(item, dict):
                raise AnalysisError("A suggestion in the model response is malformed.")
            title, detail, skill = item.get("title"), item.get("detail"), item.get("skill")
            category = item.get("category", "missing_skill")
            if not isinstance(title, str) or not isinstance(detail, str):
                raise AnalysisError("A suggestion in the model response is malformed.")
            suggestions.append(
                Suggestion(
                    title=title,
                    detail=detail,
                    skill=skill if isinstance(skill, str) and skill else None,
                    category=category if category in _CATEGORIES else "wording",
                )
            )
        return AnalysisResult(self.name, data["summary"], tuple(suggestions), match)


def _text_of(document: LoadedDocument | str) -> str:
    """Return the text of a document or string."""
    return document.text if isinstance(document, LoadedDocument) else document


def analyse_gap(
    cv: LoadedDocument | str,
    job: LoadedDocument | str,
    *,
    match: MatchResult | None = None,
    provider: Provider = "auto",
    analyser: GapAnalyser | None = None,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_OPENAI_MODEL,
    embedder: Embedder | None = None,
    backend: Backend = "auto",
    weights: ScoreWeights | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> AnalysisResult:
    """Score a CV (if needed) and produce gap analysis with suggestions.

    Args:
        cv: The CV, as a loaded document or raw text.
        job: The job description, as a loaded document or raw text.
        match: Pre-computed match; computed with :func:`match_cv` if ``None``.
        provider: ``"rules"`` is offline; ``"openai"`` needs an API key;
            ``"auto"`` uses OpenAI only when a key is available, so it never
            touches the network otherwise.
        analyser: Analyser to use instead of ``provider``.
        api_key: API key; defaults to the ``OPENAI_API_KEY`` environment variable.
        base_url: API base URL; ``OPENAI_BASE_URL`` is used if left at the default.
        model: Chat model; ``OPENAI_MODEL`` is used if left at the default.
        embedder: Embedder passed to :func:`match_cv`.
        backend: Similarity backend passed to :func:`match_cv`.
        weights: Score weights passed to :func:`match_cv`.
        model_name: Embeddings model passed to :func:`match_cv`.

    Returns:
        The analysis, including the underlying match.

    Raises:
        ValueError: If ``provider`` is unknown.
        AnalysisError: If the chosen analyser fails.
    """
    if provider not in get_args(Provider):
        raise ValueError(f"Unknown provider {provider!r}; expected one of {get_args(Provider)}.")
    if match is None:
        match = match_cv(
            cv,
            job,
            backend=backend,
            weights=weights,
            embedder=embedder,
            model_name=model_name,
        )
    chosen = analyser
    if chosen is None:
        key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        if provider == "rules" or (provider == "auto" and not key.strip()):
            chosen = RuleBasedAnalyser()
        else:
            if base_url == DEFAULT_BASE_URL:
                base_url = os.environ.get("OPENAI_BASE_URL") or base_url
            if model == DEFAULT_OPENAI_MODEL:
                model = os.environ.get("OPENAI_MODEL") or model
            chosen = OpenAICompatibleAnalyser(key, base_url=base_url, model=model)
    logger.debug("Analysing gap with provider %s", chosen.name)
    return chosen.analyse(_text_of(cv), _text_of(job), match)
