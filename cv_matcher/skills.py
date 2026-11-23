"""Skill taxonomy and extraction from CV and job-description text.

The taxonomy is a curated, software and AI focused list stored as package data
(``cv_matcher/data/skills.json``). Extraction runs in two passes:

1. Exact matching of canonical names and aliases. Matching is case-insensitive
   and token-aware, so ``Java`` does not match inside ``JavaScript``. Names that
   are also ordinary words or letters (``R``, ``Go``, ``C``) are matched
   case-sensitively with stricter boundaries.
2. Normalised and fuzzy matching of remaining word groups, which tolerates
   punctuation variants (``sci-kit learn``) and small typos (``Kubernets``).
   The fuzzy threshold is deliberately conservative.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from importlib import resources

from rapidfuzz import fuzz

from cv_matcher.models import Skill, SkillHit

__all__ = ["DEFAULT_FUZZY_THRESHOLD", "extract_skills", "load_taxonomy"]

logger = logging.getLogger(__name__)

DEFAULT_FUZZY_THRESHOLD = 88.0
"""Minimum RapidFuzz ratio (0-100) for a fuzzy skill match."""

_MIN_FUZZY_LENGTH = 7
_MAX_LENGTH_DIFFERENCE = 2
_MAX_NGRAM = 3
_ALIAS_CONFIDENCE = 0.95
_NORMALISED_CONFIDENCE = 0.9
_AMBIGUOUS_CONFIDENCE = 0.8

_RE_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#]*")
_RE_NON_ALNUM = re.compile(r"[^a-z0-9+#]")


def _squash(text: str) -> str:
    """Lower-case ``text`` and drop everything except letters, digits, ``+`` and ``#``."""
    return _RE_NON_ALNUM.sub("", text.lower())


def _phrase_pattern(phrase: str) -> str:
    """Return a regex fragment for ``phrase`` that tolerates flexible whitespace."""
    return r"\s+".join(re.escape(part) for part in phrase.split())


@lru_cache(maxsize=1)
def load_taxonomy() -> tuple[Skill, ...]:
    """Load the bundled skills taxonomy.

    The file is read through :mod:`importlib.resources`, so it works from an
    editable checkout as well as from an installed wheel.

    Returns:
        All skills in the taxonomy, in file order.

    Raises:
        ValueError: If the bundled file is malformed.
    """
    resource = resources.files("cv_matcher").joinpath("data").joinpath("skills.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    try:
        skills = tuple(
            Skill(
                name=entry["name"],
                category=entry["category"],
                aliases=tuple(entry.get("aliases", ())),
                ambiguous=bool(entry.get("ambiguous", False)),
            )
            for entry in payload["skills"]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed skills taxonomy.") from exc
    logger.debug("Loaded %d skills from the taxonomy", len(skills))
    return skills


class _Matcher:
    """Compiled lookup structures for a set of skills."""

    def __init__(self, skills: tuple[Skill, ...]) -> None:
        # form (lower-case) -> (skill, alias as written)
        self._forms: dict[str, tuple[Skill, str]] = {}
        self._ambiguous: dict[str, tuple[Skill, str]] = {}
        # squashed form -> (skill, alias as written)
        self._squashed: dict[str, tuple[Skill, str]] = {}
        for skill in skills:
            for form in (skill.name, *skill.aliases):
                # Only the bare name of an ambiguous skill needs strict matching;
                # its aliases (for example "Golang") are distinctive.
                if skill.ambiguous and form == skill.name:
                    self._ambiguous.setdefault(form, (skill, form))
                else:
                    self._forms.setdefault(form.lower(), (skill, form))
                    squashed = _squash(form)
                    if squashed:
                        self._squashed.setdefault(squashed, (skill, form))
        self._exact_re = self._compile(
            sorted(self._forms, key=len, reverse=True),
            before=r"(?<![\w+#])",
            after=r"(?![\w+#])",
            flags=re.IGNORECASE,
        )
        self._ambiguous_re = self._compile(
            sorted(self._ambiguous, key=len, reverse=True),
            before=r"(?<![\w+#.&/-])",
            after=r"(?![\w+#&/-]|\.\w)",
            flags=0,
        )
        self._fuzzy_targets = [form for form in self._squashed if len(form) >= _MIN_FUZZY_LENGTH]

    @staticmethod
    def _compile(
        forms: list[str], *, before: str, after: str, flags: int
    ) -> re.Pattern[str] | None:
        if not forms:
            return None
        body = "|".join(_phrase_pattern(form) for form in forms)
        return re.compile(f"{before}(?:{body}){after}", flags)

    def find(self, text: str, *, fuzzy: bool, threshold: float) -> list[SkillHit]:
        """Find skills in ``text`` and return one best hit per skill."""
        hits: list[tuple[int, SkillHit]] = []
        covered: list[tuple[int, int]] = []

        if self._exact_re is not None:
            for match in self._exact_re.finditer(text):
                key = " ".join(match.group(0).lower().split())
                skill, alias = self._forms[key]
                surface = match.group(0)
                if surface.lower() == skill.name.lower():
                    method, confidence = "exact", 1.0
                elif _squash(surface) == _squash(skill.name):
                    method, confidence = "normalised", _NORMALISED_CONFIDENCE
                else:
                    method, confidence = "alias", _ALIAS_CONFIDENCE
                hits.append(
                    (
                        match.start(),
                        SkillHit(skill.name, skill.category, confidence, surface, alias, method),
                    )
                )
                covered.append(match.span())

        if self._ambiguous_re is not None:
            for match in self._ambiguous_re.finditer(text):
                skill, alias = self._ambiguous[" ".join(match.group(0).split())]
                hit = SkillHit(
                    skill.name,
                    skill.category,
                    _AMBIGUOUS_CONFIDENCE,
                    match.group(0),
                    alias,
                    "exact",
                )
                hits.append((match.start(), hit))
                covered.append(match.span())

        if fuzzy:
            hits.extend(self._fuzzy_hits(text, covered, threshold))

        return self._best_per_skill(hits)

    def _fuzzy_hits(
        self, text: str, covered: list[tuple[int, int]], threshold: float
    ) -> list[tuple[int, SkillHit]]:
        tokens = list(_RE_TOKEN.finditer(text))
        found: list[tuple[int, SkillHit]] = []
        for start_idx in range(len(tokens)):
            for size in range(1, _MAX_NGRAM + 1):
                end_idx = start_idx + size
                if end_idx > len(tokens):
                    break
                first, last = tokens[start_idx], tokens[end_idx - 1]
                span = (first.start(), last.end())
                if any(span[0] < c_end and c_start < span[1] for c_start, c_end in covered):
                    continue
                surface = text[span[0] : span[1]]
                # Only join word groups separated by plain spaces or hyphens.
                if size > 1 and not re.fullmatch(r"[A-Za-z0-9+#\s-]+", surface):
                    continue
                hit = self._fuzzy_lookup(surface, threshold)
                if hit is not None:
                    found.append((span[0], hit))
        return found

    def _fuzzy_lookup(self, surface: str, threshold: float) -> SkillHit | None:
        squashed = _squash(surface)
        if len(squashed) < _MIN_FUZZY_LENGTH - _MAX_LENGTH_DIFFERENCE:
            return None
        known = self._squashed.get(squashed)
        if known is not None:
            skill, alias = known
            return SkillHit(
                skill.name, skill.category, _NORMALISED_CONFIDENCE, surface, alias, "normalised"
            )
        best_score = 0.0
        best_form = ""
        for form in self._fuzzy_targets:
            if abs(len(form) - len(squashed)) > _MAX_LENGTH_DIFFERENCE or form[0] != squashed[0]:
                continue
            score = fuzz.ratio(squashed, form)
            if score > best_score:
                best_score, best_form = score, form
        if best_score < threshold:
            return None
        skill, alias = self._squashed[best_form]
        return SkillHit(
            skill.name, skill.category, round(best_score / 100 * 0.9, 3), surface, alias, "fuzzy"
        )

    @staticmethod
    def _best_per_skill(hits: list[tuple[int, SkillHit]]) -> list[SkillHit]:
        best: dict[str, tuple[int, SkillHit]] = {}
        for position, hit in hits:
            current = best.get(hit.name)
            if current is None or (hit.confidence, -position) > (
                current[1].confidence,
                -current[0],
            ):
                best[hit.name] = (position, hit)
        return [hit for _, hit in sorted(best.values(), key=lambda item: item[0])]


@lru_cache(maxsize=1)
def _default_matcher() -> _Matcher:
    return _Matcher(load_taxonomy())


def extract_skills(
    text: str,
    *,
    fuzzy: bool = True,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    taxonomy: tuple[Skill, ...] | None = None,
) -> tuple[SkillHit, ...]:
    """Extract known skills from text.

    Each skill is reported once, using its most confident occurrence, in order
    of first appearance.

    Args:
        text: CV or job-description text. Empty text yields no hits.
        fuzzy: Also match punctuation variants and small typos.
        fuzzy_threshold: Minimum RapidFuzz ratio (0-100) for fuzzy matches.
            Only words of at least five characters are considered.
        taxonomy: Custom skills to use instead of the bundled taxonomy.

    Returns:
        The skills found, one :class:`~cv_matcher.models.SkillHit` per skill.

    Example:
        >>> [hit.name for hit in extract_skills("Built APIs in Python and Postgres")]
        ['Python', 'PostgreSQL']
    """
    if not text or not text.strip():
        return ()
    matcher = _default_matcher() if taxonomy is None else _Matcher(taxonomy)
    return tuple(matcher.find(text, fuzzy=fuzzy, threshold=fuzzy_threshold))
